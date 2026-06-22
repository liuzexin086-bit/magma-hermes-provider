"""
Distiller — Content distillation for MAGMA memory.

Converts raw conversation turns into structured memory notes:
  - Summarizes key facts/decisions/action items
  - Identifies entities and importance
  - Prepares content for vault storage

Edge forgetting rules:
  - Each memory note has an `importance` score (0.0 - 1.0)
  - On each new turn, unaccessed notes decay: importance *= 0.98
  - Notes below threshold (0.1) are marked for pruning
  - Accessing a note refreshes importance to 1.0
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Decay factor per sync_turn for unaccessed notes
_DECAY_FACTOR = 0.98
# Pruning threshold
_PRUNE_THRESHOLD = 0.1
# Highlight markers
_HIGHLIGHT_START = "==HIGH=="
_HIGHLIGHT_END = "==ENDHIGH=="


@dataclass
class DistillResult:
    """Result of distilling a conversation turn into a memory note."""
    title: str
    summary: str
    key_points: List[str]
    content: str
    entities: List[str]
    importance: float
    source: str


class Distiller:
    """Rules-based content distiller. No LLM dependency."""

    def distill_turn(
        self,
        user_msg: str,
        assistant_msg: str,
        context_hint: str = "",
    ) -> DistillResult:
        """
        Distill a conversation turn into structured memory.

        Uses rules and heuristics to extract key information:
        - Decisions (keywords: "决定", "选择", "用", "采用")
        - Data/numbers (patterns: X元, X头, X%)
        - Action items (keywords: "需要", "要做", "下一步")
        - Key facts (sentences with domain-specific terms)
        """
        combined = f"User: {user_msg}\n\nAssistant: {assistant_msg}"

        # Extract entities
        entities = self._extract_entities(combined)

        # Extract key points
        key_points = self._extract_key_points(user_msg, assistant_msg)

        # Generate title
        title = self._generate_title(user_msg, key_points)

        # Generate summary
        summary = self._generate_summary(user_msg, assistant_msg, key_points)

        # Calculate importance
        importance = self._calculate_importance(combined, key_points)

        # Highlight key content
        content = self._highlight_content(combined, key_points)

        return DistillResult(
            title=title,
            summary=summary,
            key_points=key_points,
            content=content,
            entities=entities,
            importance=importance,
            source="conversation_turn",
        )

    def distill_batch(self, events: List[Dict[str, Any]]) -> List[DistillResult]:
        """
        Distill a batch of existing events into consolidated notes.
        
        Groups events by source_doc and consolidates into single notes.
        """
        from collections import defaultdict

        # Group by source_doc
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for evt in events:
            src = evt.get("source_doc", evt.get("metadata", {}).get("source_doc", "unknown"))
            groups[src].append(evt)

        results = []
        for src, evts in groups.items():
            # Sort by timestamp
            evts.sort(key=lambda e: e.get("timestamp", datetime.min))

            # Combine
            combined_user = ""
            combined_assistant = ""
            for evt in evts:
                speaker = evt.get("speaker") or evt.get("metadata", {}).get("speaker", "unknown")
                content = evt.get("content", "")
                if speaker == "user":
                    combined_user += content + "\n"
                else:
                    combined_assistant += content + "\n"

            result = self.distill_turn(combined_user[:2000], combined_assistant[:2000],
                                       context_hint=src)
            # Override source
            result.source = src
            results.append(result)

        return results

    def _generate_title(self, user_msg: str, key_points: List[str]) -> str:
        """Generate a descriptive title from the user message."""
        # Take first meaningful sentence
        user_msg = user_msg.strip()
        if key_points:
            return key_points[0][:80]
        # Cut at first punctuation
        for sep in ["?", "！", "。", "；"]:
            idx = user_msg.find(sep)
            if idx > 5:
                return user_msg[:idx + 1]
        return user_msg[:80] if len(user_msg) > 5 else "Conversation note"

    def _extract_entities(self, text: str) -> List[str]:
        """Extract significant entities from text."""
        entities = []
        # Chinese project/product names
        proj_pattern = re.findall(r'奶爸机[\d.]*', text)
        entities.extend(proj_pattern)

        # Capitalized English terms (acronyms, product names)
        eng_terms = re.findall(r'\b[A-Z][A-Z0-9]{2,}\b', text)
        entities.extend(eng_terms)

        # Numbers + unit patterns (key data)
        data_patterns = re.findall(r'[\d.]+[元头只%天kgmlL℃°]', text)
        for d in data_patterns[:5]:
            entities.append(f"data:{d}")

        # Known domain tags
        domain_tags = ["母猪", "仔猪", "奶爸机", "PSY", "MSY", "NPD", "FCR",
                       "生猪", "猪价", "动保", "饲料", "obsidian", "hindsight",
                       "MAGMA", "vault", "GPT", "LLM"]
        tags_found = [t for t in domain_tags if t.lower() in text.lower()]
        entities.extend(tags_found)

        # Deduplicate and limit
        seen = set()
        result = []
        for e in entities:
            if e not in seen:
                seen.add(e)
                result.append(e)
        return result[:15]

    def _extract_key_points(self, user_msg: str, assistant_msg: str) -> List[str]:
        """Extract key points from a conversation turn."""
        points = []

        # Look for decision patterns
        decision_patterns = [
            (r'(决定|选择|采用|改用|确定|选定)\s*(.+?)[。\n]', "决策"),
            (r'(下一步|接下来|计划|安排)\s*(.+?)[。\n]', "行动"),
            (r'(核心|关键|重要|本质)\s*(.?[：:])?\s*(.+?)[。\n]', "要点"),
            (r'需要\s*(.+?)[。\n]', "需求"),
            (r'不[能要应该可以]\s*(.+?)[。\n]', "限制"),
        ]

        for pattern, ptype in decision_patterns:
            matches = re.findall(pattern, assistant_msg)
            for m in matches[:2]:
                text = m[-1].strip() if isinstance(m, tuple) else m.strip()
                if text and len(text) > 3:
                    points.append(f"[{ptype}] {text}")

        # Extract numbered/numeric data points
        data_matches = re.findall(r'(?:^|\n)\s*[-*]\s*(.+?)[。\n]', assistant_msg)
        for m in data_matches[:5]:
            m = m.strip()
            if m and len(m) > 5 and any(c.isdigit() for c in m):
                if m not in points:
                    points.append(m)

        # Extract table rows with data
        table_matches = re.findall(r'\|[^|]+\|[^|]+\|', assistant_msg)
        for m in table_matches[:3]:
            m = m.strip()
            if m and len(m) > 10:
                points.append(m)

        return points[:8]

    def _generate_summary(self, user_msg: str, assistant_msg: str,
                          key_points: List[str]) -> str:
        """Generate a concise summary (1-3 sentences)."""
        # Try first meaningful sentence of assistant
        assistant_first = assistant_msg.strip().split("\n")[0][:200]
        # Remove repeated patterns
        assistant_first = re.sub(r'^[#*>\- ]+', '', assistant_first).strip()

        if assistant_first and len(assistant_first) > 5:
            summary = assistant_first
            if key_points:
                summary += "。关键点：" + "；".join(key_points[:3])
            return summary[:300]

        return user_msg[:200]

    def _calculate_importance(self, text: str, key_points: List[str]) -> float:
        """Calculate importance score (0.0-1.0)."""
        score = 0.3  # baseline

        # Boost for decision content
        decision_indicators = ["决定", "选择", "采用", "计划", "确认",
                               "核心", "关键", "重要", "结论"]
        for ind in decision_indicators:
            if ind in text:
                score += 0.1

        # Boost for data-rich content
        data_count = len(re.findall(r'\d+[\.\d]*', text))
        if data_count > 5:
            score += 0.2
        if data_count > 20:
            score += 0.1

        # Boost for technical depth
        tech_terms = ["方案", "设计", "架构", "算法", "协议", "配置",
                      "代码", "实现", "部署", "测试"]
        for t in tech_terms:
            if t in text:
                score += 0.05

        # Boost for project/action oriented
        if any(kw in text for kw in ["下一步", "需要", "计划", "安排"]):
            score += 0.1

        # Context length boost
        if len(text) > 1000:
            score += 0.1

        return min(score, 1.0)

    def _highlight_content(self, content: str, key_points: List[str]) -> str:
        """Add highlight markers around key content."""
        if not key_points:
            return content

        # Highlight key points in the content
        for kp in key_points:
            # Only highlight substantial key points
            if len(kp) < 10:
                continue
            search = kp[:60]
            if search in content:
                content = content.replace(search, f"**{search}**", 1)

        return content

    # ---- Decay / Forgetting ---------------------------------------------

    def apply_decay(self, current_importance: float, turns_since_access: int = 1) -> float:
        """Apply exponential decay to importance."""
        return current_importance * (_DECAY_FACTOR ** turns_since_access)

    def should_prune(self, importance: float) -> bool:
        """Check if a note should be pruned."""
        return importance < _PRUNE_THRESHOLD


# Export constants
DECAY_FACTOR = _DECAY_FACTOR
PRUNE_THRESHOLD = _PRUNE_THRESHOLD
