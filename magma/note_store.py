"""
Note Store — Vault-backed wiki memory for MAGMA.

Manages consolidated wiki docs in the Obsidian vault:
  E:\\obsidian_hermes\\hermes\\magma\\

Instead of creating a new file per conversation turn, new content is
classified into one of 6-8 category docs and appended to the relevant
section, preserving timestamps and decision lineage.

Each wiki doc contains:
  - Sections with timestamps (e.g., "### 决策名称（2026-06-21）")
  - Timeline table at the end: ⏳ 决策演进
  - [[Wiki links]] to related docs
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_VAULT = Path("E:/obsidian_hermes/hermes/magma")
_INDEX_FILE = "index.json"

WIKI_CATEGORIES = {
    "curve":       "奶爸机-曲线设计.md",
    "system":      "奶爸机-系统架构.md",
    "magma":       "MAGMA-记忆架构.md",
    "hermes":      "Hermes-配置工具.md",
    "pig_cycle":   "生猪行业-动保.md",
    "automation":  "养殖自动化-产品哲学.md",
}

WIKI_KEYWORDS = {
    "curve":       ["曲线", "教槽", "腹泻", "旋钮", "体重", "奶粉", "爬坡",
                    "平台", "下降速度", "降太快", "FCR", "TW", "BUDGET", "creepFactor"],
    "system":      ["parameters.py", "learning.py", "PyInstaller", "Flask",
                    "数据采集", "后端", "MCU", "预览工具", "硬件"],
    "magma":       ["MAGMA", "记忆", "vault", "index.json", "词策略",
                    "信噪比", "蒸馏", "索引", "遗忘"],
    "hermes":      ["Hermes", "Desktop GUI", "config", "profile",
                    "sensenova", "deepseek", "启动延迟", "electron"],
    "pig_cycle":   ["猪周期", "猪价", "底部磨底", "去化", "GEBV",
                    "生猪", "日报", "动保", "早报", "储备肉"],
    "automation":  ["自动化", "卖点", "ROI", "饲养员", "解放",
                    "产品哲学", "用户画像", "支付能力"],
}


class NoteStore:

    def __init__(self, vault_dir: Optional[str] = None):
        self.vault_dir = Path(vault_dir or _DEFAULT_VAULT)
        self.vault_dir.mkdir(parents=True, exist_ok=True)

    # ---- Classification ---------------------------------------------------

    def classify_content(self, text: str) -> str:
        text_lower = text.lower()
        scores = {}
        for cat, kws in WIKI_KEYWORDS.items():
            score = sum(1 for kw in kws if kw.lower() in text_lower)
            if score > 0:
                scores[cat] = score
        if not scores:
            return WIKI_CATEGORIES["automation"]
        best_cat = max(scores, key=scores.get)
        return WIKI_CATEGORIES[best_cat]

    # ---- Wiki Append ------------------------------------------------------

    def append_to_wiki(
        self,
        title: str,
        summary: str,
        content: str,
        entities: List[str],
        importance: float = 1.0,
        source: str = "",
    ) -> str:
        category_doc = self.classify_content(content)
        wiki_path = self.vault_dir / category_doc
        ts = datetime.now()
        date_str = ts.strftime("%Y-%m-%d")

        section = (
            f"\n\n### {title}（{date_str}）\n\n"
            f"> {summary}\n\n"
            f"{content}"
        )

        if wiki_path.exists():
            existing = wiki_path.read_text(encoding="utf-8")
            tl_marker = "\n## ⏳ 决策演进"
            if tl_marker in existing:
                existing = existing.replace(tl_marker, section + "\n\n" + tl_marker)
            else:
                existing += section
            wiki_path.write_text(existing, encoding="utf-8")
        else:
            md = self._build_wiki_doc(
                title=category_doc.replace(".md", ""),
                sections=[(title, date_str, summary, content)],
                timeline=[(date_str, title, source or "new_entry")],
            )
            wiki_path.write_text(md, encoding="utf-8")

        logger.info("Appended to wiki: %s", wiki_path)

        note_id = f"{ts.strftime('%Y%m%d_%H%M%S')}_{title[:40]}"
        self._update_index(note_id, {
            "title": title,
            "summary": summary,
            "entities": entities,
            "importance": importance,
            "created_at": ts.isoformat(),
            "source": source,
            "filename": category_doc,
        })
        return str(wiki_path)

    def _build_wiki_doc(
        self,
        title: str,
        sections: List[Tuple[str, str, str, str]],
        timeline: List[Tuple[str, str, str]],
    ) -> str:
        parts = [f"# {title}\n"]
        for sec_title, sec_date, sec_summary, sec_content in sections:
            parts.append(f"### {sec_title}（{sec_date}）\n")
            if sec_summary:
                parts.append(f"> {sec_summary}\n")
            parts.append(f"{sec_content}\n")
        if timeline:
            parts.append("\n## ⏳ 决策演进\n\n")
            parts.append("| 时间 | 事件 | 影响 |\n")
            parts.append("|------|------|------|\n")
            for t_date, t_event, t_impact in timeline:
                parts.append(f"| {t_date} | {t_event} | {t_impact} |\n")
        return "\n".join(parts)

    def write_note(self, title, summary, key_points, content, entities,
                   importance=1.0, source="", conversation_turns=None):
        vault_path = self.append_to_wiki(
            title=title, summary=summary,
            content="\n".join(key_points) if key_points else content,
            entities=entities, importance=importance, source=source,
        )
        note_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_escaped_note"
        return vault_path, note_id

    # ---- Read -------------------------------------------------------------

    def read_note(self, query: str) -> Optional[str]:
        fpath = Path(query)
        if fpath.exists() and fpath.suffix == ".md":
            return fpath.read_text(encoding="utf-8")
        fpath = self.vault_dir / f"{query}.md"
        if fpath.exists():
            return fpath.read_text(encoding="utf-8")

        ql = query.lower()
        for wiki_file in self.vault_dir.glob("*.md"):
            if wiki_file.name == _INDEX_FILE:
                continue
            content = wiki_file.read_text(encoding="utf-8")
            if ql in content.lower():
                return self._extract_section(content, ql)

        index = self._load_index()
        for nid, meta in index.items():
            title = meta.get("title", "")
            summary = meta.get("summary", "")
            entities = " ".join(meta.get("entities", []))
            if ql in title.lower() or ql in summary.lower() or ql in entities.lower():
                wiki_file = self.vault_dir / meta.get("filename", "")
                if wiki_file.exists():
                    return wiki_file.read_text(encoding="utf-8")
        return None

    def _extract_section(self, content: str, query: str) -> str:
        lines = content.split("\n")
        indices = [i for i, l in enumerate(lines) if query in l.lower()]
        if not indices:
            return content[:2000]
        idx = indices[0]
        start = max(0, idx - 10)
        end = min(len(lines), idx + 20)
        return "\n".join(lines[start:end])

    def search_notes(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        ql = query.lower()
        results = []
        for wiki_file in sorted(self.vault_dir.glob("*.md")):
            if wiki_file.name == _INDEX_FILE:
                continue
            content = wiki_file.read_text(encoding="utf-8")
            lines = content.split("\n")
            section = ""
            for line in lines:
                if line.startswith("### ") or line.startswith("## "):
                    section = line.strip("# ").strip()
                if ql in line.lower():
                    results.append({
                        "note_id": wiki_file.stem,
                        "title": wiki_file.stem,
                        "summary": line.strip()[:200],
                        "entities": [section] if section else [],
                        "importance": 1.0,
                        "created_at": "",
                        "filename": wiki_file.name,
                        "score": 5 if line.startswith("#") else 3,
                    })
                    if len(results) >= max_results * 2:
                        break

        index = self._load_index()
        for nid, meta in index.items():
            title = meta.get("title", "")
            summary = meta.get("summary", "")
            if ql in title.lower() or ql in summary.lower():
                results.append({
                    "note_id": nid, "title": title, "summary": summary[:200],
                    "entities": meta.get("entities", []),
                    "importance": meta.get("importance", 1.0),
                    "created_at": meta.get("created_at", ""),
                    "filename": meta.get("filename", ""), "score": 10,
                })

        results.sort(key=lambda x: (-x["score"], -x.get("importance", 1)))
        return results[:max_results]

    # ---- Index -------------------------------------------------------------

    def _index_path(self) -> Path:
        return self.vault_dir / _INDEX_FILE

    def _load_index(self) -> Dict[str, Any]:
        p = self._index_path()
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _update_index(self, note_id: str, meta: Dict[str, Any]) -> None:
        index = self._load_index()
        index[note_id] = meta
        self._index_path().write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    def rebuild_index(self) -> int:
        index = {}
        for f in sorted(self.vault_dir.glob("*.md")):
            if f.name == _INDEX_FILE:
                continue
            index[f.stem] = {
                "title": f.stem, "summary": "", "entities": [],
                "importance": 1.0,
                "created_at": datetime.now().isoformat(),
                "source": "wiki_doc", "filename": f.name,
            }
        self._index_path().write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Rebuilt wiki index: %d docs", len(index))
        return len(index)

    def note_count(self) -> int:
        return sum(1 for f in self.vault_dir.glob("*.md") if f.name != _INDEX_FILE)

    def all_summaries(self) -> List[Dict[str, Any]]:
        index = self._load_index()
        results = []
        for nid, meta in index.items():
            results.append({
                "note_id": nid, "title": meta.get("title", ""),
                "summary": meta.get("summary", "")[:200],
                "entities": meta.get("entities", []),
                "importance": meta.get("importance", 1.0),
                "created_at": meta.get("created_at", ""),
                "filename": meta.get("filename", f"{nid}.md"),
            })
        results.sort(key=lambda x: -x["importance"])
        return results
