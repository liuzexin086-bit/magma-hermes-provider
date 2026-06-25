"""
Note Store — Vault-backed wiki memory for MAGMA.

Two-level structure:
  vault/
    Domain_Folder/
      Knowledge_File.md    ← tree accumulates via timestamped sections

Each turn's knowledge is classified into (domain, file) and appended.
New domains and files auto-create when no match exists.

Classification:
  1. Match content against WIKI_KEYWORDS → find best category
  2. Category resolves to (domain_folder, filename)
  3. If no match → auto-extract domain+topic → create new folder & file
  4. Domain matched but no exact file → create new file within domain
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_VAULT = Path("E:/obsidian_hermes/hermes/magma")
_INDEX_FILE = "index.json"
_ARCHIVE_DIR = "archive"

# (domain_folder, filename) — the two-level wiki structure
WIKI_CATEGORIES: Dict[str, Tuple[str, str]] = {
    "curve":       ("奶爸机",   "曲线设计.md"),
    "system":      ("奶爸机",   "系统架构.md"),
    "magma":       ("Hermes",   "MAGMA-架构.md"),
    "hermes":      ("Hermes",   "配置工具.md"),
    "pig_cycle":   ("生猪行业", "动保研究.md"),
    "automation":  ("养殖自动化", "产品哲学.md"),
}

WIKI_KEYWORDS: Dict[str, List[str]] = {
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

# Domain-level keywords — used when no category matches, to find the right domain
DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "奶爸机":     ["奶爸机", "SmartMilk", "奶粉", "教槽", "仔猪", "曲线"],
    "生猪行业":   ["生猪", "猪价", "动保", "猪周期", "母猪", "养殖"],
    "养殖自动化": ["自动化", "ROI", "饲养员", "产品哲学", "智能养殖"],
    "Hermes":     ["Hermes", "MAGMA", "插件", "memory", "gateway", "weixin"],
}


class NoteStore:

    def __init__(self, vault_dir: Optional[str] = None):
        self.vault_dir = Path(vault_dir or _DEFAULT_VAULT)
        self.vault_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================================
    #  CLASSIFICATION  — returns (domain_folder, filename, is_new)
    # =====================================================================

    def classify_content(self, text: str, title_hint: str = "") -> Tuple[str, str, bool]:
        """Classify content into (domain_folder, filename, is_new)."""
        text_lower = text.lower()
        scores = {}
        for cat, kws in WIKI_KEYWORDS.items():
            score = sum(1 for kw in kws if kw.lower() in text_lower)
            if score > 0:
                scores[cat] = score

        if not scores:
            # No existing category matches → auto-create domain + file
            domain, filename = self._extract_domain_and_topic(text, title_hint)
            # Use sanitized topic as key (prefer first meaningful word)
            stem = Path(filename).stem
            key_part = ''.join(c for c in stem if c.isascii() and c.isalnum())[:24]
            if not key_part:
                key_part = f"auto_{len(WIKI_CATEGORIES)}"
            new_key = key_part[:32]
            # Ensure unique key
            while new_key in WIKI_CATEGORIES:
                new_key = f"{key_part[:28]}_{len(WIKI_CATEGORIES)}"
            WIKI_CATEGORIES[new_key] = (domain, filename)
            WIKI_KEYWORDS[new_key] = [stem] + text.split()[:3]
            logger.info("Auto-created category: %s → %s/%s", new_key, domain, filename)
            return domain, filename, True

        best_cat = max(scores, key=scores.get)
        domain, filename = WIKI_CATEGORIES[best_cat]
        return domain, filename, False

    def classify_domain(self, text: str) -> Optional[str]:
        """If content doesn't match a full category, at least classify its domain."""
        text_lower = text.lower()
        scores = {}
        for domain, kws in DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in kws if kw.lower() in text_lower)
            if score > 0:
                scores[domain] = score
        if scores:
            return max(scores, key=scores.get)
        return None

    def _extract_domain_and_topic(self, text: str, title_hint: str = "") -> Tuple[str, str]:
        """Guess a domain folder and filename for unmatched content."""
        domain = self.classify_domain(text)
        # Prefer title_hint for the filename — it's usually the most descriptive
        if title_hint and len(title_hint) >= 4:
            topic = self._clean_topic(title_hint)
        else:
            topic = self._extract_topic(text)

        if domain and topic:
            return domain, f"{topic}.md"
        if domain:
            return domain, f"{topic}.md"
        # No domain match either — use generic
        return "其他", f"{topic}.md"

    @staticmethod
    def _extract_topic(text: str) -> str:
        """Extract a topic name for the filename from unmatched content."""
        # Prefer Chinese noun phrases
        zh_matches = re.findall(r'[\u4e00-\u9fff]{2,12}', text)
        if zh_matches:
            best = max(zh_matches, key=len)[:20]
            return best.replace(' ', '')

        # Skip leading Chinese digits for cleaner titles
        # Fallback: English words
        words = [w for w in text.split() if len(w) > 2 and w.isalpha()][:3]
        if words:
            return '-'.join(words)
        return "新分类"

    @staticmethod
    def _clean_topic(title: str) -> str:
        """Clean title text for use as a filename."""
        # Remove common suffixes: （2026-06-25）, [要点], [决策], etc.
        cleaned = re.sub(r'[（(][^）)]*[）)]', '', title)
        cleaned = re.sub(r'^\[.*?\]\s*', '', cleaned)
        # Take first meaningful chunk, limit to 20 chars
        zh_matches = re.findall(r'[\u4e00-\u9fff]{2,12}', cleaned)
        if zh_matches:
            best = max(zh_matches, key=len)[:20]
            return best.replace(' ', '')
        # Fallback: first alphanumeric word
        words = [w for w in cleaned.split() if len(w) > 2 and w.isalpha()][:3]
        if words:
            return '-'.join(words)
        return "新分类"

    # =====================================================================
    #  WIKI APPEND — append to the correct (domain/file), auto-create
    # =====================================================================

    def append_to_wiki(
        self,
        title: str,
        summary: str,
        content: str,
        entities: List[str],
        importance: float = 1.0,
        source: str = "",
    ) -> str:
        domain, filename, is_new = self.classify_content(content, title_hint=title)
        wiki_path = self.vault_dir / domain / filename
        wiki_path.parent.mkdir(parents=True, exist_ok=True)

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
                title=filename.replace(".md", ""),
                sections=[(title, date_str, summary, content)],
                timeline=[(date_str, title, source or "new_entry")],
            )
            wiki_path.write_text(md, encoding="utf-8")

        logger.info("Appended to wiki: %s", wiki_path)

        # Index entry with both domain + filename
        note_id = f"{ts.strftime('%Y%m%d_%H%M%S')}_{title[:40]}"
        self._update_index(note_id, {
            "title": title,
            "summary": summary,
            "entities": entities,
            "importance": importance,
            "created_at": ts.isoformat(),
            "source": source,
            "domain": domain,
            "filename": filename,
            "fullpath": f"{domain}/{filename}",  # e.g. "奶爸机/曲线设计.md"
        })
        return str(wiki_path)

    def write_note(self, title, summary, key_points, content, entities,
                   importance=1.0, source="", conversation_turns=None):
        vault_path = self.append_to_wiki(
            title=title, summary=summary,
            content="\n".join(key_points) if key_points else content,
            entities=entities, importance=importance, source=source,
        )
        note_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_escaped_note"
        return vault_path, note_id

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

    # =====================================================================
    #  READ — recursive search through domain folders
    # =====================================================================

    def read_note(self, query: str) -> Optional[str]:
        """Read a note by path, keyword, or content search (recursive)."""
        fpath = Path(query)
        if fpath.exists() and fpath.suffix == ".md":
            return fpath.read_text(encoding="utf-8")

        # Direct path under vault: domain/filename.md
        candidate = self.vault_dir / f"{query}.md"
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
        candidate = self.vault_dir / query
        if candidate.exists() and candidate.suffix == ".md":
            return candidate.read_text(encoding="utf-8")

        ql = query.lower()
        # Search recursively through all domain folders
        for wiki_file in self._all_wiki_files():
            content = wiki_file.read_text(encoding="utf-8")
            if ql in content.lower():
                return self._extract_section(content, ql)

        # Last resort: index lookup
        index = self._load_index()
        for nid, meta in index.items():
            title = meta.get("title", "")
            summary = meta.get("summary", "")
            entities = " ".join(meta.get("entities", []))
            if ql in title.lower() or ql in summary.lower() or ql in entities.lower():
                domain = meta.get("domain", "")
                fname = meta.get("filename", "")
                if domain:
                    wiki_file = self.vault_dir / domain / fname
                else:
                    wiki_file = self.vault_dir / fname
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

    # =====================================================================
    #  SEARCH — recursive, index-first
    # =====================================================================

    def search_notes(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        ql = query.lower()
        index = self._load_index()
        results = []

        # Index-first: score by title/summary/entities
        for nid, meta in index.items():
            title = meta.get("title", "")
            summary = meta.get("summary", "")
            entities = " ".join(meta.get("entities", []))
            section = meta.get("section", "")

            score = 0
            if ql in title.lower():
                score += 5
            if ql in summary.lower():
                score += 3
            if ql in entities.lower():
                score += 2

            if score > 0:
                domain = meta.get("domain", "")
                fname = meta.get("filename", "")
                results.append({
                    "note_id": nid,
                    "title": title,
                    "summary": summary[:200],
                    "entities": meta.get("entities", []),
                    "importance": meta.get("importance", 1.0),
                    "created_at": meta.get("created_at", ""),
                    "domain": domain,
                    "filename": fname,
                    "section": section,
                    "score": score,
                })

        # Fallback: full-text scan if index has nothing
        if not results:
            for wiki_file in sorted(self._all_wiki_files()):
                content = wiki_file.read_text(encoding="utf-8")
                for line in content.split("\n"):
                    if ql in line.lower():
                        rel_path = wiki_file.relative_to(self.vault_dir)
                        results.append({
                            "note_id": wiki_file.stem,
                            "title": wiki_file.stem,
                            "summary": line.strip()[:200],
                            "entities": [],
                            "importance": 1.0,
                            "created_at": "",
                            "domain": str(rel_path.parent) if rel_path.parent != "." else "",
                            "filename": wiki_file.name,
                            "section": "",
                            "score": 3,
                        })
                        break

        results.sort(key=lambda x: (-x["score"], -x.get("importance", 1)))
        return results[:max_results]

    # =====================================================================
    #  INDEX
    # =====================================================================

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
        self._index_path().write_text(
            json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    def rebuild_index(self) -> int:
        """Scan wiki docs recursively and build section-level index."""
        index = {}
        for f in sorted(self._all_wiki_files()):
            content = f.read_text(encoding="utf-8")
            lines = content.split("\n")
            rel_path = f.relative_to(self.vault_dir)
            domain = str(rel_path.parent) if rel_path.parent != "." else ""

            current_section = ""
            current_summary = ""
            current_entities = []
            collector = []

            for line in lines:
                if line.startswith("### "):
                    if current_section and current_summary:
                        sid = f"{f.stem}_{current_section[:30]}"
                        index[sid] = {
                            "title": current_section,
                            "summary": current_summary[:200],
                            "entities": current_entities[:10],
                            "importance": 1.0,
                            "created_at": datetime.now().isoformat(),
                            "source": "wiki_doc",
                            "domain": domain,
                            "filename": f.name,
                            "section": current_section,
                        }
                    current_section = line.strip("# ").strip()
                    current_summary = ""
                    current_entities = []
                    collector = []
                elif line.startswith("> ") and current_section and not current_summary:
                    current_summary = line.strip("> ").strip()
                elif current_section and not current_summary and line.strip():
                    current_summary = line.strip()[:200]

                if current_section and line.strip():
                    for word in re.findall(r'\b[A-Z][A-Z0-9]{2,}\b', line):
                        if word not in current_entities:
                            current_entities.append(word)
                    for word in re.findall(r'[\d.]+[元头只%天kgmlL℃]', line):
                        if word not in current_entities:
                            current_entities.append(word)

            if current_section and current_summary:
                sid = f"{f.stem}_{current_section[:30]}"
                index[sid] = {
                    "title": current_section,
                    "summary": current_summary[:200],
                    "entities": current_entities[:10],
                    "importance": 1.0,
                    "created_at": datetime.now().isoformat(),
                    "source": "wiki_doc",
                    "domain": domain,
                    "filename": f.name,
                    "section": current_section,
                }

            index[f.stem] = {
                "title": f.stem, "summary": "", "entities": [],
                "importance": 1.0,
                "created_at": datetime.now().isoformat(),
                "source": "wiki_doc",
                "domain": domain,
                "filename": f.name,
                "section": "",
            }

        self._index_path().write_text(
            json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Rebuilt section-level index: %d entries", len(index))
        return len(index)

    # =====================================================================
    #  HELPERS
    # =====================================================================

    def _all_wiki_files(self) -> List[Path]:
        """Return all .md wiki files recursively, excluding index."""
        files = []
        for f in self.vault_dir.rglob("*.md"):
            if f.name == _INDEX_FILE:
                continue
            # Skip files inside archive/
            if _ARCHIVE_DIR in f.parts:
                continue
            files.append(f)
        return files

    def note_count(self) -> int:
        return len(self._all_wiki_files())

    def all_summaries(self) -> List[Dict[str, Any]]:
        index = self._load_index()
        results = []
        for nid, meta in index.items():
            domain = meta.get("domain", "")
            fname = meta.get("filename", "")
            results.append({
                "note_id": nid,
                "title": meta.get("title", ""),
                "summary": meta.get("summary", "")[:200],
                "entities": meta.get("entities", []),
                "importance": meta.get("importance", 1.0),
                "created_at": meta.get("created_at", ""),
                "domain": domain,
                "filename": fname,
            })
        results.sort(key=lambda x: -x["importance"])
        return results