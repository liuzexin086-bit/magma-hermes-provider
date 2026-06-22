"""
Note Store — Vault-backed persistent note storage for MAGMA memory.

Writes distilled memory content as Markdown files to:
  E:\\obsidian_hermes\\hermes\\magma\\

Each note contains:
  - YAML frontmatter (title, entities, importance, source, created_at)
  - Key content with highlighted sections
  - Summary
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

# Default vault paths
_DEFAULT_VAULT = Path("E:/obsidian_hermes/hermes/magma")
_INDEX_FILE = "index.json"


class NoteStore:
    """Manages memory notes in the Obsidian vault."""

    def __init__(self, vault_dir: Optional[str] = None):
        self.vault_dir = Path(vault_dir or _DEFAULT_VAULT)
        self.vault_dir.mkdir(parents=True, exist_ok=True)

    # ---- Write -----------------------------------------------------------

    def write_note(
        self,
        title: str,
        summary: str,
        key_points: List[str],
        content: str,
        entities: List[str],
        importance: float = 1.0,
        source: str = "",
        conversation_turns: Optional[List[Tuple[str, str]]] = None,
    ) -> Tuple[str, str]:
        """
        Write a memory note to the vault.

        Returns: (vault_path, note_id)
        """
        ts = datetime.now()
        # Sanitize title for filename
        safe_title = re.sub(r'[\\/*?:"<>|]', "_", title)[:60]
        date_str = ts.strftime("%Y%m%d")
        time_str = ts.strftime("%H%M%S")
        filename = f"{date_str}_{time_str}_{safe_title}.md"
        filepath = self.vault_dir / filename

        # Build Markdown content
        md = self._build_markdown(
            title=title,
            summary=summary,
            key_points=key_points,
            content=content,
            entities=entities,
            importance=importance,
            source=source,
            timestamp=ts,
            filename=filename,
        )

        filepath.write_text(md, encoding="utf-8")
        logger.info("Wrote note: %s", filepath)

        # Update index
        note_id = filename.replace(".md", "")
        self._update_index(note_id, {
            "title": title,
            "summary": summary,
            "entities": entities,
            "importance": importance,
            "created_at": ts.isoformat(),
            "source": source,
            "filename": filename,
        })

        return str(filepath), note_id

    def _build_markdown(
        self,
        title: str,
        summary: str,
        key_points: List[str],
        content: str,
        entities: List[str],
        importance: float,
        source: str,
        timestamp: datetime,
        filename: str,
    ) -> str:
        """Build the full .md content with frontmatter."""
        # Format key points as bullet list with bold for key items
        kp_section = ""
        if key_points:
            kp_lines = []
            for kp in key_points:
                # Key points get markdown highlight
                kp_lines.append(f"- **{kp}**" if kp else "")
            kp_section = "\n".join(kp_lines)

        ts_str = timestamp.strftime("%Y-%m-%d %H:%M")

        md = f"""---
title: "{title}"
created: {ts_str}
entities: [{', '.join(f'"{e}"' for e in entities[:10])}]
importance: {importance:.2f}
source: "{source}"
note_id: "{filename.replace('.md', '')}"
---

# {title}

> {summary}

---

## 📌 关键点

{kp_section}

---

## 📝 详细内容

{content}
"""
        return md

    # ---- Read ------------------------------------------------------------

    def read_note(self, note_id_or_query: str) -> Optional[str]:
        """Read a note by its ID (filename without .md) or search by title/entities."""
        # Direct file lookup
        fpath = (self.vault_dir / f"{note_id_or_query}.md")
        if fpath.exists():
            return fpath.read_text(encoding="utf-8")

        # Try as full path
        fpath = Path(note_id_or_query)
        if fpath.exists() and fpath.suffix == ".md":
            return fpath.read_text(encoding="utf-8")

        # Search by partial name
        for f in sorted(self.vault_dir.glob("*.md"), reverse=True):
            if note_id_or_query.lower() in f.stem.lower():
                return f.read_text(encoding="utf-8")

        # Search in index
        index = self._load_index()
        for nid, meta in index.items():
            title = meta.get("title", "")
            summary = meta.get("summary", "")
            entities = " ".join(meta.get("entities", []))
            if (note_id_or_query.lower() in title.lower()
                    or note_id_or_query.lower() in summary.lower()
                    or note_id_or_query.lower() in entities.lower()):
                fpath = self.vault_dir / meta.get("filename", f"{nid}.md")
                if fpath.exists():
                    return fpath.read_text(encoding="utf-8")

        return None

    def search_notes(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search the index by title/summary/entities. Returns metadata list."""
        index = self._load_index()
        query_lower = query.lower()
        results = []

        for nid, meta in index.items():
            title = meta.get("title", "")
            summary = meta.get("summary", "")
            entities = " ".join(meta.get("entities", []))

            score = 0
            if query_lower in title.lower():
                score += 5
            if query_lower in summary.lower():
                score += 3
            if query_lower in entities.lower():
                score += 2

            if score > 0:
                results.append({
                    "note_id": nid,
                    "title": title,
                    "summary": summary,
                    "entities": meta.get("entities", []),
                    "importance": meta.get("importance", 1.0),
                    "created_at": meta.get("created_at", ""),
                    "filename": meta.get("filename", f"{nid}.md"),
                    "score": score,
                })

        results.sort(key=lambda x: (-x["score"], -x.get("importance", 1)))
        return results[:max_results]

    # ---- Index -----------------------------------------------------------

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
            json.dumps(index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def rebuild_index(self) -> int:
        """Scan vault directory and rebuild index from .md frontmatter."""
        index = {}
        for f in sorted(self.vault_dir.glob("*.md")):
            if f.name == _INDEX_FILE:
                continue
            content = f.read_text(encoding="utf-8")
            meta = self._parse_frontmatter(content)
            if meta:
                note_id = f.stem
                meta["filename"] = f.name
                index[note_id] = meta
        self._index_path().write_text(
            json.dumps(index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Rebuilt index: %d notes", len(index))
        return len(index)

    def _parse_frontmatter(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract YAML-like frontmatter from .md content."""
        if not content.startswith("---"):
            # Try to create minimal metadata
            return None
        end = content.find("---", 3)
        if end < 0:
            return None
        fm_block = content[3:end].strip()
        meta = {}
        for line in fm_block.split("\n"):
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key == "entities":
                try:
                    # Parse ["a", "b"] list format
                    import ast
                    val = ast.literal_eval(val)
                except Exception:
                    val = [v.strip().strip('"').strip("'") for v in val.split(",")]
            elif key == "importance":
                try:
                    val = float(val)
                except Exception:
                    val = 1.0
            meta[key] = val
        return meta

    def note_count(self) -> int:
        return len(self._load_index())

    def all_summaries(self) -> List[Dict[str, Any]]:
        """Return compact summary entries for MAGMA graph."""
        index = self._load_index()
        results = []
        for nid, meta in index.items():
            results.append({
                "note_id": nid,
                "title": meta.get("title", ""),
                "summary": meta.get("summary", "")[:200],
                "entities": meta.get("entities", []),
                "importance": meta.get("importance", 1.0),
                "created_at": meta.get("created_at", ""),
                "filename": meta.get("filename", f"{nid}.md"),
            })
        results.sort(key=lambda x: -x["importance"])
        return results
