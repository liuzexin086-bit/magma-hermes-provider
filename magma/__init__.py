"""
MAGMA Memory Provider — Two-Layer Memory: Index + Vault

Architecture:
  - MAGMA graph: lightweight index (summary + entities + vault_path)
  - Obsidian vault: full content as .md files in E:\\obsidian_hermes\\hermes\\magma\\

Flow:
  sync_turn → distill → write vault.md → store summary+path in MAGMA
  prefetch  → return summaries + vault paths (lightweight)
  on demand → magma_read_note() reads full vault .md
  forgetting → edge weights decay each turn, pruned below threshold
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

from .trg_memory import TRGMemory
from .graph_db import TraversalConstraints
from .note_store import NoteStore
from .distiller import Distiller

logger = logging.getLogger(__name__)

_PROVIDER_NAME = "magma"

# Default vault path
_VAULT_DIR = "E:/obsidian_hermes/hermes/magma"


def _load_config(hermes_home: Path) -> dict:
    config_path = hermes_home / "magma_config.json"
    defaults = {
        "persist_dir": str(hermes_home / "magma"),
        "embedding": "minilm",
        "max_events": 10000,
        "enable_consolidation": True,
        "vault_dir": _VAULT_DIR,
    }
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            defaults.update(file_cfg)
        except Exception as e:
            logger.warning("Failed to load magma config: %s", e)
    return defaults


class MagmaMemoryProvider(MemoryProvider):

    @property
    def name(self) -> str:
        return _PROVIDER_NAME

    # -- Lifecycle ---------------------------------------------------------

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        hermes_home = kwargs.get("hermes_home")
        if hermes_home:
            hermes_home = Path(hermes_home)
        else:
            from hermes_constants import get_hermes_home
            hermes_home = get_hermes_home()

        config = _load_config(hermes_home)

        persist_dir = config.get("persist_dir", str(hermes_home / "magma"))
        embedding = config.get("embedding", "minilm")
        vault_dir = config.get("vault_dir", _VAULT_DIR)

        self._memory = TRGMemory(
            persist_dir=persist_dir,
            embedding_model=embedding,
            enable_consolidation=config.get("enable_consolidation", True),
        )
        self._max_events = config.get("max_events", 10000)
        self._persist_dir = Path(persist_dir)
        self._session_id = session_id
        self._turn_count = 0

        # NoteStore for vault
        self._note_store = NoteStore(vault_dir=vault_dir)

        # Distiller
        self._distiller = Distiller()

        # Attach note_store to TRGMemory for status reporting
        self._memory.note_store = self._note_store

        # Load persisted state
        self._memory.load()
        self._note_store.rebuild_index()

        logger.info(
            "MAGMA provider initialized (vault=%s, persist=%s, embedding=%s)",
            vault_dir, persist_dir, embedding,
        )

    def system_prompt_block(self) -> str:
        return (
            "[MAGMA Memory - 两层记忆架构]\n"
            "记忆分为索引层(轻量)+vault(完整内容)。\n"
            "自动注入的context是索引摘要+vault路径。\n\n"
            "Available tools:\n"
            "  - magma_search(query): 搜索记忆索引\n"
            "  - magma_read_note(topic): 读取vault中完整笔记\n"
            "  - magma_status(): 显示记忆统计\n"
        )

    # ---- Prefetch: lightweight index only --------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return lightweight index results (summaries + vault paths)."""
        if not query or len(query.strip()) < 3:
            return ""

        try:
            # Search vault index (lightweight, no embedding)
            results = self._note_store.search_notes(query, max_results=5)
            if not results:
                return ""

            parts = ["[MAGMA Memory Index — 相关笔记摘要]"]
            for r in results:
                title = r["title"]
                summary = r["summary"][:150]
                entities = ", ".join(r["entities"][:4])
                vault_path = f"E:/obsidian_hermes/hermes/magma/{r['filename']}"
                parts.append(
                    f"- **{title}**\n"
                    f"  {summary}...\n"
                    f"  tags: [{entities}]  |  vault: {vault_path}"
                )

            parts.append("\n需要深入了解请用 magma_read_note 读取完整笔记。")
            return "\n\n".join(parts)

        except Exception as e:
            logger.debug("MAGMA prefetch failed (non-fatal): %s", e)
            return ""

    # ---- Sync Turn: distill → vault → index -----------------------------

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Distill conversation → write vault → store summary in MAGMA."""
        self._turn_count += 1

        if not user_content or not assistant_content:
            return

        # Run async to avoid blocking
        def _process():
            try:
                # 1. Distill
                result = self._distiller.distill_turn(
                    user_content.strip(),
                    assistant_content.strip(),
                )

                # 2. Write to vault
                vault_path, note_id = self._note_store.write_note(
                    title=result.title,
                    summary=result.summary,
                    key_points=result.key_points,
                    content=result.content,
                    entities=result.entities,
                    importance=result.importance,
                    source=f"session:{self._session_id}",
                )

                # 3. Store summary in MAGMA graph (lightweight event)
                summary_text = (
                    f"[{result.title}] {result.summary[:200]} "
                    f"→ vault: {vault_path}"
                )
                self._memory.add_event(
                    content=summary_text,
                    metadata={
                        "speaker": "system",
                        "source": "distilled_note",
                        "note_id": note_id,
                        "vault_path": vault_path,
                        "entities": result.entities,
                        "importance": result.importance,
                        "session_id": session_id or self._session_id,
                    },
                )

                # 4. Apply edge forgetting (decay all edges)
                self._memory.apply_decay(turns=1)

            except Exception as e:
                logger.warning("MAGMA sync_turn process failed: %s", e)

        content_len = len(user_content or "") + len(assistant_content or "")
        if content_len > 100:
            t = threading.Thread(target=_process, daemon=True, name="magma-ingest")
            t.start()
        else:
            _process()

    # ---- Tool Schemas ----------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "magma_search",
                "description": (
                    "Search the MAGMA memory index. Returns note summaries "
                    "with their vault paths. Lightweight — no full content."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for memory notes",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Max results",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 50,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "magma_read_note",
                "description": (
                    "Read a full memory note from the Obsidian vault. "
                    "Provide a topic/keyword to find the relevant note. "
                    "Returns the complete markdown content."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "Topic, title, or keyword to find the note",
                        },
                    },
                    "required": ["topic"],
                },
            },
            {
                "name": "magma_status",
                "description": "Show MAGMA memory system statistics — notes, edges, vault.",
                "parameters": {"type": "object", "properties": {}},
            },
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "magma_search":
            return self._handle_search(args)
        elif tool_name == "magma_read_note":
            return self._handle_read_note(args)
        elif tool_name == "magma_status":
            return self._handle_status()
        else:
            raise NotImplementedError(f"MAGMA does not handle {tool_name}")

    def _handle_search(self, args: Dict[str, Any]) -> str:
        query = args.get("query", "")
        max_results = args.get("max_results", 10)
        if not query.strip():
            return json.dumps({"success": False, "error": "query is required"})

        try:
            # Search vault index
            results = self._note_store.search_notes(query, max_results=max_results)
            return json.dumps({
                "success": True,
                "results": results,
                "count": len(results),
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    def _handle_read_note(self, args: Dict[str, Any]) -> str:
        topic = args.get("topic", "")
        if not topic.strip():
            return json.dumps({"success": False, "error": "topic is required"})

        try:
            content = self._note_store.read_note(topic.strip())
            if content:
                # Find note_id to refresh importance
                for node in self._memory.graph.all_nodes():
                    vp = node.attributes.get("vault_path", "")
                    if vp and topic.lower() in vp.lower():
                        self._memory.access_note(node.node_id)
                        break
                return json.dumps({
                    "success": True,
                    "content": content,
                }, ensure_ascii=False)
            else:
                return json.dumps({
                    "success": False,
                    "error": f"No note found for: {topic}",
                })
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    def _handle_status(self) -> str:
        try:
            status = self._memory.status()
            status["vault_dir"] = str(self._note_store.vault_dir)
            status["vault_notes"] = self._note_store.note_count()
            return json.dumps({"success": True, "memory": status}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    def shutdown(self) -> None:
        """Save state async."""
        def _do():
            try:
                self._memory.request_consolidation()
                self._memory.save()
                logger.info("MAGMA saved")
            except Exception as e:
                logger.warning("MAGMA shutdown save failed: %s", e)

        t = threading.Thread(target=_do, daemon=True, name="magma-shutdown")
        t.start()

    # -- Memory write mirroring --------------------------------------------

    def on_memory_write(self, action: str, target: str, content: str,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        if content:
            # Write to vault as a note
            self._note_store.write_note(
                title=f"Manual memory: {target}",
                summary=content[:200],
                key_points=[content[:100]] if len(content) > 100 else [content],
                content=content,
                entities=[target],
                importance=1.0,
                source="memory_tool",
            )

    # -- Config schema -----------------------------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "persist_dir",
                "description": "MAGMA graph state directory",
                "required": False, "secret": False,
            },
            {
                "key": "embedding",
                "description": "'minilm', 'openai', or 'char'",
                "required": False, "secret": False,
                "default": "minilm",
                "choices": ["minilm", "openai", "char"],
            },
            {
                "key": "vault_dir",
                "description": f"Obsidian vault directory (default: {_VAULT_DIR})",
                "required": False, "secret": False,
            },
            {
                "key": "apikey",
                "description": "OpenAI API key (only for embedding='openai')",
                "required": False, "secret": True,
                "env_var": "OPENAI_API_KEY",
            },
        ]

    # -- Session hooks -----------------------------------------------------

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        try:
            self._memory.save()
        except Exception as e:
            logger.warning("MAGMA on_session_end save failed: %s", e)
