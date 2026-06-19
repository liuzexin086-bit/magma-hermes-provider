"""
MAGMA Memory Provider — Multi-Graph Agentic Memory for Hermes Agent

Implements the MemoryProvider ABC to bring MAGMA-style multi-graph memory
to Hermes Agent. Designed to coexist with the built-in memory provider.

Architecture overview:
  - Every conversation turn is automatically ingested as an EventNode
  - Events are linked via 4 orthogonal graph types (TEMPORAL, SEMANTIC,
    CAUSAL, ENTITY)
  - Prefetch uses MAGMA's intent-aware graph traversal for context retrieval
  - A `magma_search` tool allows manual graph queries
  - Dual-stream: fast event ingestion + background structural consolidation

Config (in config.yaml under memory.magma):
  persist_dir:  path to store graph state (default: $HERMES_HOME/magma/)
  embedding:    'minilm' (default), 'openai', or 'char'
  max_events:   max events before oldest are pruned (default: 10000)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

from .trg_memory import TRGMemory
from .graph_db import TraversalConstraints

logger = logging.getLogger(__name__)

_PROVIDER_NAME = "magma"

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_config(hermes_home: Path) -> dict:
    """Load MAGMA provider config from $HERMES_HOME/magma_config.json."""
    config_path = hermes_home / "magma_config.json"
    defaults = {
        "persist_dir": str(hermes_home / "magma"),
        "embedding": "minilm",
        "max_events": 10000,
        "enable_consolidation": True,
    }
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            defaults.update(file_cfg)
        except Exception as e:
            logger.warning("Failed to load magma config: %s", e)
    return defaults


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class MagmaMemoryProvider(MemoryProvider):
    """MAGMA-based multi-graph memory provider for Hermes Agent."""

    @property
    def name(self) -> str:
        return _PROVIDER_NAME

    # -- Lifecycle ---------------------------------------------------------

    def is_available(self) -> bool:
        return True  # always available; no external deps required

    def initialize(self, session_id: str, **kwargs) -> None:
        hermes_home = kwargs.get("hermes_home")
        if hermes_home:
            hermes_home = Path(hermes_home)
        else:
            from hermes_constants import get_hermes_home
            hermes_home = get_hermes_home()

        config = _load_config(hermes_home)

        persist_dir = config.get("persist_dir")
        if not persist_dir:
            persist_dir = str(hermes_home / "magma")
        embedding = config.get("embedding", "minilm")
        enable_consolidation = config.get("enable_consolidation", True)

        self._memory = TRGMemory(
            persist_dir=persist_dir,
            embedding_model=embedding,
            enable_consolidation=enable_consolidation,
        )
        self._max_events = config.get("max_events", 10000)
        self._persist_dir = Path(persist_dir)
        self._session_id = session_id

        # Load persisted state
        self._memory.load()

        # Track turns for auto-write
        self._turn_buffer: List[str] = []

        logger.info(
            "MAGMA provider initialized (persist=%s, embedding=%s, max_events=%d)",
            persist_dir, embedding, self._max_events,
        )

    def system_prompt_block(self) -> str:
        return (
            "[MAGMA Memory]\n"
            "You have a multi-graph memory system (MAGMA) that automatically "
            "records every turn as an event and builds temporal, semantic, "
            "causal, and entity relationships between events.\n\n"
            "Available tools:\n"
            "  - magma_search(query, intent, max_results): Search memory graph\n"
            "  - magma_status(): Show memory statistics\n\n"
            "Memory context from past interactions is injected automatically "
            "before each turn."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recall relevant context using MAGMA graph traversal."""
        if not query or not query.strip():
            return ""

        try:
            context, nodes, _ = self._memory.query(
                query_text=query,
                max_results=8,
            )
            if context and context != "(No relevant memory found.)":
                return context
        except Exception as e:
            logger.debug("MAGMA prefetch failed (non-fatal): %s", e)

        return ""

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Auto-ingest each turn as an event (Fast Path)."""
        try:
            # Prune if over limit
            if self._memory.graph.node_count() > self._max_events:
                self._prune_oldest_events()

            # Ingest user turn
            if user_content and user_content.strip():
                self._memory.add_event(
                    content=user_content.strip(),
                    metadata={
                        "speaker": "user",
                        "session_id": session_id or self._session_id,
                    },
                )

            # Ingest assistant turn
            if assistant_content and assistant_content.strip():
                self._memory.add_event(
                    content=assistant_content.strip(),
                    metadata={
                        "speaker": "assistant",
                        "session_id": session_id or self._session_id,
                    },
                )

        except Exception as e:
            logger.warning("MAGMA sync_turn failed (non-fatal): %s", e)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "magma_search",
                "description": (
                    "Search the multi-graph memory using MAGMA's adaptive traversal. "
                    "Returns relevant past interactions structured by relationship type. "
                    "Use this when you need to recall specific facts, trace causal chains, "
                    "or find entities across conversations."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query about past interactions",
                        },
                        "intent": {
                            "type": "string",
                            "enum": ["auto", "why", "when", "entity", "general"],
                            "description": "Query intent for traversal weighting. 'auto' detects from query.",
                            "default": "auto",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of events to return",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 100,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "magma_status",
                "description": "Show MAGMA memory system statistics — event count, edge count, vector DB size, consolidation status.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "magma_search":
            return self._handle_search(args)
        elif tool_name == "magma_status":
            return self._handle_status()
        else:
            raise NotImplementedError(f"MAGMA provider does not handle {tool_name}")

    def _handle_search(self, args: Dict[str, Any]) -> str:
        query = args.get("query", "")
        intent_raw = args.get("intent", "auto")
        max_results = args.get("max_results", 10)

        if not query.strip():
            return json.dumps({"success": False, "error": "query is required"})

        if intent_raw == "auto":
            intent = None  # let TRGMemory classify
        else:
            intent = intent_raw

        try:
            context, nodes, edges = self._memory.query(
                query_text=query,
                max_results=max_results,
                intent=intent,
            )

            # Build result summary
            edge_counts: Dict[str, int] = {}
            from .graph_db import LinkType
            for e in edges:
                et = e.link_type.value
                edge_counts[et] = edge_counts.get(et, 0) + 1

            return json.dumps({
                "success": True,
                "context": context,
                "events_retrieved": len(nodes),
                "edges_retrieved": len(edges),
                "edge_type_counts": edge_counts,
            }, ensure_ascii=False)
        except Exception as e:
            logger.exception("magma_search failed")
            return json.dumps({"success": False, "error": str(e)})

    def _handle_status(self) -> str:
        try:
            status = self._memory.status()
            return json.dumps({
                "success": True,
                "memory": status,
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    def shutdown(self) -> None:
        """Save state on shutdown."""
        try:
            self._memory.request_consolidation()
            self._memory.save()
            logger.info("MAGMA memory saved on shutdown")
        except Exception as e:
            logger.warning("MAGMA shutdown save failed: %s", e)

    # -- Memory write mirroring --------------------------------------------

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mirror built-in memory writes into the MAGMA graph."""
        if action == "add" and content:
            self._memory.add_event(
                content=f"[{target}] {content}",
                metadata={
                    "source": "memory_tool",
                    "action": action,
                    "target": target,
                    **(metadata or {}),
                },
            )
        elif action == "replace" and content:
            self._memory.add_event(
                content=f"[{target} replace] {content}",
                metadata={
                    "source": "memory_tool",
                    "action": action,
                    "target": target,
                    **(metadata or {}),
                },
            )

    # -- Config schema for 'hermes memory setup' ---------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "persist_dir",
                "description": "Directory to store MAGMA graph state (default: $HERMES_HOME/magma/)",
                "required": False,
                "secret": False,
            },
            {
                "key": "embedding",
                "description": "Embedding model: 'minilm' (default, fast local), 'openai' (API key needed), or 'char' (no deps)",
                "required": False,
                "secret": False,
                "default": "minilm",
                "choices": ["minilm", "openai", "char"],
            },
            {
                "key": "max_events",
                "description": "Maximum events before pruning oldest (default: 10000)",
                "required": False,
                "secret": False,
                "default": 10000,
            },
            {
                "key": "apikey",
                "description": "OpenAI API key for embedding (only needed if embedding='openai')",
                "required": False,
                "secret": True,
                "env_var": "OPENAI_API_KEY",
            },
        ]

    # -- Pruning -----------------------------------------------------------

    def _prune_oldest_events(self) -> None:
        """Remove the oldest 10% of events to stay under max_events."""
        all_nodes = self._memory.graph.all_nodes()
        if len(all_nodes) <= self._max_events:
            return

        sorted_nodes = sorted(all_nodes, key=lambda n: n.timestamp)
        to_remove = len(sorted_nodes) - self._max_events + int(self._max_events * 0.1)
        for node in sorted_nodes[:to_remove]:
            self._memory.graph.remove_node(node.node_id)
        logger.info("Pruned %d oldest events", to_remove)

    # -- Session hooks -----------------------------------------------------

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Save state at session boundaries."""
        try:
            self._memory.save()
        except Exception as e:
            logger.warning("MAGMA on_session_end save failed: %s", e)

    def on_delegation(
        self,
        task: str,
        result: str,
        *,
        child_session_id: str = "",
        **kwargs,
    ) -> None:
        """Record delegation events in the parent's memory graph."""
        if task:
            self._memory.add_event(
                content=f"[delegated task] {task}",
                metadata={"source": "delegation", "role": "task", "child_session": child_session_id},
            )
        if result:
            self._memory.add_event(
                content=f"[delegation result] {result[:500]}",
                metadata={"source": "delegation", "role": "result", "child_session": child_session_id},
            )
