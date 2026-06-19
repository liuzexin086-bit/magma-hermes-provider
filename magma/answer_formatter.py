"""
Answer Formatter — Structure-Aware Narrative Context Construction

Transforms retrieved sub-graph into a linearized, type-aligned prompt context
with provenance, temporal ordering, and salience-based budgeting.

Based on MAGMA's Stage 4: Narrative Synthesis via Graph Linearization.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from .graph_db import EventNode, Link, LinkType

logger = logging.getLogger(__name__)

# Token budget for linearized context (approximate)
_MAX_CONTEXT_TOKENS = 3000
_CHARS_PER_TOKEN = 4  # rough estimate


class AnswerFormatter:
    """Formats retrieved graph context into a structured narrative."""

    def format_context(
        self,
        nodes: List[EventNode],
        edges: List[Link],
        query_text: str = "",
        max_tokens: int = _MAX_CONTEXT_TOKENS,
    ) -> str:
        """
        Linearize a sub-graph into a context block.

        Steps:
        1. Sort nodes by query-relevant ordering (temporal vs causal)
        2. Build provenance blocks with timestamp and ref ID
        3. Apply salience-based token budgeting
        """
        if not nodes:
            return ""

        # classify query type from text
        query_lower = query_text.lower()
        is_causal = any(w in query_lower for w in [
            "why", "cause", "because", "reason", "led to", "result",
            "impact", "effect", "affect",
        ])
        is_temporal = any(w in query_lower for w in [
            "when", "before", "after", "during", "timeline", "sequence",
            "what happened", "order", "first", "then", "later",
        ])
        is_entity = any(w in query_lower for w in [
            "who", "which", "about", "regarding", "involving",
        ])

        # 1. Topological ordering
        if is_causal:
            sorted_nodes = self._topological_sort(nodes, edges)
        elif is_temporal:
            sorted_nodes = sorted(nodes, key=lambda n: n.timestamp)
        else:
            # default: temporal with causal preference
            sorted_nodes = self._sort_by_relevance(nodes, query_text)

        # 2. Context scaffolding with provenance
        char_budget = max_tokens * _CHARS_PER_TOKEN
        context_parts: List[str] = []
        char_used = 0
        node_ids_in_context: Set[str] = set()

        for i, node in enumerate(sorted_nodes):
            if node.node_id in node_ids_in_context:
                continue
            node_ids_in_context.add(node.node_id)

            block = self._format_node_block(node, i + 1)
            block_len = len(block)

            # 3. Salience-based budgeting
            if char_used + block_len <= char_budget:
                context_parts.append(block)
                char_used += block_len
            else:
                # summarize remaining as brevity codes
                remaining = len(sorted_nodes) - i
                if remaining > 1:
                    context_parts.append(
                        f"[... {remaining} additional events summarized ...]")
                break

        if not context_parts:
            return ""

        result = "\n\n".join(context_parts)

        # If multiple nodes, prefix with a summary note
        if len(context_parts) > 1:
            result = (
                "Memory context from past interactions:\n\n" + result
            )

        return result

    def _format_node_block(self, node: EventNode, index: int) -> str:
        """Serialize a single event node into a structured text block."""
        ts = node.timestamp.strftime("%Y-%m-%d %H:%M") if node.timestamp else "unknown"
        entities = node.attributes.get("entities", [])
        entity_str = f" [entities: {', '.join(entities[:5])}]" if entities else ""

        return (
            f"<t:{ts}> {node.content} <ref:{node.node_id[:8]}>{entity_str}"
        )

    def _topological_sort(self, nodes: List[EventNode],
                          edges: List[Link]) -> List[EventNode]:
        """Sort nodes by causal dependency (causes before effects)."""
        node_map = {n.node_id: n for n in nodes}
        # Build in-degree map
        in_degree: Dict[str, int] = {n.node_id: 0 for n in nodes}
        causal_edges = [e for e in edges if e.link_type == LinkType.CAUSAL]

        adj: Dict[str, List[str]] = {n.node_id: [] for n in nodes}
        for e in causal_edges:
            if e.source_id in node_map and e.target_id in node_map:
                adj.setdefault(e.source_id, []).append(e.target_id)
                in_degree[e.target_id] = in_degree.get(e.target_id, 0) + 1

        # Kahn's algorithm
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        sorted_ids: List[str] = []
        while queue:
            nid = queue.pop(0)
            sorted_ids.append(nid)
            for nbr in adj.get(nid, []):
                in_degree[nbr] -= 1
                if in_degree[nbr] == 0:
                    queue.append(nbr)

        # Append any remaining nodes not in causal chain (sorted by time)
        remaining = [n for n in nodes if n.node_id not in sorted_ids]
        remaining.sort(key=lambda n: n.timestamp)

        # sort remaining also by time
        result = [node_map[nid] for nid in sorted_ids if nid in node_map]
        result.extend(remaining)
        return result

    def _sort_by_relevance(self, nodes: List[EventNode],
                           query: str) -> List[EventNode]:
        """Default sort: temporal recency, with query-matched nodes first."""
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        def score(node: EventNode) -> float:
            s = 0.0
            content_lower = node.content.lower()
            # boost nodes matching query terms
            for term in query_terms:
                if term in content_lower:
                    s += 2.0
            # penalize older nodes slightly
            age_hours = (datetime.now() - node.timestamp).total_seconds() / 3600
            s -= age_hours * 0.001
            return s

        return sorted(nodes, key=score, reverse=True)
