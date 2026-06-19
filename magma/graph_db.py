"""
Graph Database — Multi-Graph Memory Substrate

Four orthogonal relation graphs for agentic memory:
  - TEMPORAL: strict time-ordered chain (immutable backbone)
  - SEMANTIC: undirected cosine-similarity edges
  - CAUSAL: directed LLM-inferred entailment
  - ENTITY: event-to-entity cross-links (object permanence)

Based on MAGMA: Multi-Graph based Agentic Memory Architecture (ACL 2026).
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class NodeType(Enum):
    EVENT = "EVENT"
    EPISODE = "EPISODE"
    ENTITY = "ENTITY"
    SESSION = "SESSION"


class LinkType(Enum):
    TEMPORAL = "TEMPORAL"
    SEMANTIC = "SEMANTIC"
    CAUSAL = "CAUSAL"
    ENTITY = "ENTITY"


class LinkStatus(Enum):
    ACTIVE = "ACTIVE"
    DEPRECATED = "DEPRECATED"
    PENDING = "PENDING"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class EventNode:
    """:cvar
    A single memory event with content, timestamp, embedding, and attributes.
    """
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_type: NodeType = NodeType.EVENT
    timestamp: datetime = field(default_factory=datetime.now)
    content: str = ""
    embedding: Optional[List[float]] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "content": self.content,
            "embedding": self.embedding,
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EventNode":
        node = cls(
            node_id=data.get("node_id", str(uuid.uuid4())),
            content=data.get("content", ""),
            embedding=data.get("embedding"),
            attributes=data.get("attributes", {}),
        )
        if "timestamp" in data and data["timestamp"]:
            node.timestamp = datetime.fromisoformat(data["timestamp"])
        if "node_type" in data:
            node.node_type = NodeType(data["node_type"])
        return node


@dataclass
class Link:
    """Edge connecting two nodes in the memory graph."""
    link_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    link_type: LinkType = LinkType.TEMPORAL
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if "created_at" not in self.metadata:
            self.metadata["created_at"] = datetime.now().isoformat()
        if "status" not in self.metadata:
            self.metadata["status"] = LinkStatus.ACTIVE.value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "link_id": self.link_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "link_type": self.link_type.value,
            "weight": self.weight,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Link":
        link = cls(
            link_id=data.get("link_id", str(uuid.uuid4())),
            source_id=data.get("source_id", ""),
            target_id=data.get("target_id", ""),
            weight=data.get("weight", 1.0),
            metadata=data.get("metadata", {}),
        )
        if "link_type" in data:
            link.link_type = LinkType(data["link_type"])
        return link


@dataclass
class TraversalConstraints:
    """Controls graph traversal behaviour."""
    max_depth: int = 3
    max_nodes: int = 100
    link_types: Optional[Set[LinkType]] = None
    time_window: Optional[Tuple[datetime, datetime]] = None
    follow_temporal: bool = True
    follow_semantic: bool = True
    follow_causal: bool = True
    follow_entity: bool = True

    def allows_link(self, link: Link) -> bool:
        if self.link_types and link.link_type not in self.link_types:
            return False
        if link.metadata.get("status") == LinkStatus.DEPRECATED.value:
            return False
        if link.link_type == LinkType.TEMPORAL and not self.follow_temporal:
            return False
        if link.link_type == LinkType.SEMANTIC and not self.follow_semantic:
            return False
        if link.link_type == LinkType.CAUSAL and not self.follow_causal:
            return False
        if link.link_type == LinkType.ENTITY and not self.follow_entity:
            return False
        return True


# ---------------------------------------------------------------------------
# NetworkX-backed Graph Database
# ---------------------------------------------------------------------------

class GraphDB:
    """
    In-memory multi-graph using NetworkX.
    Each node can have edges of all four LinkTypes.
    """

    def __init__(self):
        self._nodes: Dict[str, EventNode] = {}
        self._edges: Dict[str, Link] = {}
        # adjacency: node_id -> {node_id -> [link_id, ...]}
        self._adj: Dict[str, Dict[str, List[str]]] = {}

    # ---- Nodes ------------------------------------------------------------

    def add_node(self, node: EventNode) -> None:
        self._nodes[node.node_id] = node
        if node.node_id not in self._adj:
            self._adj[node.node_id] = {}

    def get_node(self, node_id: str) -> Optional[EventNode]:
        return self._nodes.get(node_id)

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def remove_node(self, node_id: str) -> None:
        self._nodes.pop(node_id, None)
        # remove all edges touching this node
        dead: List[str] = []
        for lid, link in self._edges.items():
            if link.source_id == node_id or link.target_id == node_id:
                dead.append(lid)
        for lid in dead:
            self._remove_edge(lid)
        self._adj.pop(node_id, None)
        for nbrs in self._adj.values():
            nbrs.pop(node_id, None)

    def all_nodes(self) -> List[EventNode]:
        return list(self._nodes.values())

    def node_count(self) -> int:
        return len(self._nodes)

    # ---- Edges ------------------------------------------------------------

    def add_edge(self, link: Link) -> str:
        self._edges[link.link_id] = link
        # bidir adjacency: source->target
        nbrs = self._adj.setdefault(link.source_id, {})
        nbrs.setdefault(link.target_id, []).append(link.link_id)
        # also for traversal we want reverse lookup
        rev = self._adj.setdefault(link.target_id, {})
        rev.setdefault(link.source_id, []).append(link.link_id)
        return link.link_id

    def get_edge(self, link_id: str) -> Optional[Link]:
        return self._edges.get(link_id)

    def get_edges_between(self, src: str, tgt: str,
                          link_type: Optional[LinkType] = None) -> List[Link]:
        lids = self._adj.get(src, {}).get(tgt, [])
        results = [self._edges[lid] for lid in lids if lid in self._edges]
        if link_type:
            results = [e for e in results if e.link_type == link_type]
        return results

    def _remove_edge(self, link_id: str) -> None:
        link = self._edges.pop(link_id, None)
        if link is None:
            return
        # remove from adjacency
        for s, t in [(link.source_id, link.target_id),
                      (link.target_id, link.source_id)]:
            nbrs = self._adj.get(s)
            if nbrs:
                lids = nbrs.get(t, [])
                if link_id in lids:
                    lids.remove(link_id)
                    if not lids:
                        del nbrs[t]

    def edge_count(self) -> int:
        return len(self._edges)

    # ---- Traversal --------------------------------------------------------

    def get_neighbors(self, node_id: str,
                      constraints: Optional[TraversalConstraints] = None
                      ) -> List[Tuple[str, List[Link]]]:
        """Return (neighbor_id, [links]) for each neighbor reachable via allowed edges."""
        result: Dict[str, List[Link]] = {}
        nbr_map = self._adj.get(node_id, {})
        for nbr_id, lid_list in nbr_map.items():
            edges = [self._edges[lid] for lid in lid_list if lid in self._edges]
            if constraints:
                edges = [e for e in edges if constraints.allows_link(e)]
            if edges:
                result[nbr_id] = edges
        return list(result.items())

    def traverse(self, start_ids: List[str],
                 constraints: Optional[TraversalConstraints] = None
                 ) -> Dict[str, Set[str]]:
        """
        BFS traversal from start nodes.

        Returns: {depth: {node_id, ...}} where depth 0 are the start nodes.
        """
        if constraints is None:
            constraints = TraversalConstraints()

        visited: Set[str] = set(start_ids)
        result: Dict[int, Set[str]] = {0: set(start_ids)}

        current = list(start_ids)
        for depth in range(1, constraints.max_depth + 1):
            if not current or len(visited) >= constraints.max_nodes:
                break
            next_level: Set[str] = set()
            for nid in current:
                for nbr_id, _ in self.get_neighbors(nid, constraints):
                    if nbr_id not in visited and len(visited) < constraints.max_nodes:
                        next_level.add(nbr_id)
                        visited.add(nbr_id)
            if next_level:
                result[depth] = next_level
            current = list(next_level)

        return result

    def bfs_subgraph(self, start_ids: List[str],
                     constraints: Optional[TraversalConstraints] = None
                     ) -> Tuple[List[EventNode], List[Link]]:
        """
        BFS from start_ids, collecting all visited nodes + edges between them.
        """
        level_map = self.traverse(start_ids, constraints)
        all_ids: Set[str] = set()
        for ids in level_map.values():
            all_ids.update(ids)

        nodes = [self._nodes[nid] for nid in all_ids if nid in self._nodes]
        edges: List[Link] = []
        for nid in all_ids:
            nbrs = self._adj.get(nid, {})
            for nbr_id, lids in nbrs.items():
                if nbr_id in all_ids:
                    for lid in lids:
                        if lid in self._edges:
                            edges.append(self._edges[lid])
        return nodes, edges

    # ---- Persistence ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": {nid: n.to_dict() for nid, n in self._nodes.items()},
            "edges": {lid: e.to_dict() for lid, e in self._edges.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphDB":
        db = cls()
        for nid, ndata in data.get("nodes", {}).items():
            node = EventNode.from_dict(ndata)
            db._nodes[node.node_id] = node
        for lid, edata in data.get("edges", {}).items():
            link = Link.from_dict(edata)
            db._edges[link.link_id] = link
        # rebuild adjacency
        for link in db._edges.values():
            nbrs = db._adj.setdefault(link.source_id, {})
            nbrs.setdefault(link.target_id, []).append(link.link_id)
            rev = db._adj.setdefault(link.target_id, {})
            rev.setdefault(link.source_id, []).append(link.link_id)
        return db

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                        encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "GraphDB":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)
