"""
Temporal Resonance Graph Memory (TRG) — Core Engine

Multi-graph agentic memory based on MAGMA (ACL 2026).
Manages event ingestion, relation graph construction, and query traversal.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .graph_db import (
    EventNode,
    GraphDB,
    Link,
    LinkType,
    LinkStatus,
    NodeType,
    TraversalConstraints,
)
from .vector_db import VectorDB, create_vector_db
from .keyword_enrichment import KeywordEnricher

logger = logging.getLogger(__name__)

# Maximum number of events before consolidation triggers
_CONSOLIDATION_THRESHOLD = 50


# ---------------------------------------------------------------------------
# Simple embedding fallback (character-level hashing for zero-dependency mode)
# ---------------------------------------------------------------------------

class _CharEmbedder:
    """Character-based bag-of-ngrams embedding — no model dependencies.

    Produces 384-dimensional vectors (compatible with default vector_db).
    Used when sentence-transformers is not available.
    """

    def __init__(self, dimension: int = 384):
        self.dim = dimension

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        if not text:
            return vec
        text_lower = text.lower()
        # character n-gram hashing (2-grams and 3-grams)
        for n in (2, 3):
            for i in range(len(text_lower) - n + 1):
                gram = text_lower[i:i + n]
                idx = abs(hash(gram)) % self.dim
                vec[idx] += 1.0
        # normalise
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


# Try to use sentence-transformers if available
_SENTENCE_TRANSFORMER = None
try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMER = True
except ImportError:
    _SENTENCE_TRANSFORMER = False


def _get_embedder(model_name: str = "all-MiniLM-L6-v2"):
    """Return an embedder function: text -> np.ndarray."""
    if _SENTENCE_TRANSFORMER:
        try:
            model = SentenceTransformer(model_name)
            logger.info("Using SentenceTransformer: %s", model_name)

            def encode(text: str) -> np.ndarray:
                emb = model.encode(text)
                if len(emb.shape) == 2:
                    emb = emb[0]
                return emb.astype(np.float32)

            return encode, 384
        except Exception as e:
            logger.warning("SentenceTransformer init failed: %s. Falling back.", e)

    embedder = _CharEmbedder(384)
    logger.info("Using character-n-gram embedding fallback (dim=384)")
    return embedder.encode, 384


# ---------------------------------------------------------------------------
# TRG Memory Engine
# ---------------------------------------------------------------------------

class TRGMemory:
    """
    Multi-graph agentic memory engine.

    Key design:
      - Events are stored as nodes with 4 orthogonal edge types
      - Dual-stream ingestion: Fast Path (sync) + Slow Path (async consolidation)
      - Query: adaptive traversal with intent-aware routing
    """

    def __init__(
        self,
        graph_db: Optional[GraphDB] = None,
        vector_db: Optional[VectorDB] = None,
        persist_dir: Optional[str] = None,
        embedding_model: str = "minilm",
        enable_consolidation: bool = True,
    ):
        self.graph = graph_db or GraphDB()
        self.keyword_enricher = KeywordEnricher()

        # Embedding
        if embedding_model == "openai":
            # Use OpenAI API - require env var
            self._embed, self._embed_dim = self._init_openai_embedder()
        else:
            self._embed, self._embed_dim = _get_embedder()

        self.vector_db = vector_db or create_vector_db(
            backend="auto",
            dimension=self._embed_dim,
        )
        self.persist_dir = Path(persist_dir) if persist_dir else None

        # Dual-stream memory evolution
        self._consolidation_queue: Queue = Queue()
        self._consolidation_thread: Optional[Thread] = None
        self._enable_consolidation = enable_consolidation
        self._consolidation_lock = Lock()
        self._events_since_consolidation = 0
        self._last_node_id: Optional[str] = None

        # Statistics
        self.stats = {
            "events_added": 0,
            "queries_run": 0,
            "links_created": 0,
            "consolidations_run": 0,
        }

        logger.info(
            "TRGMemory initialized (embed_dim=%d, consolidation=%s)",
            self._embed_dim, enable_consolidation,
        )

    def _init_openai_embedder(self):
        """Initialize OpenAI embedding API."""
        import os

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set; falling back to char embedder")
            embedder = _CharEmbedder(384)
            return embedder.encode, 384

        try:
            import openai

            client = openai.OpenAI(api_key=api_key)

            def encode(text: str) -> np.ndarray:
                resp = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text,
                )
                return np.array(resp.data[0].embedding, dtype=np.float32)

            return encode, 1536
        except Exception as e:
            logger.warning("OpenAI embedding init failed: %s. Falling back.", e)
            embedder = _CharEmbedder(384)
            return embedder.encode, 384

    # ---- Embedding helper -------------------------------------------------

    def _embed_text(self, text: str) -> np.ndarray:
        return self._embed(text)

    # ---- Fast Path: Synaptic Ingestion -----------------------------------

    def add_event(
        self,
        content: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add an event to memory (Fast Path).

        1. Segment content
        2. Create node with embedding
        3. Index in vector DB
        4. Create temporal link to previous event
        5. Create semantic links to similar events
        6. Trigger async consolidation
        """
        ts = timestamp or datetime.now()
        meta = metadata or {}

        # Extract entities and keywords
        entities = meta.get("entities", [])
        if not entities:
            # simple entity extraction fallback
            entities = self._extract_entities(content)
        keywords = meta.get("keywords", [])
        if not keywords:
            keywords = self.keyword_enricher.extract(content, max_keywords=8)

        # Create event node
        node = EventNode(
            node_id=str(uuid.uuid4()),
            timestamp=ts,
            content=content,
            attributes={
                "entities": entities,
                "keywords": keywords,
                "speaker": meta.get("speaker"),
                "session_id": meta.get("session_id"),
                "emotion": meta.get("emotion"),
                "raw_content": meta.get("raw_content", content),
            },
        )

        # Enrich content for embedding
        enriched = content
        if keywords:
            enriched = f"{content} keywords: {' '.join(keywords)}"

        # Generate embedding
        embedding = self._embed_text(enriched)
        node.embedding = embedding.tolist()

        # Add to graph
        self.graph.add_node(node)

        # Add to vector DB
        self.vector_db.add(
            vector_id=node.node_id,
            vector=embedding,
            metadata={
                "timestamp": ts.isoformat(),
                "keywords": keywords,
                "entities": entities,
            },
        )

        # Create temporal link to previous event
        if self._last_node_id:
            temp_link = Link(
                source_id=self._last_node_id,
                target_id=node.node_id,
                link_type=LinkType.TEMPORAL,
                weight=1.0,
                metadata={"sub_type": "PRECEDES"},
            )
            self.graph.add_edge(temp_link)
            self.stats["links_created"] += 1

        self._last_node_id = node.node_id

        # Create semantic links (cosine similarity to recent events)
        self._create_semantic_links(node)

        # Update stats
        self.stats["events_added"] += 1
        self._events_since_consolidation += 1

        # Trigger consolidation if threshold reached
        if (
            self._enable_consolidation
            and self._events_since_consolidation >= _CONSOLIDATION_THRESHOLD
        ):
            self._consolidation_queue.put(node.node_id)
            self._start_consolidation_worker()

        logger.debug("Added event: %s (%.60s…)", node.node_id[:8], content)
        return node.node_id

    def _extract_entities(self, text: str) -> List[str]:
        """Simple entity extraction — capitalized words that aren't stop words."""
        import re

        stop = {
            "The", "This", "That", "These", "Those", "What", "When",
            "Where", "Who", "Why", "How", "I", "You", "He", "She", "It",
            "We", "They", "My", "Your", "His", "Her", "Its", "Our",
            "Their", "A", "An", "And", "Or", "But", "If", "So", "Then",
            "Now", "Here", "There", "Please", "Thanks", "Hello", "Hi",
            "Yes", "No", "Good", "Great", "Ok", "Okay", "Sure", "Right",
            "Actually", "Basically", "Really", "Just", "Also", "Well",
            "Let", "Go", "Get", "Make", "Take", "Use", "Say", "Know",
            "Think", "Want", "Need", "Like", "Would", "Could", "Should",
        }
        candidates = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
        return list(dict.fromkeys(c for c in candidates if c not in stop))[:10]

    def _create_semantic_links(self, node: EventNode, max_links: int = 5):
        """Connect a new node to semantically similar existing nodes."""
        if self.graph.node_count() <= 1:
            return

        # Find similar via vector search (excluding self)
        results = self.vector_db.search(
            np.array(node.embedding, dtype=np.float32) if node.embedding
            else self._embed_text(node.content),
            k=max_links + 1,
        )
        count = 0
        for vid, score, _ in results:
            if vid == node.node_id:
                continue
            if score < 0.3:
                continue
            link = Link(
                source_id=node.node_id,
                target_id=vid,
                link_type=LinkType.SEMANTIC,
                weight=float(score),
            )
            self.graph.add_edge(link)
            self.stats["links_created"] += 1
            count += 1
            if count >= max_links:
                break

    # ---- Slow Path: Structural Consolidation -----------------------------

    def _start_consolidation_worker(self):
        """Ensure the background consolidation thread is running."""
        with self._consolidation_lock:
            if self._consolidation_thread is None or not self._consolidation_thread.is_alive():
                self._consolidation_thread = Thread(
                    target=self._consolidation_worker,
                    daemon=True,
                    name="magma-consolidation",
                )
                self._consolidation_thread.start()
                logger.debug("Started consolidation worker")

    def _consolidation_worker(self):
        """Background worker: infer causal and entity links."""
        while not self._consolidation_queue.empty():
            try:
                node_id = self._consolidation_queue.get_nowait()
                self._consolidate_event(node_id)
                self._consolidation_queue.task_done()
            except Exception:
                break

    def _consolidate_event(self, node_id: str) -> None:
        """Infer causal and entity links for a given event node."""
        node = self.graph.get_node(node_id)
        if node is None:
            return

        with self._consolidation_lock:
            self._events_since_consolidation = 0
            self.stats["consolidations_run"] += 1

        # Get neighbourhood (2-hop)
        constraints = TraversalConstraints(
            max_depth=2, max_nodes=50,
            follow_temporal=True, follow_semantic=True,
            follow_causal=False, follow_entity=False,
        )
        neighbours, _ = self.graph.bfs_subgraph([node_id], constraints)

        # --- Entity links ---
        my_entities = node.attributes.get("entities", [])
        for nbr in neighbours:
            if nbr.node_id == node_id:
                continue
            nbr_entities = nbr.attributes.get("entities", [])
            shared = set(my_entities) & set(nbr_entities)
            if shared:
                link = Link(
                    source_id=node_id,
                    target_id=nbr.node_id,
                    link_type=LinkType.ENTITY,
                    weight=len(shared) / max(len(my_entities), 1),
                    metadata={"shared_entities": list(shared)},
                )
                # Don't duplicate if link already exists
                existing = self.graph.get_edges_between(
                    node_id, nbr.node_id, LinkType.ENTITY)
                if not existing:
                    self.graph.add_edge(link)
                    self.stats["links_created"] += 1

        logger.debug(
            "Consolidated event %s (neighbours: %d, entities: %s)",
            node_id[:8], len(neighbours), my_entities[:3],
        )

    def request_consolidation(self) -> None:
        """Manually trigger consolidation for all events (blocking call for shutdown)."""
        all_nodes = self.graph.all_nodes()
        for n in all_nodes:
            self._consolidation_queue.put(n.node_id)
        self._start_consolidation_worker()
        if self._consolidation_thread:
            self._consolidation_thread.join(timeout=10)
        logger.info("Consolidation complete: %d events processed", len(all_nodes))

    # ---- Query -----------------------------------------------------------

    def query(
        self,
        query_text: str,
        max_results: int = 10,
        constraints: Optional[TraversalConstraints] = None,
        intent: Optional[str] = None,
    ) -> Tuple[str, List[EventNode], List[Link]]:
        """
        Query memory graph with adaptive retrieval.

        1. Enrich query
        2. Vector search for anchor nodes
        3. Graph traversal from anchors
        4. Structure-aware context formatting

        Returns: (formatted_context_string, nodes_list, edges_list)
        """
        # 1. Enrich query
        enriched_query = self.keyword_enricher.enrich_query(query_text)

        # 2. Embed
        query_vec = self._embed_text(enriched_query)

        # 3. Vector search for anchors
        num_anchors = max_results * 2
        search_results = self.vector_db.search(query_vec, k=num_anchors)

        anchor_ids = [r[0] for r in search_results]
        anchor_nodes = [
            self.graph.get_node(aid) for aid in anchor_ids
            if self.graph.get_node(aid)
        ]

        if not anchor_nodes:
            return "(No relevant memory found.)", [], []

        # 4. Graph traversal
        if constraints is None:
            constraints = TraversalConstraints(
                max_depth=3,
                max_nodes=max_results * 3,
                follow_temporal=True,
                follow_semantic=True,
                follow_causal=True,
                follow_entity=True,
            )

        # Intent-aware weight adjustment
        # (MAGMA Eq. 6: phi(r, T_q) = w_Tq^T * 1_r)
        query_lower = query_text.lower()
        intent_type = intent or self._classify_intent(query_lower)

        # Score anchors by intent-aligned relevance
        scored_anchors = self._score_by_intent(anchor_nodes, query_lower, intent_type)
        best_anchors = [n.node_id for n, _ in scored_anchors[:max(5, max_results)]]

        # BFS subgraph from best anchors
        sub_nodes, sub_edges = self.graph.bfs_subgraph(best_anchors, constraints)

        # Deduplicate + sort by relevance
        seen: Set[str] = set()
        unique_nodes: List[EventNode] = []
        for n in sorted(sub_nodes, key=lambda x: -self._node_intent_score(x, query_lower, intent_type)):
            if n.node_id not in seen:
                seen.add(n.node_id)
                unique_nodes.append(n)

        # Limit to max_results
        unique_nodes = unique_nodes[:max_results]

        # 5. Format context
        from .answer_formatter import AnswerFormatter
        formatter = AnswerFormatter()
        context = formatter.format_context(
            unique_nodes, sub_edges, query_text)

        self.stats["queries_run"] += 1
        return context, unique_nodes, sub_edges

    def _classify_intent(self, query_lower: str) -> str:
        """Classify query intent: 'why', 'when', 'entity', or 'general'."""
        if any(w in query_lower for w in ["why", "cause", "because", "reason",
                                            "led to", "result", "impact", "effect"]):
            return "why"
        if any(w in query_lower for w in ["when", "before", "after", "during",
                                            "timeline", "sequence", "first", "then",
                                            "later", "ago", "yesterday", "today"]):
            return "when"
        if any(w in query_lower for w in ["who", "whom", "which person",
                                            "which people"]):
            return "entity"
        return "general"

    def _score_by_intent(
        self,
        nodes: List[EventNode],
        query_lower: str,
        intent: str,
    ) -> List[Tuple[EventNode, float]]:
        """Score nodes by intent-aligned relevance (MAGMA Eq. 5)."""
        scored = []
        for node in nodes:
            score = self._node_intent_score(node, query_lower, intent)
            scored.append((node, score))
        return sorted(scored, key=lambda x: -x[1])

    def _node_intent_score(self, node: EventNode, query_lower: str, intent: str) -> float:
        """Calculate alignment score for a node given query intent.

        Maps to MAGMA Eq. 5: S(n_j|n_i,q) = exp(lambda1*phi(type(e_ij), T_q)
        + lambda2*sim(n_j, q))
        """
        score = 0.0
        content_lower = node.content.lower()

        # Semantic affinity: query term overlap
        query_terms = set(query_lower.split())
        content_terms = set(content_lower.split())
        overlap = query_terms & content_terms
        score += len(overlap) * 2.0

        # Intent-specific boost
        if intent == "why":
            # Boost nodes with causal keywords
            causal_kw = {"because", "cause", "since", "therefore", "thus",
                         "consequently", "led to", "resulted", "due to"}
            if causal_kw & content_terms:
                score += 5.0
        elif intent == "when":
            # Boost nodes with temporal markers
            if node.timestamp:
                recency_hours = (datetime.now() - node.timestamp).total_seconds() / 3600
                score -= recency_hours * 0.01
        elif intent == "entity":
            entities = set(node.attributes.get("entities", []))
            query_entities = set(self._extract_entities(query_lower))
            shared = entities & query_entities
            score += len(shared) * 3.0

        # Temporal decay: older events get slightly lower score
        if node.timestamp:
            age_days = (datetime.now() - node.timestamp).total_seconds() / 86400
            score -= age_days * 0.1

        return max(score, 0.0)

    # ---- Statistics & Status ---------------------------------------------

    def status(self) -> Dict[str, Any]:
        return {
            "events": self.graph.node_count(),
            "edges": self.graph.edge_count(),
            "vector_entries": self.vector_db.size(),
            "consolidation_queue": self._consolidation_queue.qsize(),
            "stats": dict(self.stats),
        }

    # ---- Persistence ------------------------------------------------------

    def save(self) -> None:
        """Persist memory state to disk."""
        if not self.persist_dir:
            logger.warning("No persist_dir set — skipping save")
            return
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Save graph
        self.graph.save(self.persist_dir / "graph.json")

        # Save vector DB
        if isinstance(self.vector_db, VectorDB):
            try:
                from .vector_db import NumpyVectorDB
                if isinstance(self.vector_db, NumpyVectorDB):
                    self.vector_db.persist(self.persist_dir / "vectors.json")
            except Exception as e:
                logger.warning("Vector DB persistence failed: %s", e)

        # Save metadata
        meta = {
            "last_node_id": self._last_node_id,
            "stats": self.stats,
            "events_since_consolidation": self._events_since_consolidation,
            "saved_at": datetime.now().isoformat(),
        }
        (self.persist_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Memory saved to %s", self.persist_dir)

    def load(self) -> None:
        """Load memory state from disk."""
        if not self.persist_dir or not self.persist_dir.exists():
            logger.info("No saved memory found at %s", self.persist_dir)
            return

        # Load graph
        graph_path = self.persist_dir / "graph.json"
        if graph_path.exists():
            self.graph = GraphDB.load(graph_path)
            logger.info("Loaded graph: %d nodes, %d edges",
                        self.graph.node_count(), self.graph.edge_count())

        # Load vector DB
        vectors_path = self.persist_dir / "vectors.json"
        if vectors_path.exists():
            from .vector_db import NumpyVectorDB
            if isinstance(self.vector_db, NumpyVectorDB):
                self.vector_db = NumpyVectorDB.load(
                    vectors_path, dimension=self._embed_dim)
                logger.info("Loaded vector DB: %d entries", self.vector_db.size())

            # Rebuild FAISS if needed
            from .vector_db import FAISSVectorDB
            if isinstance(self.vector_db, FAISSVectorDB):
                logger.info("FAISS vector DB not persisted to JSON; skipping")

        # Load metadata
        meta_path = self.persist_dir / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                self._last_node_id = meta.get("last_node_id")
                self.stats = meta.get("stats", self.stats)
                self._events_since_consolidation = meta.get(
                    "events_since_consolidation", 0)
            except Exception as e:
                logger.warning("Failed to load metadata: %s", e)

        # Rebuild last_node by finding the most recent event
        if self._last_node_id is None or not self.graph.has_node(self._last_node_id):
            all_nodes = self.graph.all_nodes()
            if all_nodes:
                latest = max(all_nodes, key=lambda n: n.timestamp)
                self._last_node_id = latest.node_id

        logger.info("Memory loaded from %s", self.persist_dir)
