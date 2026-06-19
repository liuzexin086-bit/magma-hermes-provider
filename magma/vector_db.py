"""
Vector Database — Lightweight NumPy-based Vector Storage

Supports two backends:
  - numpy (default): in-memory brute-force cosine search, no extra deps
  - faiss: FAISS IVF/index for larger-scale use (optional)

Falls back gracefully if FAISS is not installed.
"""

from __future__ import annotations

import json
import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


@dataclass
class VectorEntry:
    vector_id: str
    vector: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class VectorDB(ABC):
    """Abstract vector database interface."""

    @abstractmethod
    def add(self, vector_id: str, vector: np.ndarray,
            metadata: Optional[Dict[str, Any]] = None) -> bool:
        ...

    @abstractmethod
    def search(self, query: np.ndarray, k: int = 10,
               ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Return [(vector_id, score, metadata), ...] sorted by similarity desc."""
        ...

    @abstractmethod
    def get(self, vector_id: str) -> Optional[VectorEntry]:
        ...

    @abstractmethod
    def delete(self, vector_id: str) -> bool:
        ...

    @abstractmethod
    def size(self) -> int:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...


class NumpyVectorDB(VectorDB):
    """In-memory brute-force cosine similarity search using NumPy.

    For small to medium memory stores (up to ~50K vectors).  Fast enough for
    agentic memory where stores rarely exceed a few thousand events.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._entries: Dict[str, VectorEntry] = {}
        self._matrix: Optional[np.ndarray] = None
        self._ids: List[str] = []
        self._dirty = False

    def add(self, vector_id: str, vector: np.ndarray,
            metadata: Optional[Dict[str, Any]] = None) -> bool:
        assert vector.shape == (self.dimension,), \
            f"Expected dim {self.dimension}, got {vector.shape}"
        self._entries[vector_id] = VectorEntry(
            vector_id=vector_id,
            vector=vector.copy(),
            metadata=metadata or {},
        )
        self._dirty = True
        return True

    def _rebuild(self) -> None:
        self._ids = list(self._entries.keys())
        if not self._ids:
            self._matrix = None
            return
        self._matrix = np.stack([self._entries[i].vector for i in self._ids])
        self._dirty = False

    def search(self, query: np.ndarray, k: int = 10
               ) -> List[Tuple[str, float, Dict[str, Any]]]:
        if not self._entries:
            return []
        if self._dirty:
            self._rebuild()
        # cosine similarity
        q = query.reshape(1, -1).astype(np.float32)
        # normalise
        q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        m_norm = self._matrix / (np.linalg.norm(self._matrix, axis=1, keepdims=True) + 1e-12)
        scores = m_norm @ q_norm.T  # (N, 1)
        scores = scores.flatten()
        top_k = min(k, len(scores))
        indices = np.argsort(-scores)[:top_k]
        return [
            (self._ids[i], float(scores[i]), dict(self._entries[self._ids[i]].metadata))
            for i in indices
        ]

    def get(self, vector_id: str) -> Optional[VectorEntry]:
        return self._entries.get(vector_id)

    def delete(self, vector_id: str) -> bool:
        if vector_id not in self._entries:
            return False
        del self._entries[vector_id]
        self._dirty = True
        return True

    def size(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()
        self._matrix = None
        self._ids = []
        self._dirty = False

    def persist(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "dimension": self.dimension,
            "entries": {vid: {
                "vector": entry.vector.tolist(),
                "metadata": entry.metadata,
                "timestamp": entry.timestamp.isoformat(),
            } for vid, entry in self._entries.items()},
        }
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, dimension: int = 384) -> "NumpyVectorDB":
        if not path.exists():
            return cls(dimension=dimension)
        data = json.loads(path.read_text(encoding="utf-8"))
        db = cls(dimension=data.get("dimension", dimension))
        for vid, edata in data.get("entries", {}).items():
            vec = np.array(edata["vector"], dtype=np.float32)
            db._entries[vid] = VectorEntry(
                vector_id=vid,
                vector=vec,
                metadata=edata.get("metadata", {}),
                timestamp=datetime.fromisoformat(edata["timestamp"])
                if "timestamp" in edata else datetime.now(),
            )
        db._dirty = True
        return db


class FAISSVectorDB(VectorDB):
    """FAISS-backed vector DB for larger stores."""

    def __init__(self, dimension: int = 384):
        assert FAISS_AVAILABLE, "FAISS not installed"
        self.dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)  # inner product (cosine after norm)
        self._ids: List[str] = []
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def add(self, vector_id: str, vector: np.ndarray,
            metadata: Optional[Dict[str, Any]] = None) -> bool:
        vec = vector.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(vec)
        self._index.add(vec)
        self._ids.append(vector_id)
        self._metadata[vector_id] = metadata or {}
        return True

    def search(self, query: np.ndarray, k: int = 10
               ) -> List[Tuple[str, float, Dict[str, Any]]]:
        if self._index.ntotal == 0:
            return []
        q = query.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(q)
        scores, idxs = self._index.search(q, min(k, self._index.ntotal))
        return [
            (self._ids[int(i)], float(scores[0][j]), dict(self._metadata.get(self._ids[int(i)], {})))
            for j, i in enumerate(idxs[0])
        ]

    def get(self, vector_id: str) -> Optional[VectorEntry]:
        # FAISS doesn't store raw vectors by default; rebuild from ids
        idx = self._ids.index(vector_id) if vector_id in self._ids else -1
        if idx < 0:
            return None
        vec = self._index.reconstruct(idx)
        return VectorEntry(
            vector_id=vector_id,
            vector=vec,
            metadata=self._metadata.get(vector_id, {}),
        )

    def delete(self, vector_id: str) -> bool:
        if vector_id not in self._metadata:
            return False
        # FAISS has no cheap removal; just flag
        self._metadata.pop(vector_id, None)
        # rebuild from remaining
        remaining = [(iid, self._index.reconstruct(idx))
                     for idx, iid in enumerate(self._ids) if iid != vector_id]
        self._index.reset()
        self._ids.clear()
        for iid, vec in remaining:
            self._index.add(vec.reshape(1, -1).astype(np.float32))
            self._ids.append(iid)
        return True

    def size(self) -> int:
        return self._index.ntotal

    def clear(self) -> None:
        self._index.reset()
        self._ids.clear()
        self._metadata.clear()


def create_vector_db(backend: str = "auto", dimension: int = 384,
                     persist_path: Optional[str] = None) -> VectorDB:
    """Factory: returns the best available backend."""
    if backend == "faiss" and FAISS_AVAILABLE:
        return FAISSVectorDB(dimension=dimension)
    if backend == "numpy" or backend == "auto":
        return NumpyVectorDB(dimension=dimension)
    if backend == "faiss" and not FAISS_AVAILABLE:
        logger.warning("FAISS not available; falling back to NumPy vector DB")
        return NumpyVectorDB(dimension=dimension)
    raise ValueError(f"Unknown vector DB backend: {backend}")
