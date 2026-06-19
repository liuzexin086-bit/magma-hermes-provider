"""
Keyword Enricher — Lightweight keyword extraction and query enrichment.

Simplified version of MAGMA's keyword enrichment module.
Uses TF-IDF-like heuristics rather than an external NLP library.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Common English stop words filtered during extraction
_STOP_WORDS: set = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "over", "after",
    "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "can", "could", "shall",
    "should", "may", "might", "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them", "my", "your", "his", "its", "our",
    "their", "this", "that", "these", "those", "what", "which", "who",
    "whom", "when", "where", "why", "how", "all", "each", "every", "both",
    "few", "more", "most", "some", "any", "no", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "also", "now", "then",
    "here", "there", "please", "thanks", "ok", "okay", "yes", "no",
}

# Domain-specific boost words for agentic memory
_DOMAIN_BOOST: Dict[str, float] = {
    "remember": 1.5, "recall": 1.5, "earlier": 1.3, "before": 1.3,
    "previously": 1.5, "last": 1.2, "previous": 1.3, "mention": 1.3,
    "said": 1.2, "told": 1.2, "asked": 1.2, "discussed": 1.3,
    "decided": 1.3, "planned": 1.3, "created": 1.2, "changed": 1.2,
    "fixed": 1.2, "updated": 1.2, "set up": 1.2, "installed": 1.2,
    "configured": 1.3, "deployed": 1.2,
}


class KeywordEnricher:
    """Lightweight keyword extraction without external NLP dependencies."""

    def extract(self, text: str, max_keywords: int = 15) -> List[str]:
        """Extract significant keywords from text."""
        text_lower = text.lower()
        # split on non-alphanumeric (keep hyphens and underscores)
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", text_lower)
        # filter stop words and short tokens
        candidates = [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]

        # score by frequency with domain boost
        freq: Dict[str, float] = {}
        for tok in candidates:
            score = freq.get(tok, 0) + 1.0
            if tok in _DOMAIN_BOOST:
                score += _DOMAIN_BOOST[tok] - 1.0
            freq[tok] = score

        # sort by frequency desc
        sorted_kw = sorted(freq.items(), key=lambda x: -x[1])
        return [kw for kw, _ in sorted_kw[:max_keywords]]

    def enrich_query(self, query: str) -> str:
        """Enrich query text with extracted keywords for better embedding."""
        keywords = self.extract(query, max_keywords=5)
        if not keywords:
            return query
        # append keywords to help embedding capture key terms
        enriched = f"{query} keywords: {' '.join(keywords)}"
        return enriched

    def enrich_content(self, content: str,
                       metadata: Optional[Dict[str, Any]] = None
                       ) -> str:
        """Enrich content text for embedding."""
        _ = metadata  # reserved for future use
        keywords = self.extract(content, max_keywords=8)
        if not keywords:
            return content
        enriched = f"{content} keywords: {' '.join(keywords)}"
        return enriched



