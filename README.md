# MAGMA Memory Provider for Hermes Agent

**Multi-Graph Agentic Memory — ACL 2026 Main Conference.**

A pluggable memory provider that brings the [MAGMA](https://arxiv.org/abs/2601.03236) architecture to [Hermes Agent](https://hermes-agent.nousresearch.com). Every conversation turn is automatically recorded as a graph node, interlinked across four orthogonal relation types — temporal, semantic, causal, and entity — enabling intent-aware retrieval via graph traversal rather than flat keyword search.

---

## Features

- **Automatic event ingestion.** Each user/assistant turn is stored as a structured event node with content, embedding, timestamp, entities, and keywords.
- **Four orthogonal relation graphs.** `TEMPORAL` (strict time chain), `SEMANTIC` (cosine-similarity clustering), `CAUSAL` (LLM-inferred entailment), `ENTITY` (cross-event entity linking).
- **Intent-aware retrieval.** Queries are classified as *Why*, *When*, *Entity*, or *General* — traversal weights shift dynamically to prioritise the relevant edge type (MAGMA Eq. 5–6).
- **Dual-stream memory evolution.** Fast Path ingests events synchronously (vector index + temporal backbone); Slow Path asynchronously infers causal and entity links in the background.
- **Zero compulsory external dependencies.** Ships with a character n-gram embedder and NumPy-based brute-force vector search. Optionally use `sentence-transformers`, OpenAI embeddings, or FAISS for larger stores.
- **Persistent across sessions.** Graph state, vectors, and metadata are serialised to `$HERMES_HOME/magma/` and restored on restart.
- **Coexists with built-in memory.** The existing `memory` tool (`MEMORY.md` / `USER.md`) operates independently; MAGMA adds an additional graph layer on top.

---

## Quick Start

```bash
# 1. Copy the plugin to your Hermes user plugins directory
cp -r magma ~/.hermes/plugins/magma/

# 2. Install core dependencies
pip install numpy networkx

# 3. Activate the provider in Hermes config
hermes config set memory.provider magma

# 4. Start a new session
hermes --continue
# or /reset in an active session
```

**Windows (PowerShell):**

```powershell
Copy-Item -Recurse magma $env:USERPROFILE\.hermes\plugins\magma\
pip install numpy networkx
hermes config set memory.provider magma
```

---

## Usage

Once activated, the provider works transparently:

| Trigger | Behaviour |
|---------|-----------|
| Every `sync_turn()` | User + assistant messages are ingested as `EventNode`s with auto-extracted entities, keywords, temporal links, and semantic links. |
| `prefetch(query)` | Before each LLM call, MAGMA runs intent-aware graph traversal and injects relevant context as a structured, provenance-tagged block. |
| `magma_search` tool | Manual graph query. Supports `intent` override and `max_results` limit. |
| `magma_status` tool | Returns event count, edge count, vector DB size, consolidation status. |

**Example — manual query:**

```
Use magma_search to find what we discussed about PostgreSQL.
```

**With explicit intent hint:**

```
Use magma_search with intent=why to investigate why we chose PostgreSQL over MySQL.
```

---

## Configuration

Optional: create `$HERMES_HOME/magma_config.json`:

```json
{
  "persist_dir": "/path/to/custom/storage",
  "embedding": "minilm",
  "max_events": 10000,
  "enable_consolidation": true
}
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `embedding` | `"minilm"`, `"openai"`, `"char"` | `"minilm"` | Embedding backend. `minilm` uses sentence-transformers if installed, otherwise falls back to `char`. |
| `persist_dir` | string | `$HERMES_HOME/magma/` | Directory for graph state persistence. |
| `max_events` | integer | 10000 | Maximum events before oldest 10% are pruned. |
| `enable_consolidation` | bool | `true` | Enable background causal/entity inference. |

### Embedding Backends

| Backend | Quality | Dependencies |
|---------|---------|--------------|
| `char` | Low (fallback) | None |
| `minilm` (default) | Medium | `sentence-transformers` + PyTorch (optional) |
| `openai` | High | `openai` + `OPENAI_API_KEY` env var |

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    Hermes Agent                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  MemoryManager                                        │  │
│  │  ├─ builtin (MEMORY.md / USER.md)                    │  │
│  │  └─ MAGMA (multi-graph)                               │  │
│  └──────────────────────────────────────────────────────┘  │
│           │                        │                        │
│     prefetch()               sync_turn()                    │
│           ▼                        ▼                        │
│  ┌──────────────────────────────────────────────────┐       │
│  │  TRGMemory Engine                                 │       │
│  │                                                   │       │
│  │  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │       │
│  │  │ Graph DB │  │ VectorDB │  │ Query Engine   │  │       │
│  │  │NetworkX  │  │ NumPy/   │  │ Intent classify│  │       │
│  │  │4 graphs  │  │ FAISS    │  │ Beam Search    │  │       │
│  │  └──────────┘  └──────────┘  └───────────────┘  │       │
│  │                                                   │       │
│  │  Fast Path: sync → add_event()                   │       │
│  │  Slow Path: async → consolidate()                │       │
│  └──────────────────────────────────────────────────┘       │
└────────────────────────────────────────────────────────────┘
```

### File Layout

```
magma/
├── __init__.py              # MemoryProvider implementation (Hermes lifecycle)
├── graph_db.py              # NetworkX-backed multi-graph (4 edge types)
├── vector_db.py             # Vector storage (NumPy brute-force / FAISS)
├── trg_memory.py            # Core engine: ingestion, traversal, consolidation
├── keyword_enrichment.py    # TF-IDF-like keyword extraction
├── temporal_parser.py       # Relative time expression resolution
└── answer_formatter.py      # Structure-aware context linearization
```

---

## Dependencies

| Package | Required | Purpose |
|---------|----------|---------|
| `numpy` | Yes | Vector operations |
| `networkx` | Yes | In-memory graph storage |
| `sentence-transformers` | Optional | Local semantic embeddings (`minilm` backend) |
| `openai` | Optional | OpenAI embedding API (`openai` backend) |
| `faiss-cpu` | Optional | Faster vector search for large stores |

---

## Paper Reference

**MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents**  
*Dongming Jiang, Yi Li, Guanpeng Li, Bingzhe Li* — **ACL 2026 Main**  

- Paper: https://arxiv.org/abs/2601.03236  
- Original implementation: https://github.com/FredJiang0324/MAGMA  

If you find this work useful, please consider citing the original paper:

```bibtex
@misc{jiang2026magma,
  title={MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents},
  author={Jiang, Dongming and Li, Yi and Li, Guanpeng and Li, Bingzhe},
  year={2026},
  eprint={2601.03236},
  archivePrefix={arXiv},
}
```

---

## License

MIT
