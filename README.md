# MAGMA Memory Provider for Hermes Agent

**Multi-Graph Agentic Memory — ACL 2026 Main Conference.**

A pluggable memory provider that brings the [MAGMA](https://arxiv.org/abs/2601.03236) architecture to [Hermes Agent](https://hermes-agent.nousresearch.com).

**v2.1 Update (2026-06-24):** Dynamic classification + query pipeline documented. When MAGMA has no match for a topic, it automatically creates new wiki docs or appends to existing ones. See [Query Pipeline](#query-pipeline-v21--2026-06-24) and [Architecture](#architecture) below.

---

## Features

- **Wiki-style memory vault.** Each conversation turn is distilled and appended to one of **dynamic category docs** (6 baseline, auto-extends on new topics).
- **Timestamped sections.** Every entry carries the decision date. Each doc ends with a **⏳ 决策演进** (Decision Timeline) table.
- **[[Wiki links]]** between related docs for cross-referencing.
- **Intent-aware classification.** Content is automatically classified into the right category by keyword matching.
- **Persistent across sessions.** Wiki docs are stored in `E:\obsidian_hermes\hermes\magma\` and viewable in any Markdown or Obsidian editor.
- **Zero compulsory external dependencies.** Pure Python, no PyTorch/sentence-transformers required.

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
| Every `sync_turn()` | Conversation is distilled → classified → appended to the appropriate wiki doc |
| `prefetch(query)` | Searches wiki docs and injects relevant sections as context |
| `magma_search` tool | Full-text search across all wiki docs |
| `magma_read_note` tool | Read a specific wiki doc or search by keyword |
| `magma_status` tool | Returns wiki doc count, vault path |

**Example — manual query:**

```
Use magma_search to find what we discussed about creep feed.
```

---

## Query Pipeline (v2.1 — 2026-06-24)

When Hermes queries MAGMA for knowledge, it follows this deterministic pipeline:

```
MAGMA search → broaden query → retry
   ↓ not found
 Topic within known domains?
   ↓ Yes → Create new wiki doc / append to existing → persist to vault
   ↓ No  → web_search → Create new wiki doc → persist to vault
   ↓ Nothing found → Report "not found" honestly
```

**Hard constraint:** Never delete existing distilled knowledge or original notes. Only add, never remove.

**Known domains (auto-triggered):** livestock automation, SmartMilk (奶爸机), pig/swine industry, animal health (动保), Shanchuan Biology (山川生物), Anyou (安佑), pig cycle (猪周期), plant essential oils (植物精油), product philosophy.

**session_search (historical transcript recall)** is excluded from the auto-pipeline. It is used only when the user explicitly asks to revisit a past conversation.

---

## Architecture

### Wiki Vault

Instead of the original two-layer design (index + graph), MAGMA now maintains a **wiki vault** at `E:\obsidian_hermes\hermes\magma\`:

```
magma/
├── 奶爸机-曲线设计.md          # SmartMilk curve: formula, knobs, feedback
├── 奶爸机-系统架构.md          # System: data collection, MCU, Flask
├── MAGMA-记忆架构.md           # MAGMA memory evolution & design decisions
├── Hermes-配置工具.md          # Hermes config, performance, deployment
├── 生猪行业-动保.md            # Pig cycle, prices, industry research
├── 养殖自动化-产品哲学.md      # Automation principles, ROI, user research
└── index.json                  # Lightweight index for backward compat
```

### Each Wiki Doc Contains

```
# 奶爸机-曲线设计

## 📐 曲线结构
### 四段式 21 天饲喂曲线（2026-06-17 确认）
...

## 🎛️ 旋钮系统
### #8 教槽过渡 — creepFactor（2026-06-21 决策）
...

## ⏳ 决策演进
| 时间 | 事件 | 影响 |
|------|------|------|
| 06-17 | 首次提出四段式曲线 | 起步→爬坡→平台→控奶 |
| 06-21 | 教槽→趋势反馈 | creepFactor 0.5/1.0/1.4x |

**🔗 相关笔记**：[[奶爸机-系统架构]] | [[养殖自动化-产品哲学]]
```

### Classification

New content is auto-classified by keyword matching into a **dynamic set of category docs**.
The baseline 6 categories are predefined; new topics automatically create new docs:

| Category | Keywords | Target Doc |
|----------|----------|------------|
| Curve Design | 曲线, 教槽, 腹泻, 旋钮, FCR, TW | 奶爸机-曲线设计.md |
| System | parameters.py, MCU, Flask, PyInstaller | 奶爸机-系统架构.md |
| MAGMA | MAGMA, 记忆, vault, 索引 | MAGMA-记忆架构.md |
| Hermes | config, Desktop GUI, profile, sensenova | Hermes-配置工具.md |
| Pig Industry | 猪周期, 猪价, 动保, 日报 | 生猪行业-动保.md |
| Automation | 自动化, ROI, 饲养员, 产品哲学 | 养殖自动化-产品哲学.md |

**When content doesn't match any existing category**, the system creates a new `.md` doc with an appropriate title and appends it to the vault. No content is ever rejected for missing a category.

---

## Configuration

Optional: create `$HERMES_HOME/magma_config.json`:

```json
{
  "vault_dir": "E:/obsidian_hermes/hermes/magma",
  "embedding": "minilm",
  "max_events": 10000
}
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `vault_dir` | string | `E:/obsidian_hermes/hermes/magma` | Path to wiki vault directory |
| `embedding` | `"minilm"`, `"openai"`, `"char"` | `"minilm"` | Embedding backend (minimal effect in wiki mode) |
| `max_events` | integer | 10000 | Maximum events before oldest 10% pruned |

---

## Upgrade from v1.x

If you have v1.x installed (two-layer index + graph), upgrade as follows:

```bash
# 1. Pull the new code
cd ~/.hermes/plugins/magma/
git pull origin main

# 2. (Optional) Archive old individual notes
mkdir -p archive
mv *.md archive/ 2>/dev/null

# 3. The provider auto-detects the new vault structure
# New sync_turn calls will now append to wiki docs instead of creating new files
```

The old index.json is preserved for backward compatibility — search first checks wiki docs, then falls back to the old index.

---

## File Layout

```
magma/
├── __init__.py              # MemoryProvider implementation
├── note_store.py            # Wiki vault management (classify, append, search)
├── distiller.py             # Content distillation from conversation turns
├── graph_db.py              # NetworkX-backed multi-graph (legacy, kept for compat)
├── vector_db.py             # Vector storage (legacy)
├── trg_memory.py            # Core engine (legacy)
├── keyword_enrichment.py    # Keyword extraction
├── temporal_parser.py       # Time parsing
├── answer_formatter.py      # Output formatting
└── requirements.txt         # Dependencies
```

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