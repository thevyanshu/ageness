# Project Status

> Tracking file for the ageness cognitive memory project.
> Update this file as components are implemented, tested, or changed.

---

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Done |
| 🔶 | In progress |
| ❌ | Not started |
| 🚫 | Cancelled / removed |

---

## Foundation

| Component | Status | Notes |
|-----------|--------|-------|
| Project scaffold (dirs, pyproject.toml) | ✅ | |
| `__init__.py` files | ✅ | All packages |
| `ARCHITECTURE.md` | ✅ | Full architecture document |
| `STATUS.md` | ✅ | This file |
| `.gitignore` | ✅ | |
| Dev tooling (ruff, mypy, pytest) | ✅ | Config in pyproject.toml, verified working |
| Package rename `harness` → `ageness` | ✅ | |

---

## Core Infrastructure

| Component | Status | Notes |
|-----------|--------|-------|
| `src/ageness/memory/models.py` — data models | ✅ | MemoryItem, CognitiveState, ContextWindow, etc. |
| `src/ageness/graph/workflow.py` — base LangGraph | ✅ | 4-node pipeline, salience scoring, store integration |

---

## Cognitive Modules

### Active Context Builder

| Component | Status | Notes |
|-----------|--------|-------|
| `src/ageness/cognition/active_context/builder.py` | ✅ | Full implementation |
| Token budgeting / compression | ✅ | 4-char-per-token estimate, budget-aware inclusion |
| Priority-based inclusion logic | ✅ | Goals > Decisions > Episodes > Facts, sorted by salience |

### Async Distillation Pipeline

| Component | Status | Notes |
|-----------|--------|-------|
| `src/ageness/cognition/distillation/pipeline.py` | ✅ | Full implementation |
| Decision extractor | ✅ | From state.decisions, metadata, assistant messages |
| Trace compression | ✅ | Input/output summary + message role counts |
| Unresolved goal identifier | ✅ | Filters out completed/resolved/cancelled/failed |

### Memory Salience Engine

| Component | Status | Notes |
|-----------|--------|-------|
| `src/ageness/cognition/salience/engine.py` | ✅ | Full implementation |
| Salience scoring function | ✅ | Type-based, freshness, metadata boosts |
| Decay scheduler | ✅ | Exponential decay, threshold pruning |
| Consolidation / merging logic | ✅ | Jaccard similarity merge |

### Hybrid Retrieval System

| Component | Status | Notes |
|-----------|--------|-------|
| `src/ageness/cognition/retrieval/hybrid.py` | ✅ | Full implementation |
| Semantic retrieval (keyword overlap) | ✅ | Token scoring with optional embedding model |
| Temporal retrieval | ✅ | Filter by `created_at` range |
| Checkpoint traversal | ✅ | LangGraph state history iteration |
| Fusion / dedup / ranking | ✅ | Dedup by key, sort by salience desc |

### Cognitive State Reconstruction

| Component | Status | Notes |
|-----------|--------|-------|
| `src/ageness/cognition/reconstruction/reconstructor.py` | ✅ | Full implementation |
| Goal reconstruction | ✅ | Filters resolved goals, returns active only |
| Decision surfacing | ✅ | With rationale, sorted by salience |
| Dependency resolution | ✅ | Cross-references decision/goal key relationships |

---

## LangGraph Wiring

| Component | Status | Notes |
|-----------|--------|-------|
| State schema definitions | ✅ | AgentState with context, distilled, messages, decisions, goals |
| Graph topology (nodes / edges) | ✅ | retrieve_context → agent_step → distill → store_memories |
| Checkpointer integration | ✅ | InMemorySaver default, pass-through arg in compile_workflow |
| Store integration | ✅ | InMemoryStore default, accessible via build_workflow dependencies |
| Interrupt / resume handling | 🔶 | Not yet wired |

---

## Tests

| Component | Status | Notes |
|-----------|--------|-------|
| `tests/test_active_context.py` | ✅ | 10 tests: classification, priority, budget, tokens |
| `tests/test_distillation.py` | ✅ | 13 tests: extraction, dedup, goals, compression, memories |
| `tests/test_salience.py` | ✅ | 10 tests covering scoring, decay, novelty, boosts |
| `tests/test_retrieval.py` | ✅ | 10 tests: semantic, temporal, fusion, filtering |
| `tests/test_reconstruction.py` | ✅ | 11 tests: goals, decisions, tasks, dependencies, confidence |
| `tests/test_workflow.py` | ✅ | 3 tests: compile, end-to-end run, memory storage |

---

## Notes & Decisions

| Date | Decision |
|------|----------|
| 2026-06-15 | Wired LangGraph workflow: 4-node pipeline with retrieval, agent step, distillation, and memory storage |
