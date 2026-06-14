# ageness

Cognitive memory layer on LangGraph.

## Modules

| Module | What it does |
|--------|-------------|
| **Salience** | Score memories by relevance, importance, novelty |
| **Retrieval** | Multi-strategy retrieval: semantic, temporal, recency |
| **Active Context** | Assemble context from stored memories |
| **Distillation** | Extract decisions, goals, compressed traces from conversation |
| **Reconstruction** | Rebuild full cognitive state from stored memories |
| **Workflow** | 4-node LangGraph pipeline with interrupt/resume |

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pytest tests/
```

See `benchmark/` for comparing memory architectures (Transcript Replay, Hybrid Retrieval, Reconstructed State). Docs in `docs/` and `benchmark/docs/`.
