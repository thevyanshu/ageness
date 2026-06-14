# Async Distillation Pipeline

**Location:** `src/ageness/cognition/distillation/pipeline.py`

## Purpose

Background cognition system that runs after key execution steps to extract decisions, identify unresolved goals, compress execution traces, and persist structured memories.

## Pipeline Stages

```
State → Extract Decisions → Identify Unresolved Goals → Compress Trace → Build & Score Memories
```

| Stage | Method | Description |
|-------|--------|-------------|
| Decision extraction | `_extract_decisions()` | Scans `state.decisions`, `state.metadata[*].decision`, and assistant messages containing decision keywords. Deduplicates by content prefix. |
| Goal identification | `_identify_unresolved_goals()` | Filters `state.goals` / `state.active_goals`, excluding any with status in `{completed, resolved, done, cancelled, failed}`. Handles both string and dict goal formats. |
| Trace compression | `_compress_trace()` | Builds a compact summary from `input`, `output`, message role counts, and key fields. Returns `None` if no data. |
| Memory building | `_build_memories()` | Converts each decision → `RawMemory(DECISION)`, each goal → `RawMemory(GOAL)`, trace → `RawMemory(EPISODIC)` |

## API

```python
pipeline = AsyncDistillationPipeline(
    salience_engine=MemorySalienceEngine(),
    checkpointer=optional_checkpointer,   # LangGraph checkpointer for state fetch
)

# Pass state directly
result = await pipeline.distill(
    thread_id="session-1",
    state={"input": "...", "decisions": [...], "goals": [...]},
)

# Or let pipeline fetch from checkpointer
result = await pipeline.distill(thread_id="session-1", checkpoint_id="ckpt-123")

# result.decisions         → list of extracted decisions
# result.unresolved_goals  → list of open goals
# result.compressed_trace  → compact string summary
# result.memory_items      → scored RawMemory objects
```

## Data Sources

The pipeline scans these state fields in order:
- `state["decisions"]` or `state["decisions_made"]` — explicit decision list
- `state["metadata"]` — entries with a `"decision"` key
- `state["messages"]` — assistant messages containing decision keywords
- `state["goals"]` or `state["active_goals"]` — goal list

## Test Coverage

13 tests in `tests/test_distillation.py` covering all extraction methods, dedup, goal status filtering, trace compression, memory building, and full pipeline integration.
