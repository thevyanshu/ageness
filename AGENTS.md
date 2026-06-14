# AGENTS.md — ageness

## Project identity

**ageness** = cognitive memory layer on LangGraph. Package imports: `ageness.*`. Python 3.11+.

## Hard constraint

No custom orchestration, checkpointing, state machines, or graph runtimes. Every graph edge, loop, interrupt, retry, and fan-out uses LangGraph primitives directly. Innovation goes only into the cognitive layer (`src/ageness/cognition/`).

## Commands

Run these inside `.venv`:
```
.venv\Scripts\activate
pip install -e ".[dev]"          # first time or after dep changes
ruff check src\ageness            # lint
ruff check --fix src\ageness      # auto-fix
pytest tests\                     # all tests
pytest tests\test_salience.py -v  # single test file
ruff check benchmark\              # lint benchmark
pytest benchmark\tests\ -v        # benchmark tests
pytest tests\ benchmark\tests\    # full suite
```

Every change follows: `ruff check → pytest → git commit`. Never skip tests. Commit after every logical change, no matter how small — a real developer commits frequently for quick rollback. Never batch unrelated changes into one commit.

## Package layout

```
src/ageness/
├── memory/models.py          — Pydantic models (MemoryItem, CognitiveState, etc.)
├── graph/workflow.py         — LangGraph AgentState, build/compile_workflow
└── cognition/
    ├── active_context/       — Context assembly from retrieval
    ├── distillation/         — Background trace compression & decision extraction
    ├── salience/             — Scoring, decay, consolidation (IMPLEMENTED)
    ├── retrieval/            — Multi-strategy memory retrieval
    └── reconstruction/       — Full cognitive state rebuild
docs/                         — Module docs updated alongside implementation
tests/                        — Mirrors src layout
```

## Architecture

- **LangGraph** is the operating kernel, not the innovation layer.
- **Checkpointer** = thread-scoped short-term memory (PostgresSaver in prod).
- **Store** = cross-thread long-term memory. Namespace convention:
  `(user_id, "episodic")`, `(user_id, "semantic")`, `(user_id, "goals")`, `(user_id, "decisions")`.
- Access Store via `runtime.store` (the `Runtime` parameter), never directly.
- All `Timestamp` fields in models must be timezone-aware: `datetime.now(timezone.utc)`, not `datetime.utcnow()`.

## Testing

- `asyncio_mode = auto` in pytest config — async tests work without decorators.
- Use `score_sync()` on `MemorySalienceEngine` for synchronous test convenience.
- Test files import from `ageness.*` (editable install required).

## Documentation

Every implemented module gets a `docs/<module>.md` file. Update docs in the same commit as the implementation — never as an afterthought.

## Commits

Small, focused, one logical change per commit. Run lint + tests before every commit. Descriptive message body. Commit allows quick rollback if something breaks.
