# LangGraph Workflow

**Location:** `src/ageness/graph/workflow.py`

## Purpose

Wires all cognitive modules into a LangGraph `StateGraph` for end-to-end agent runs with memory. The graph orchestrates retrieval, active context building, agent execution, distillation, and memory storage in a single linear pipeline.

## Pipeline

```
START → retrieve_context → agent_step → distill → store_memories → END
```

| Node | Module | Function |
|------|--------|----------|
| `retrieve_context` | `ActiveContextBuilder` | Retrieves memories related to the input goals and builds a `ContextWindow`; flattens into a context string. |
| `agent_step` | n/a (callable) | Calls a user-provided `predict(user_input, context) -> str` function, or echoes the input as a placeholder. Appends the output to `messages`. |
| `distill` | `AsyncDistillationPipeline` | Extracts decisions from messages/metadata, identifies unresolved goals, compresses the trace, and builds `RawMemory` objects. |
| `store_memories` | `MemorySalienceEngine` | Scores each `RawMemory` for salience, stores in the Store under `("*", memory_type)` namespace, then runs decay and consolidation on episodic memories. |

## API

```python
graph = build_workflow(
    retrieval=hybrid_retrieval_system,      # optional, defaults to InMemoryStore
    context_builder=active_context_builder,  # optional
    salience=salience_engine,               # optional
    distillation=distillation_pipeline,     # optional
    predict=mock_llm_callable,              # optional, defaults to echo
)

app = compile_workflow(
    graph=graph,
    checkpointer=in_memory_saver,           # optional
    store=in_memory_store,                  # optional
)

result = await app.ainvoke(
    state_dict,
    {"configurable": {"thread_id": "..."}},
)
```

## AgentState

| Field | Type | Notes |
|-------|------|-------|
| `input` | `str` | User query |
| `context` | `str \| None` | Flattened context string (set by `retrieve_context`) |
| `output` | `str \| None` | Agent response (set by `agent_step`) |
| `thread_id` | `str` | Session identifier |
| `metadata` | `list[dict]` | Accumulated metadata via `operator.add` |
| `goals` | `list[str]` | Current goals (enriched by `distill`) |
| `messages` | `list[dict]` | Message history (appended by `agent_step`) |
| `distilled` | `DistillationResult \| None` | Distillation output (set by `distill`) |
| `context_window` | `ContextWindow \| None` | Full window object (set by `retrieve_context`) |

## Test Coverage

3 tests in `tests/test_workflow.py` covering compilation, end-to-end run, and memory storage verification.
