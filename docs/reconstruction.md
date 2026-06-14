# Cognitive State Reconstruction

**Location:** `src/ageness/cognition/reconstruction/reconstructor.py`

## Purpose

Rebuilds an agent's full cognitive state from distributed memory in the Store. Produces a structured `CognitiveState` object with active goals, relevant decisions, unresolved tasks, and contextual dependencies.

## Reconstruction Pipeline

```
Query → Retrieve GOAL memories → Filter resolved → Reconstructed goals
      → Retrieve DECISION memories → Surface with rationale → Relevant decisions
      → Retrieve EPISODIC+GOAL → Filter completed → Unresolved tasks
      → Scan all memories → Cross-reference keys → Dependencies
      → Compute confidence → CognitiveState
```

## API

```python
reconstructor = CognitiveStateReconstruction(retrieval=hybrid_retrieval_system)

state = await reconstructor.reconstruct(
    thread_id="session-1",
    query="fix login bug",
)

# state.active_goals              → [{"goal": "...", "status": "active", "salience": 0.9}, ...]
# state.relevant_decisions        → [{"decision": "...", "rationale": "...", "salience": 0.8}, ...]
# state.unresolved_tasks          → [{"task": "...", "status": "active", "salience": 0.3}, ...]
# state.contextual_dependencies   → {"related_goal_keys": [...], "related_decision_keys": [...], ...}
# state.reconstruction_confidence → 0.0 - 1.0
```

## Confidence Score

Weighted formula based on data density:

| Factor | Weight | Max at |
|--------|--------|--------|
| Goals found | 35% | 5+ goals |
| Decisions found | 30% | 5+ decisions |
| Tasks found | 35% | 5+ tasks |

Returns 0.0 when no memories exist.

## Test Coverage

11 tests in `tests/test_reconstruction.py` covering goal reconstruction, decision surfacing, task resolution, dependency resolution, confidence scoring, and sorting.
