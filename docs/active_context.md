# Active Context Builder

**Location:** `src/ageness/cognition/active_context/builder.py`

## Purpose

Dynamically assembles a minimal working context from memory, scoped to current goals and bounded by a token budget. Runs at graph node entry to inject relevant context before LLM execution.

## Priority Order

Items are sorted into categories and consumed in strict priority:

1. **Goals** — active objectives (highest salience first)
2. **Decisions** — past choices with rationale
3. **Episodes** — execution traces and events
4. **Facts** — semantic knowledge (lowest priority)

Within each category, items are sorted by `salience.composite` descending.

## Token Budgeting

- Default budget: **4000 tokens**
- Estimate: `len(content) // 4` (rough 4 chars per token, minimum 1)
- Items that would exceed the remaining budget are **skipped**, not truncated
- Budget is checked before inclusion, so high-priority items always get first access

## API

```python
builder = ActiveContextBuilder(
    retrieval=hybrid_retrieval_system,
    max_tokens=4000,
)

window = await builder.build_context(
    current_state={"input": "..."},
    thread_id="session-1",
    goals=["fix login bug", "deploy to production"],
)

# window.goals       → list of MemoryItem (highest priority)
# window.decisions   → list of MemoryItem
# window.episodes    → list of MemoryItem
# window.facts       → list of MemoryItem (lowest priority)
# window.total_tokens → int (sum of included items)
```

## Test Coverage

10 tests in `tests/test_active_context.py` covering classification, priority ordering, within-category salience sorting, token budget enforcement, empty input, and token estimation.
