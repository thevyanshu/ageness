# Hybrid Retrieval System

**Location:** `src/ageness/cognition/retrieval/hybrid.py`

## Purpose

Combines multiple retrieval strategies to find relevant memories from the Store and Checkpointer. The fusion layer deduplicates and ranks results by salience.

## Strategies

| Strategy | Source | Behavior |
|----------|--------|----------|
| **Semantic** | Store | Token overlap scoring; uses embedding model if available |
| **Temporal** | Store | Filters by `created_at` within a time range |
| **Checkpoint** | Checkpointer | Traverses LangGraph state history for a thread |
| **Graph** | — | Placeholder for relationship traversal |

## API

```python
retrieval = HybridRetrievalSystem(
    store=InMemoryStore(),
    embedding_model=None,      # optional: any embedder with .embed_query()
    checkpointer=None,         # optional: for checkpoint traversal
)

query = RetrievalQuery(
    text="deploy production",
    memory_types=[MemoryType.EPISODIC],   # optional filter
    max_results=10,
    min_salience=0.3,                     # optional threshold
    temporal_range=(start, end),          # optional time window
    metadata={"user_id": "alice"},        # passed to namespace resolution
)

# Default strategies: semantic + temporal
results = await retrieval.retrieve(query)

# Or pick specific strategies
results = await retrieval.retrieve(query, strategies=[RetrievalStrategy.SEMANTIC])
```

## Fusion Logic

- Deduplicates by `MemoryItem.key`
- Sorts descending by `salience.composite`
- Applies `min_salience` and `max_results` from query

## Namespace Convention

Items are stored under `(user_id, memory_type)` namespaces. The `metadata.user_id` field in `RetrievalQuery` determines which user's store items to search. Defaults to `"*"` (wildcard scope).

## Test Coverage

10 tests in `tests/test_retrieval.py` covering semantic scoring, temporal filtering, fusion dedup, salience sorting, type filtering, threshold filtering, and edge cases.
