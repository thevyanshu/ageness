# Memory Salience Engine

**Location:** `src/ageness/cognition/salience/engine.py`

## Purpose

Determines what should persist, decay, or be ignored in long-term memory. Every memory item is scored on three axes, then a weighted composite decides its fate.

## Scoring Axes

| Axis | Description | Influenced By |
|------|-------------|---------------|
| **Relevance** | How pertinent to current context | Memory type (GOAL > DECISION > EPISODIC > SEMANTIC), recency, `active`/`direct_match`/`error` metadata |
| **Importance** | Inherent significance | Memory type, `error`/`user_priority` flags, task count |
| **Novelty** | How distinct from existing items | Jaccard word overlap against existing memory content |

Composite = `relevance_weight × relevance + importance_weight × importance + novelty_weight × novelty`. Default weights: 0.4 / 0.35 / 0.25. Weights are normalized to sum to 1.0.

## API

```python
engine = MemorySalienceEngine(
    relevance_weight=0.4,
    importance_weight=0.35,
    novelty_weight=0.25,
    decay_rate=0.05,
    decay_threshold=0.1,
    consolidation_similarity=0.6,
)

# Score a single memory item
score = await engine.score(item, existing_items=[...])

# Sync convenience for tests (no await needed)
score = engine.score_sync(item)

# Apply exponential decay to a store namespace
removed = await engine.decay(store, namespace)

# Merge similar items in a namespace
merged = await engine.consolidate(store, namespace)
```

## Decay

Exponential decay function: `score(t) = score_0 × exp(-decay_rate × hours_since_access)`. Items below `decay_threshold` are pruned automatically.

## Consolidation

Pairs of items with Jaccard similarity ≥ `consolidation_similarity` are merged. The merged item keeps the higher composite salience, the earliest creation time, and both original keys in a `merged_from` list.

## Test Coverage

10 tests in `tests/test_salience.py` covering all scoring axes, novelty drop, metadata boosts, freshness decay, weight normalization, and output bounds.
