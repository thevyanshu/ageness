# Benchmark Framework — Phases 1-3

**Location:** `benchmark/`

## Purpose

Evaluates persistent cognitive memory architectures against transcript replay baselines. The core hypothesis: dynamically reconstructed cognitive state from structured memory outperforms transcript replay in latency, token usage, context stability, and long-horizon coherence.

## Systems

### System A — Transcript Replay Baseline (`systems/transcript_replay.py`)
Traditional chatbot architecture. Three truncation strategies:

| Strategy | Behavior |
|----------|----------|
| `LAST_N` | Keeps the last N messages. Linear token growth. |
| `ROLLING_SUMMARY` | Maintains summary of older messages + last N. |
| `TOKEN_BUDGET` | Fits recent messages into a fixed token budget. |

### System B — Reconstructed State (`systems/reconstructed_state.py`)
Primary PCAA architecture. Integrates all ageness cognitive modules:

1. `HybridRetrievalSystem` with LM Studio embeddings for semantic retrieval
2. `ActiveContextBuilder` to assemble context from stored memories
3. `AsyncDistillationPipeline` to extract decisions, goals, compressed traces
4. `MemorySalienceEngine` to score each new memory
5. `CognitiveStateReconstruction` for full cognitive state rebuild

Each turn: LLM call → distill → salience score → store → reconstruct.

### System C — Hybrid Retrieval (`systems/hybrid_retrieval.py`)
Industry-style memory architecture. Combines:
- **Rolling summary** of older transcript history
- **Vector retrieval** via LM Studio embeddings (`_cosine_sim` over in-memory store)
- **Transcript-assisted recall** — retrieved chunks appended to context

## Metrics Collected Per Turn

| Metric | Description |
|--------|-------------|
| `input_tokens` / `output_tokens` | Token counts |
| `context_size_tokens` | Size of the built context window |
| `retrieval_latency_ms` | Time for memory retrieval |
| `inference_latency_ms` | Time for LLM call + async pipeline |
| `total_latency_ms` | End-to-end turn latency |
| `memories_retrieved` | Number of items fetched from memory |
| `active_memory_count` | Total stored memories / history |
| `output` | Full LLM response text |
| `hallucination_indicators` | Detected fabrications |
| `contradiction_indicators` | Detected conflicts |

## Evaluation Scores

| Score | Formula | Purpose |
|-------|---------|---------|
| `decision_recall` | matched / total | Fraction of expected decisions recalled |
| `avg_context_size_tokens` | mean per turn | Context compactness |
| `latency_p50/p95/p99` | percentile | Latency stability over session length |
| `context_efficiency_ratio` | decisions / total_tokens | Useful output per token |
| `hallucination_events` | count | Memory corruption |
| `contradiction_events` | count | Self-contradiction |
| `hallucination_count` | detector | Hallucinated memory events |
| `contradiction_count` | detector | Contradiction events |
| `temporal_dependency_accuracy` | pairwise ordering | Chronological reasoning quality |
| `compression_ratio` | avg_context / avg_input | Info density (lower = more compressed) |
| `context_stability` | 1/(1+CV) | Stability over session duration |

## Architecture Comparison

The `compare_architectures()` method runs the same scenario through multiple architectures and returns side-by-side scores:

```python
results = await runner.compare_architectures(scenario, [
    BenchmarkConfig(architecture=TRANSCRIPT_REPLAY, ...),
    BenchmarkConfig(architecture=HYBRID_RETRIEVAL, ...),
    BenchmarkConfig(architecture=RECONSTRUCTED_STATE, ...),
])
# results["transcript_replay"]["scores"]["decision_recall"]
# results["reconstructed_state"]["scores"]["avg_context_size_tokens"]
```

## Test Coverage

| File | Tests | What |
|------|-------|------|
| `tests/test_models.py` | 9 | Data models |
| `tests/test_simulator.py` | 6 | Scenario generator |
| `tests/test_transcript_replay.py` | 7 | System A |
| `tests/test_hybrid_retrieval.py` | 6 | System C |
| `tests/test_reconstructed_state.py` | 6 | System B |
| `tests/test_runner.py` | 5 | Runner + evaluation |
| `tests/test_runner_phase2.py` | 4 | Phase 2 integration |
| `tests/test_metrics_evaluator.py` | 6 | Phase 3 advanced metrics |
| **Total** | **50** | All passing |

## Generated Scenarios

- **Architecture Planning** (14 turns) — DDD, Kafka, Saga, Stripe, Istio, strangler fig
- **Debugging Session** (10+ turns) — production bug diagnosis
- **Long Session** (configurable) — 100+ turns across 8 engineering topics

## Running

```bash
# benchmark tests only
python -m pytest benchmark/tests/ -v

# full suite (ageness + benchmark + real-world)
python -m pytest tests/ benchmark/tests/

# with real-world LLM
$env:RUN_REAL_WORLD = "1"
python -m pytest tests/ benchmark/tests/
```
