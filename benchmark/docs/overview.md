# Benchmark Framework — Phase 1

**Location:** `benchmark/`

## Purpose

Evaluates persistent cognitive memory architectures against transcript replay baselines. The core hypothesis: dynamically reconstructed cognitive state from structured memory outperforms transcript replay in latency, token usage, context stability, and long-horizon coherence.

## Phase 1 Scope

Minimal benchmark harness:

| Component | File | Status |
|-----------|------|--------|
| Data models | `benchmark/models.py` | ✅ |
| Conversation simulator | `benchmark/simulator/generator.py` | ✅ |
| System A — Transcript Replay | `benchmark/systems/transcript_replay.py` | ✅ |
| Benchmark runner | `benchmark/core/runner.py` | ✅ |
| Model tests | `benchmark/tests/test_models.py` | ✅ 9 tests |
| Simulator tests | `benchmark/tests/test_simulator.py` | ✅ 6 tests |
| System A tests | `benchmark/tests/test_transcript_replay.py` | ✅ 7 tests |
| Runner tests | `benchmark/tests/test_runner.py` | ✅ 6 tests |

## Systems

### System A — Transcript Replay Baseline

Three truncation strategies:

| Strategy | Behavior |
|----------|----------|
| `LAST_N` | Keeps the last N messages as context. Linear token growth. |
| `ROLLING_SUMMARY` | Maintains a summary of older messages + last N. Sub-linear growth. |
| `TOKEN_BUDGET` | Fits as many recent messages as possible into a fixed token budget. |

Supports LM Studio for real LLM inference, or falls back to mock responses.

### System B — Reconstructed State (Phase 2)

Integrates ageness cognitive modules: `HybridRetrievalSystem`,
`ActiveContextBuilder`, `AsyncDistillationPipeline`, `MemorySalienceEngine`,
`CognitiveStateReconstruction`.

### System C — Hybrid Retrieval (Phase 2)

Combines rolling summary + vector retrieval + checkpoint traversal.

## Metrics Collected Per Turn

- `input_tokens` / `output_tokens`
- `context_size_tokens` — size of the built context
- `inference_latency_ms` / `retrieval_latency_ms` / `total_latency_ms`
- `memories_retrieved` / `active_memory_count`
- `hallucination_indicators` / `contradiction_indicators`

## Evaluation Scores

| Score | Formula | Purpose |
|-------|---------|---------|
| `decision_recall` | matched / total | Does the system recall decisions? |
| `avg_context_size_tokens` | mean per turn | How compact is the context? |
| `latency_p50/p95/p99` | percentile | Is latency stable over time? |
| `context_efficiency_ratio` | decisions / tokens | Useful output per token |
| `hallucination_events` | count | Memory corruption detection |
| `contradiction_events` | count | Self-contradiction detection |

## Generated Scenarios

### Architecture Planning (14+ turns)
Full architecture planning session: domain-driven design, database-per-service, Kafka, Saga pattern, Stripe Connect, Istio service mesh, strangler fig migration. 7 injected facts, 4 query pairs.

### Debugging Session (10+ turns)
Production bug diagnosis: connection pool leak, race conditions, cache invalidation, auth timeout, DB deadlock. 5 injected facts.

### Long Session (configurable)
100+ turn session covering 8 engineering topics (auth, cache, queue, monitoring, logging, deploy, security, compliance). Cycles through topics with facts at recall intervals.

## Running

```bash
# run all benchmark tests
python -m pytest benchmark/tests/ -v

# run with full suite
python -m pytest tests/ benchmark/tests/
```
