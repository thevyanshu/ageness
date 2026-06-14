from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from benchmark.metrics.evaluator import AdvancedEvaluator
from benchmark.models import (
    BenchmarkConfig,
    BenchmarkRun,
    EvaluationScore,
    MemoryArchitecture,
    MetricsSnapshot,
    Scenario,
    SystemAResult,
)
from benchmark.systems.hybrid_retrieval import HybridRetrievalSystemBench
from benchmark.systems.reconstructed_state import ReconstructedStateSystem
from benchmark.systems.transcript_replay import TranscriptReplaySystem

_CONTRADICTION_PAIRS = [
    ("use", "avoid"),
    ("enable", "disable"),
    ("should", "should not"),
    ("increase", "decrease"),
    ("start", "stop"),
    ("allow", "block"),
    ("accept", "reject"),
    ("forward", "discard"),
]


def _post_process_turn(
    metrics: MetricsSnapshot,
    scenario: Scenario,
    previous_outputs: list[str],
    recalled_fact_indices: set[int],
    turn_idx: int,
) -> None:
    output_lower = metrics.output.lower()

    new_facts = 0
    for i, fact in enumerate(scenario.facts):
        if i not in recalled_fact_indices and fact.content.lower() in output_lower:
            new_facts += 1
            recalled_fact_indices.add(i)
    metrics.facts_recalled_this_turn = new_facts

    for fact in scenario.facts:
        fact_lower = fact.content.lower()
        for prefix in ("use ", "set up", "implement", "deploy", "run"):
            if fact_lower.startswith(prefix):
                tool = fact_lower[len(prefix):].split()[0].rstrip(",. ")
                if tool and len(tool) > 2:
                    neg_pat = rf"\b(avoid|don't|not|instead of|reject)\b.*\b{re.escape(tool)}\b"
                    if re.search(neg_pat, output_lower):
                        metrics.hallucination_indicators.append(
                            f"turn {metrics.turn_id}: contradicts fact '{fact.fact_id}' "
                            f"('{tool}' should be used per scenario)"
                        )

    for word_a, word_b in _CONTRADICTION_PAIRS:
        if word_a in output_lower and word_b in output_lower:
            metrics.contradiction_indicators.append(
                f"turn {metrics.turn_id}: conflicting terms '{word_a}'/'{word_b}'"
            )

    if previous_outputs:
        for prev in previous_outputs[-5:]:
            prev_lower = prev.lower()
            for word_a, word_b in _CONTRADICTION_PAIRS:
                if (word_a in output_lower and word_b in prev_lower) or \
                   (word_b in output_lower and word_a in prev_lower):
                    metrics.contradiction_indicators.append(
                        f"turn {metrics.turn_id}: '{word_a}'/'{word_b}' across turns"
                    )
                    break

class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self._evaluator = AdvancedEvaluator()

    def _build_system(self) -> Any:
        arch = self.config.architecture
        if arch == MemoryArchitecture.TRANSCRIPT_REPLAY:
            return TranscriptReplaySystem(self.config)
        elif arch == MemoryArchitecture.HYBRID_RETRIEVAL:
            return HybridRetrievalSystemBench(self.config)
        elif arch == MemoryArchitecture.RECONSTRUCTED_STATE:
            return ReconstructedStateSystem(self.config)
        raise ValueError(f"Unsupported architecture: {arch}")

    async def run_scenario(self, scenario: Scenario) -> BenchmarkRun:
        run = BenchmarkRun(
            run_id=f"{scenario.scenario_id}_{datetime.now(timezone.utc).isoformat()}",
            scenario_id=scenario.scenario_id,
            config=self.config,
        )

        system = self._build_system()
        system.reset()

        previous_outputs: list[str] = []
        recalled_fact_indices: set[int] = set()
        turn_idx = 0

        for turn in scenario.turns:
            if turn.role != "user":
                continue

            qa_prompt = turn.content
            for qp in scenario.query_pairs:
                if qp.get("turn", 0) <= turn.turn_id:
                    qa_prompt = qp["query"]

            result: SystemAResult = await system.process_turn(
                user_input=qa_prompt,
            )

            if result.metrics:
                result.metrics.output = result.output
                _post_process_turn(
                    result.metrics, scenario,
                    previous_outputs, recalled_fact_indices,
                    turn_idx,
                )
                run.metrics.append(result.metrics)
                m = result.metrics
                run.total_input_tokens += m.input_tokens
                run.total_output_tokens += m.output_tokens
                run.total_latency_ms += m.total_latency_ms
                run.facts_recalled += m.facts_recalled_this_turn
                if m.hallucination_indicators:
                    run.hallucination_events += len(m.hallucination_indicators)
                if m.contradiction_indicators:
                    run.contradictions_detected += len(m.contradiction_indicators)
                previous_outputs.append(result.output)
                turn_idx += 1

        run.decisions_total = len(scenario.expected_decisions)
        run.facts_total = len(scenario.facts)

        if run.decisions_total == 0 and run.facts_total > 0:
            run.decisions_total = run.facts_total
            output_text = " ".join(
                m.output.lower() for m in run.metrics if m.output
            )
            for fact in scenario.facts:
                if fact.content.lower() in output_text:
                    run.decisions_matched += 1

        else:
            output_text = " ".join(
                m.output for m in run.metrics if m.output
            ).lower()
            for expected in scenario.expected_decisions:
                if expected.lower() in output_text:
                    run.decisions_matched += 1

        run.completed_at = datetime.now(timezone.utc)

        return run

    def evaluate(
        self,
        run: BenchmarkRun,
        scenario: Scenario | None = None,
    ) -> list[EvaluationScore]:
        scores: list[EvaluationScore] = []

        if run.decisions_total > 0:
            recall = run.decisions_matched / run.decisions_total
        else:
            recall = 0.0
        scores.append(EvaluationScore(
            name="decision_recall",
            value=round(recall, 4),
            description="Fraction of expected decisions present in output",
        ))

        if run.facts_total > 0:
            fact_recall = run.facts_recalled / run.facts_total
        else:
            fact_recall = 0.0
        scores.append(EvaluationScore(
            name="fact_recall",
            value=round(fact_recall, 4),
            description="Fraction of injected facts recalled across all turns",
        ))

        if run.metrics:
            avg_context = sum(
                m.context_size_tokens for m in run.metrics
            ) / len(run.metrics)
        else:
            avg_context = 0.0
        scores.append(EvaluationScore(
            name="avg_context_size_tokens",
            value=round(avg_context, 2),
            description="Average context size in tokens per turn",
        ))

        if run.metrics:
            latencies = [m.total_latency_ms for m in run.metrics]
            sorted_lat = sorted(latencies)
            n = len(sorted_lat)
            p50 = sorted_lat[max(0, int(n * 0.50) - 1)] if n > 0 else 0.0
            p95 = sorted_lat[max(0, int(n * 0.95) - 1)] if n > 0 else 0.0
            p99 = sorted_lat[max(0, int(n * 0.99) - 1)] if n > 0 else 0.0
        else:
            p50 = p95 = p99 = 0.0
        scores.append(EvaluationScore(
            name="latency_p50_ms", value=round(p50, 2),
            description="Median request-response latency",
        ))
        scores.append(EvaluationScore(
            name="latency_p95_ms", value=round(p95, 2),
            description="95th percentile latency",
        ))
        scores.append(EvaluationScore(
            name="latency_p99_ms", value=round(p99, 2),
            description="99th percentile latency",
        ))

        total_tokens = run.total_input_tokens + run.total_output_tokens
        if total_tokens > 0:
            cer = (run.decisions_matched + run.facts_recalled) / total_tokens
        else:
            cer = 0.0
        scores.append(EvaluationScore(
            name="context_efficiency_ratio",
            value=round(cer, 6),
            description="Useful output per token consumed",
        ))

        scores.append(EvaluationScore(
            name="hallucination_events",
            value=float(run.hallucination_events),
            description="Total hallucinated memory events detected",
        ))
        scores.append(EvaluationScore(
            name="contradiction_events",
            value=float(run.contradictions_detected),
            description="Total contradictions detected",
        ))

        if scenario:
            scores.extend(
                self._evaluator.evaluate_all(run, scenario)
            )

        return scores

    async def compare_architectures(
        self,
        scenario: Scenario,
        configs: list[BenchmarkConfig],
    ) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        for cfg in configs:
            runner = BenchmarkRunner(cfg)
            run = await runner.run_scenario(scenario)
            scores = runner.evaluate(run, scenario)
            results[cfg.architecture.value] = {
                "run": run,
                "scores": {s.name: s.value for s in scores},
                "score_details": scores,
            }
        return results
