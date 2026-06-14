from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from benchmark.models import (
    BenchmarkConfig,
    BenchmarkRun,
    EvaluationScore,
    MemoryArchitecture,
    Scenario,
)
from benchmark.systems.transcript_replay import TranscriptReplaySystem


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config

    def _build_system(self) -> TranscriptReplaySystem:
        if self.config.architecture == MemoryArchitecture.TRANSCRIPT_REPLAY:
            return TranscriptReplaySystem(self.config)
        raise ValueError(f"Unsupported architecture: {self.config.architecture}")

    async def run_scenario(self, scenario: Scenario) -> BenchmarkRun:
        run = BenchmarkRun(
            run_id=f"{scenario.scenario_id}_{datetime.now(timezone.utc).isoformat()}",
            scenario_id=scenario.scenario_id,
            config=self.config,
        )

        system = self._build_system()
        system.reset()

        qa_index = 0
        query_pairs = sorted(
            scenario.query_pairs, key=lambda qp: qp.get("turn", 999)
        )

        for turn in scenario.turns:
            if turn.role != "user":
                continue

            query_prompt = turn.content
            if qa_index < len(query_pairs) \
                    and query_pairs[qa_index].get("turn", 0) <= turn.turn_id:
                query_prompt = query_pairs[qa_index]["query"]
                qa_index += 1

            result = await system.process_turn(user_input=query_prompt)

            if result.metrics:
                run.metrics.append(result.metrics)
                m = result.metrics
                run.total_input_tokens += m.input_tokens
                run.total_output_tokens += m.output_tokens
                run.total_latency_ms += m.total_latency_ms
                if m.hallucination_indicators:
                    run.hallucination_events += len(m.hallucination_indicators)
                if m.contradiction_indicators:
                    run.contradictions_detected += len(m.contradiction_indicators)

        run.decisions_total = len(scenario.expected_decisions)
        run.facts_total = len(scenario.facts)
        run.completed_at = datetime.now(timezone.utc)

        return run

    def evaluate(self, run: BenchmarkRun) -> list[EvaluationScore]:
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
            p50 = sorted_lat[max(0, int(n * 0.50) - 1)]
            p95 = sorted_lat[max(0, int(n * 0.95) - 1)]
            p99 = sorted_lat[max(0, int(n * 0.99) - 1)]
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
            cer = run.decisions_matched / total_tokens
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
            scores = runner.evaluate(run)
            results[cfg.architecture.value] = {
                "run": run,
                "scores": {s.name: s.value for s in scores},
            }
        return results
