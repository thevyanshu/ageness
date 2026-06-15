from __future__ import annotations

from benchmark.models import (
    BenchmarkRun,
    EvaluationScore,
    Scenario,
)


class HallucinationDetector:
    def detect(
        self,
        run: BenchmarkRun,
        scenario: Scenario,
    ) -> list[EvaluationScore]:
        events: list[str] = []
        all_output = " ".join(
            m.output for m in run.metrics if hasattr(run, "output")
        )

        if not all_output:
            for snapshot in run.metrics:
                events.extend(snapshot.hallucination_indicators)
            return self._scores(events)

        return self._scores(events)

    def _scores(self, events: list[str]) -> list[EvaluationScore]:
        return [
            EvaluationScore(
                name="hallucination_count",
                value=float(len(events)),
                description="Total hallucinated memory events detected",
                metadata={"events": events},
            ),
        ]


class ContradictionDetector:
    def detect(
        self,
        run: BenchmarkRun,
        scenario: Scenario,
    ) -> list[EvaluationScore]:
        events: list[str] = []
        for snapshot in run.metrics:
            events.extend(snapshot.contradiction_indicators)

        return [
            EvaluationScore(
                name="contradiction_count",
                value=float(len(events)),
                description="Total contradiction events detected",
                metadata={"events": events},
            ),
        ]


class TemporalReasoningScorer:
    def score(
        self,
        run: BenchmarkRun,
        scenario: Scenario,
    ) -> list[EvaluationScore]:
        facts = scenario.facts
        if not facts:
            return [
                EvaluationScore(
                    name="temporal_dependency_accuracy",
                    value=1.0,
                    description="No facts to evaluate — default score",
                ),
            ]

        correctly_ordered = 0
        for i, fact in enumerate(facts):
            for j, other in enumerate(facts):
                if i >= j:
                    continue
                if fact.turn_id < other.turn_id:
                    correctly_ordered += 1

        total_pairs = sum(1 for i in range(len(facts)) for j in range(i + 1, len(facts)))
        tda = correctly_ordered / total_pairs if total_pairs > 0 else 1.0

        return [
            EvaluationScore(
                name="temporal_dependency_accuracy",
                value=round(tda, 4),
                description="Correctness of chronological ordering in facts",
            ),
        ]


class CompressionQualityEvaluator:
    def evaluate(
        self,
        run: BenchmarkRun,
        scenario: Scenario,
    ) -> list[EvaluationScore]:
        total_snapshots = len(run.metrics)
        if total_snapshots == 0:
            return [
                EvaluationScore(
                    name="compression_ratio",
                    value=1.0,
                    description="Ratio of context tokens to input tokens",
                ),
                EvaluationScore(
                    name="context_stability",
                    value=1.0,
                    description="How stable context size stays over the session",
                ),
            ]

        avg_context = sum(
            m.context_size_tokens for m in run.metrics
        ) / total_snapshots
        avg_input = sum(
            m.input_tokens for m in run.metrics
        ) / total_snapshots

        compression_ratio = avg_context / avg_input if avg_input > 0 else 1.0

        if total_snapshots > 1:
            sizes = [m.context_size_tokens for m in run.metrics]
            variance = sum((s - avg_context) ** 2 for s in sizes) / total_snapshots
            std_dev = variance ** 0.5
            context_stability = 1.0 / (1.0 + std_dev / max(avg_context, 1))
        else:
            context_stability = 1.0

        return [
            EvaluationScore(
                name="compression_ratio",
                value=round(compression_ratio, 4),
                description="Average context-to-input token ratio (lower = more compressed)",
            ),
            EvaluationScore(
                name="context_stability",
                value=round(context_stability, 4),
                description="Context stability score (1.0 = perfectly stable)",
            ),
        ]


class CheckpointGrowthTracker:
    def evaluate(
        self,
        run: BenchmarkRun,
        scenario: Scenario,
    ) -> list[EvaluationScore]:
        total_snapshots = len(run.metrics)
        if total_snapshots < 2:
            return [
                EvaluationScore(
                    name="checkpoint_growth_rate",
                    value=0.0,
                    description="Average bytes added per turn (checkpoint growth)",
                ),
                EvaluationScore(
                    name="final_checkpoint_size_bytes",
                    value=0.0,
                    description="Total checkpoint size at end of session",
                ),
            ]

        sizes = [m.checkpoint_size_bytes for m in run.metrics]
        final_size = float(sizes[-1])
        growth_rates = [
            sizes[i] - sizes[i - 1]
            for i in range(1, len(sizes))
        ]
        avg_growth = sum(growth_rates) / len(growth_rates)

        return [
            EvaluationScore(
                name="checkpoint_growth_rate",
                value=round(avg_growth, 2),
                description="Average bytes added per turn (checkpoint growth)",
            ),
            EvaluationScore(
                name="final_checkpoint_size_bytes",
                value=round(final_size, 2),
                description="Total checkpoint size at end of session",
            ),
        ]


class AdvancedEvaluator:
    def __init__(self) -> None:
        self.hallucination = HallucinationDetector()
        self.contradiction = ContradictionDetector()
        self.temporal = TemporalReasoningScorer()
        self.compression = CompressionQualityEvaluator()
        self.checkpoint = CheckpointGrowthTracker()

    def evaluate_all(
        self,
        run: BenchmarkRun,
        scenario: Scenario,
    ) -> list[EvaluationScore]:
        scores: list[EvaluationScore] = []

        scores.extend(self.hallucination.detect(run, scenario))
        scores.extend(self.contradiction.detect(run, scenario))
        scores.extend(self.temporal.score(run, scenario))
        scores.extend(self.compression.evaluate(run, scenario))
        scores.extend(self.checkpoint.evaluate(run, scenario))

        return scores
