from __future__ import annotations

from benchmark.metrics.evaluator import (
    CompressionQualityEvaluator,
    ContradictionDetector,
    HallucinationDetector,
    TemporalReasoningScorer,
)
from benchmark.models import (
    BenchmarkConfig,
    BenchmarkRun,
    InjectedFact,
    MemoryArchitecture,
    MetricsSnapshot,
    Scenario,
)


def _make_run(metrics: list[MetricsSnapshot] | None = None) -> BenchmarkRun:
    cfg = BenchmarkConfig(architecture=MemoryArchitecture.TRANSCRIPT_REPLAY)
    run = BenchmarkRun(run_id="r1", scenario_id="s1", config=cfg)
    if metrics:
        run.metrics = metrics
    return run


def test_hallucination_detector_empty():
    run = _make_run()
    detector = HallucinationDetector()
    scores = detector.detect(run, Scenario(
        scenario_id="s1", title="t", description="d", turns=[],
    ))
    names = [s.name for s in scores]
    assert "hallucination_count" in names


def test_contradiction_detector():
    m1 = MetricsSnapshot(turn_id=1, contradiction_indicators=["conflict A"])
    run = _make_run([m1])
    detector = ContradictionDetector()
    scores = detector.detect(run, Scenario(
        scenario_id="s1", title="t", description="d", turns=[],
    ))
    assert scores[0].value == 1


def test_temporal_reasoning_accuracy_perfect():
    facts = [
        InjectedFact(fact_id="f1", turn_id=1, content="first", category="decision"),
        InjectedFact(fact_id="f2", turn_id=2, content="second", category="decision"),
        InjectedFact(fact_id="f3", turn_id=3, content="third", category="decision"),
    ]
    scenario = Scenario(
        scenario_id="s1", title="t", description="d", turns=[], facts=facts,
    )
    run = _make_run()
    scorer = TemporalReasoningScorer()
    scores = scorer.score(run, scenario)
    tda = [s for s in scores if s.name == "temporal_dependency_accuracy"][0]
    assert tda.value == 1.0


def test_compression_quality_evaluator():
    m1 = MetricsSnapshot(turn_id=1, context_size_tokens=100, input_tokens=200)
    m2 = MetricsSnapshot(turn_id=2, context_size_tokens=120, input_tokens=250)
    run = _make_run([m1, m2])
    evaluator = CompressionQualityEvaluator()
    scores = evaluator.evaluate(run, Scenario(
        scenario_id="s1", title="t", description="d", turns=[],
    ))
    names = [s.name for s in scores]
    assert "compression_ratio" in names
    assert "context_stability" in names


def test_compression_ratio_calculation():
    m1 = MetricsSnapshot(turn_id=1, context_size_tokens=50, input_tokens=200)
    m2 = MetricsSnapshot(turn_id=2, context_size_tokens=50, input_tokens=200)
    run = _make_run([m1, m2])
    evaluator = CompressionQualityEvaluator()
    scores = evaluator.evaluate(run, Scenario(
        scenario_id="s1", title="t", description="d", turns=[],
    ))
    ratio = [s for s in scores if s.name == "compression_ratio"][0]
    assert ratio.value == 0.25


def test_advanced_evaluator_runs_all():
    from benchmark.metrics.evaluator import AdvancedEvaluator
    evaluator = AdvancedEvaluator()
    run = _make_run()
    scenario = Scenario(
        scenario_id="s1", title="t", description="d", turns=[],
    )
    scores = evaluator.evaluate_all(run, scenario)
    names = [s.name for s in scores]
    assert "hallucination_count" in names
    assert "contradiction_count" in names
    assert "temporal_dependency_accuracy" in names
    assert "compression_ratio" in names
    assert "context_stability" in names
