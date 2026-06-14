from __future__ import annotations

from benchmark.core.runner import BenchmarkRunner
from benchmark.models import BenchmarkConfig, MemoryArchitecture
from benchmark.simulator.generator import ConversationGenerator


async def test_runner_initialization():
    config = BenchmarkConfig(
        architecture=MemoryArchitecture.TRANSCRIPT_REPLAY,
        lm_studio_url="",
    )
    runner = BenchmarkRunner(config)
    assert runner.config.architecture == MemoryArchitecture.TRANSCRIPT_REPLAY


async def test_runner_runs_short_scenario():
    config = BenchmarkConfig(
        architecture=MemoryArchitecture.TRANSCRIPT_REPLAY,
        lm_studio_url="",
    )
    runner = BenchmarkRunner(config)
    gen = ConversationGenerator(seed=42)
    scenario = gen.generate_architecture_planning()
    run = await runner.run_scenario(scenario)
    assert run.run_id is not None
    assert run.scenario_id == "arch_planning_001"
    assert run.completed_at is not None
    assert len(run.metrics) > 0


async def test_runner_collects_metrics():
    config = BenchmarkConfig(
        architecture=MemoryArchitecture.TRANSCRIPT_REPLAY,
        lm_studio_url="",
    )
    runner = BenchmarkRunner(config)
    gen = ConversationGenerator(seed=42)
    scenario = gen.generate_architecture_planning()
    run = await runner.run_scenario(scenario)
    assert run.total_input_tokens > 0
    assert run.total_output_tokens > 0
    assert run.total_latency_ms >= 0


async def test_runner_evaluation_scores():
    config = BenchmarkConfig(
        architecture=MemoryArchitecture.TRANSCRIPT_REPLAY,
        lm_studio_url="",
    )
    runner = BenchmarkRunner(config)
    gen = ConversationGenerator(seed=42)
    scenario = gen.generate_architecture_planning()
    run = await runner.run_scenario(scenario)
    scores = runner.evaluate(run)
    score_names = [s.name for s in scores]
    assert "decision_recall" in score_names
    assert "avg_context_size_tokens" in score_names
    assert "latency_p50_ms" in score_names
    assert "latency_p95_ms" in score_names
    assert "context_efficiency_ratio" in score_names


async def test_runner_comparison():
    config_a = BenchmarkConfig(
        architecture=MemoryArchitecture.TRANSCRIPT_REPLAY,
        truncation="last_n",
        lm_studio_url="",
    )
    runner = BenchmarkRunner(config_a)
    gen = ConversationGenerator(seed=42)
    scenario = gen.generate_architecture_planning()
    results = await runner.compare_architectures(scenario, [config_a])
    assert "transcript_replay" in results
    assert "scores" in results["transcript_replay"]
    assert "run" in results["transcript_replay"]
