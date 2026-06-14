from __future__ import annotations

from benchmark.core.runner import BenchmarkRunner
from benchmark.models import BenchmarkConfig, MemoryArchitecture
from benchmark.simulator.generator import ConversationGenerator


async def test_runner_builds_hybrid_retrieval():
    config = BenchmarkConfig(
        architecture=MemoryArchitecture.HYBRID_RETRIEVAL,
        lm_studio_url="",
    )
    runner = BenchmarkRunner(config)
    gen = ConversationGenerator(seed=42)
    scenario = gen.generate_architecture_planning()
    run = await runner.run_scenario(scenario)
    assert run.completed_at is not None
    assert len(run.metrics) > 0


async def test_runner_builds_reconstructed_state():
    config = BenchmarkConfig(
        architecture=MemoryArchitecture.RECONSTRUCTED_STATE,
        lm_studio_url="",
    )
    runner = BenchmarkRunner(config)
    gen = ConversationGenerator(seed=42)
    scenario = gen.generate_architecture_planning()
    run = await runner.run_scenario(scenario)
    assert run.completed_at is not None
    assert len(run.metrics) > 0


async def test_runner_evaluates_hybrid_retrieval():
    config = BenchmarkConfig(
        architecture=MemoryArchitecture.HYBRID_RETRIEVAL,
        lm_studio_url="",
    )
    runner = BenchmarkRunner(config)
    gen = ConversationGenerator(seed=42)
    scenario = gen.generate_architecture_planning()
    run = await runner.run_scenario(scenario)
    scores = runner.evaluate(run, scenario)
    names = [s.name for s in scores]
    assert "decision_recall" in names
    assert "avg_context_size_tokens" in names
    assert "hallucination_count" in names
    assert "context_stability" in names


async def test_compare_three_architectures():
    configs = [
        BenchmarkConfig(
            architecture=MemoryArchitecture.TRANSCRIPT_REPLAY,
            lm_studio_url="",
        ),
        BenchmarkConfig(
            architecture=MemoryArchitecture.HYBRID_RETRIEVAL,
            lm_studio_url="",
        ),
        BenchmarkConfig(
            architecture=MemoryArchitecture.RECONSTRUCTED_STATE,
            lm_studio_url="",
        ),
    ]
    runner = BenchmarkRunner(configs[0])
    gen = ConversationGenerator(seed=42)
    scenario = gen.generate_architecture_planning()
    results = await runner.compare_architectures(scenario, configs)
    assert "transcript_replay" in results
    assert "hybrid_retrieval" in results
    assert "reconstructed_state" in results
    for arch in results:
        assert "scores" in results[arch]
        assert "run" in results[arch]
        assert "score_details" in results[arch]
