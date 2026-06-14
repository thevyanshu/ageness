from __future__ import annotations

from pathlib import Path

from benchmark.core.runner import BenchmarkRunner
from benchmark.models import BenchmarkConfig, MemoryArchitecture
from benchmark.simulator.generator import ConversationGenerator
from benchmark.visualization.reporter import (
    generate_report,
    plot_context_growth,
    plot_hallucination_rate,
    plot_latency_stability,
    plot_memory_growth,
    plot_recall_accuracy,
    plot_token_cost,
)


async def test_all_plot_functions_return_figures():
    config = BenchmarkConfig(architecture=MemoryArchitecture.TRANSCRIPT_REPLAY)
    runner = BenchmarkRunner(config)
    gen = ConversationGenerator(seed=42)
    scenario = gen.generate_architecture_planning()

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
    results = await runner.compare_architectures(scenario, configs)

    assert len(results) == 3

    for fn in [
        plot_context_growth,
        plot_latency_stability,
        plot_token_cost,
        plot_recall_accuracy,
        plot_hallucination_rate,
        plot_memory_growth,
    ]:
        fig = fn(results)
        assert fig is not None, f"{fn.__name__} returned None"


async def test_generate_report_creates_files(tmp_path):
    config = BenchmarkConfig(architecture=MemoryArchitecture.TRANSCRIPT_REPLAY)
    runner = BenchmarkRunner(config)
    gen = ConversationGenerator(seed=42)
    scenario = gen.generate_architecture_planning()

    configs = [
        BenchmarkConfig(
            architecture=MemoryArchitecture.TRANSCRIPT_REPLAY,
            lm_studio_url="",
        ),
        BenchmarkConfig(
            architecture=MemoryArchitecture.RECONSTRUCTED_STATE,
            lm_studio_url="",
        ),
    ]
    results = await runner.compare_architectures(scenario, configs)

    out_dir = str(tmp_path / "report")
    saved = generate_report(results, output_dir=out_dir, scenario_name="test")
    assert "report.html" in saved
    assert Path(saved["report.html"]).exists()
    assert "context_growth.html" in saved
    assert "latency_stability.html" in saved
    assert Path(saved["context_growth.html"]).exists()


async def test_generate_report_handles_single_architecture():
    config = BenchmarkConfig(architecture=MemoryArchitecture.TRANSCRIPT_REPLAY)
    runner = BenchmarkRunner(config)
    gen = ConversationGenerator(seed=42)
    scenario = gen.generate_architecture_planning()

    configs = [BenchmarkConfig(
        architecture=MemoryArchitecture.TRANSCRIPT_REPLAY,
        lm_studio_url="",
    )]
    results = await runner.compare_architectures(scenario, configs)

    import tempfile
    with tempfile.TemporaryDirectory() as out_dir:
        saved = generate_report(results, output_dir=out_dir, scenario_name="single")
        assert "report.html" in saved
        assert Path(saved["report.html"]).exists()
