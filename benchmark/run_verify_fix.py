"""Quick verification: fact_recall <= 1.0, hallucinations sane."""

from __future__ import annotations

import asyncio

from benchmark.core.runner import BenchmarkRunner
from benchmark.models import BenchmarkConfig, MemoryArchitecture
from benchmark.simulator.generator import ConversationGenerator

LM = "http://localhost:1234"


async def main() -> None:
    sc = ConversationGenerator(seed=42).generate_long_session(num_turns=30)

    cfg_a = BenchmarkConfig(
        architecture=MemoryArchitecture.TRANSCRIPT_REPLAY,
        lm_studio_url=LM, max_output_tokens=64, temperature=0.0,
    )
    cfg_b = BenchmarkConfig(
        architecture=MemoryArchitecture.RECONSTRUCTED_STATE,
        lm_studio_url=LM, max_output_tokens=64, temperature=0.0,
    )

    for label, cfg in [("transcript_replay", cfg_a), ("reconstructed_state", cfg_b)]:
        runner = BenchmarkRunner(cfg)
        run = await runner.run_scenario(sc)
        scores = runner.evaluate(run, sc)

        print(f"\n=== {label} ===")
        for s in scores:
            print(f"  {s.name}: {s.value}")
        print(f"  recall_raw: {run.facts_recalled}/{run.facts_total}")
        h = sum(len(m.hallucination_indicators) for m in run.metrics)
        c = sum(len(m.contradiction_indicators) for m in run.metrics)
        print(f"  hallucination_indicators: {h}")
        print(f"  contradiction_indicators: {c}")

        # Assertions that prove the fixes work
        msg = f"fact_recall exceeded {run.facts_recalled} > {run.facts_total}"
        assert run.facts_recalled <= run.facts_total, msg
        assert h <= len(run.metrics), f"too many hallucinations {h} > {len(run.metrics)}"
        print("  ASSERTIONS PASSED")

    print("\nALL OK")


asyncio.run(main())
