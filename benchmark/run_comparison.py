"""Run the comparative benchmark: all 3 architectures against LM Studio."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from benchmark.core.runner import BenchmarkRunner
from benchmark.models import BenchmarkConfig, MemoryArchitecture
from benchmark.simulator.generator import ConversationGenerator
from benchmark.visualization.reporter import generate_report

LM_URL = "http://localhost:1234"


async def main() -> None:
    ts = datetime.now(timezone.utc).isoformat()
    print(f"[{ts}] Generating long session scenario (30 turns)...")
    gen = ConversationGenerator(seed=42)
    scenario = gen.generate_long_session(num_turns=100)

    configs = [
        BenchmarkConfig(
            architecture=MemoryArchitecture.TRANSCRIPT_REPLAY,
            lm_studio_url=LM_URL,
            model="qwen/qwen3.5-9b",
            last_n_messages=20,
            max_output_tokens=64,
            temperature=0.0,
        ),
        BenchmarkConfig(
            architecture=MemoryArchitecture.HYBRID_RETRIEVAL,
            lm_studio_url=LM_URL,
            model="qwen/qwen3.5-9b",
            last_n_messages=20,
            max_output_tokens=64,
            temperature=0.0,
        ),
        BenchmarkConfig(
            architecture=MemoryArchitecture.RECONSTRUCTED_STATE,
            lm_studio_url=LM_URL,
            model="qwen/qwen3.5-9b",
            max_output_tokens=64,
            temperature=0.0,
        ),
    ]

    runner = BenchmarkRunner(configs[0])
    ts = datetime.now(timezone.utc).isoformat()
    print(f"[{ts}] Running 3 architectures on 100-turn session...")
    print("  This will call the LLM ~150 times across 3 architectures")
    bench_start = datetime.now(timezone.utc)

    results = await runner.compare_architectures(scenario, configs)

    elapsed = (datetime.now(timezone.utc) - bench_start).total_seconds()
    ts = datetime.now(timezone.utc).isoformat()
    print(f"[{ts}] Benchmark complete in {elapsed:.0f}s")

    for arch_key, data in results.items():
        run_obj = data["run"]
        scores = data["scores"]
        print(f"\n=== {arch_key} ===")
        print(f"  Turns: {len(run_obj.metrics)}")
        print(f"  Input tokens:  {run_obj.total_input_tokens}")
        print(f"  Output tokens: {run_obj.total_output_tokens}")
        print(f"  Total latency: {run_obj.total_latency_ms:.0f} ms")
        avg_lat = run_obj.total_latency_ms / max(len(run_obj.metrics), 1)
        print(f"  Avg latency:   {avg_lat:.0f} ms/turn")
        for s_name, s_val in sorted(scores.items()):
            print(f"  {s_name}: {s_val}")

    ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = f"benchmark_report_long_session_{ts_str}"
    ts = datetime.now(timezone.utc).isoformat()
    print(f"\n[{ts}] Generating report in {out_dir}/ ...")
    saved = generate_report(results, output_dir=out_dir, scenario_name="long_session_100")
    for name, path in sorted(saved.items()):
        print(f"  {name}: {path}")

    raw_path = Path(out_dir) / "results.json"
    serializable: dict = {}
    for ak, ad in results.items():
        run_obj = ad["run"]
        serializable[ak] = {
            "scores": ad["scores"],
            "total_input_tokens": run_obj.total_input_tokens,
            "total_output_tokens": run_obj.total_output_tokens,
            "total_latency_ms": run_obj.total_latency_ms,
            "num_turns": len(run_obj.metrics),
            "metrics": [
                {
                    "turn_id": m.turn_id,
                    "input_tokens": m.input_tokens,
                    "output_tokens": m.output_tokens,
                    "context_size_tokens": m.context_size_tokens,
                    "total_latency_ms": m.total_latency_ms,
                    "active_memory_count": m.active_memory_count,
                    "checkpoint_size_bytes": m.checkpoint_size_bytes,
                    "memories_retrieved": m.memories_retrieved,
                }
                for m in run_obj.metrics
            ],
        }
    raw_path.write_text(json.dumps(serializable, indent=2, default=str), encoding="utf-8")
    print(f"  results.json: {raw_path}")

    print(f"\nDone! Report → {out_dir}/report.html")


if __name__ == "__main__":
    asyncio.run(main())
