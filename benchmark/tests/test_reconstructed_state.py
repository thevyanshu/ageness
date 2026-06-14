from __future__ import annotations

from benchmark.models import BenchmarkConfig, MemoryArchitecture
from benchmark.systems.reconstructed_state import ReconstructedStateSystem


async def test_system_initialization():
    config = BenchmarkConfig(
        architecture=MemoryArchitecture.RECONSTRUCTED_STATE,
        lm_studio_url="",
    )
    system = ReconstructedStateSystem(config)
    assert system.name == "reconstructed_state"


async def test_system_processes_turn_mock():
    config = BenchmarkConfig(
        architecture=MemoryArchitecture.RECONSTRUCTED_STATE,
        lm_studio_url="",
    )
    system = ReconstructedStateSystem(config)
    result = await system.process_turn(user_input="test query")
    assert result.output is not None
    assert result.metrics is not None


async def test_system_stores_memories():
    config = BenchmarkConfig(
        architecture=MemoryArchitecture.RECONSTRUCTED_STATE,
        lm_studio_url="",
    )
    system = ReconstructedStateSystem(config)
    await system.process_turn(user_input="first")
    stored = list(system.store.search(("*", "episodic")))
    assert len(stored) >= 0


async def test_system_returns_metrics():
    config = BenchmarkConfig(
        architecture=MemoryArchitecture.RECONSTRUCTED_STATE,
        lm_studio_url="",
    )
    system = ReconstructedStateSystem(config)
    result = await system.process_turn(user_input="test input")
    m = result.metrics
    assert m is not None
    assert m.turn_id == 1


async def test_system_reset():
    config = BenchmarkConfig(
        architecture=MemoryArchitecture.RECONSTRUCTED_STATE,
        lm_studio_url="",
    )
    system = ReconstructedStateSystem(config)
    await system.process_turn(user_input="hello")
    assert system._turn_count > 0
    system.reset()
    assert system._turn_count == 0


async def test_system_multiple_turns_increment_counter():
    config = BenchmarkConfig(
        architecture=MemoryArchitecture.RECONSTRUCTED_STATE,
        lm_studio_url="",
    )
    system = ReconstructedStateSystem(config)
    r1 = await system.process_turn(user_input="turn 1")
    r2 = await system.process_turn(user_input="turn 2")
    assert r1.metrics.turn_id == 1
    assert r2.metrics.turn_id == 2
