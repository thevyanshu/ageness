from __future__ import annotations

from benchmark.models import BenchmarkConfig, MemoryArchitecture
from benchmark.systems.hybrid_retrieval import HybridRetrievalSystemBench


async def test_system_initialization():
    config = BenchmarkConfig(
        architecture=MemoryArchitecture.HYBRID_RETRIEVAL,
        lm_studio_url="",
    )
    system = HybridRetrievalSystemBench(config)
    assert system.name == "hybrid_retrieval"


async def test_system_processes_turn_mock():
    config = BenchmarkConfig(
        architecture=MemoryArchitecture.HYBRID_RETRIEVAL,
        lm_studio_url="",
    )
    system = HybridRetrievalSystemBench(config)
    result = await system.process_turn(user_input="hello")
    assert result.output is not None
    assert result.metrics is not None


async def test_system_tracks_vector_store():
    config = BenchmarkConfig(
        architecture=MemoryArchitecture.HYBRID_RETRIEVAL,
        lm_studio_url="",
    )
    system = HybridRetrievalSystemBench(config)
    await system.process_turn(user_input="first message")
    await system.process_turn(user_input="second message")
    assert len(system._vector_store) == 2


async def test_system_returns_metrics():
    config = BenchmarkConfig(
        architecture=MemoryArchitecture.HYBRID_RETRIEVAL,
        lm_studio_url="",
    )
    system = HybridRetrievalSystemBench(config)
    result = await system.process_turn(user_input="test input")
    m = result.metrics
    assert m is not None
    assert m.turn_id == 1
    assert m.input_tokens > 0


async def test_system_empty_embedding_returns_empty_vector():
    config = BenchmarkConfig(
        architecture=MemoryArchitecture.HYBRID_RETRIEVAL,
        lm_studio_url="",
    )
    system = HybridRetrievalSystemBench(config)
    chunks = system._vector_retrieve("test query")
    assert chunks == []


async def test_system_reset():
    config = BenchmarkConfig(
        architecture=MemoryArchitecture.HYBRID_RETRIEVAL,
        lm_studio_url="",
    )
    system = HybridRetrievalSystemBench(config)
    await system.process_turn(user_input="hello")
    assert len(system._history) > 0
    system.reset()
    assert len(system._history) == 0
    assert len(system._vector_store) == 0
