from __future__ import annotations

from benchmark.models import (
    BenchmarkConfig,
    ContextTruncation,
    MemoryArchitecture,
)
from benchmark.systems.transcript_replay import TranscriptReplaySystem


async def test_system_initialization():
    config = BenchmarkConfig(architecture=MemoryArchitecture.TRANSCRIPT_REPLAY)
    system = TranscriptReplaySystem(config)
    assert system.name == "transcript_replay"
    assert system.get_history_length() == 0


async def test_system_processes_turn():
    config = BenchmarkConfig(
        architecture=MemoryArchitecture.TRANSCRIPT_REPLAY,
        lm_studio_url="",
    )
    system = TranscriptReplaySystem(config)
    result = await system.process_turn(user_input="hello")
    assert result.output is not None
    assert "hello" in result.output
    assert result.metrics is not None
    assert result.metrics.turn_id == 1


async def test_system_tracks_history():
    config = BenchmarkConfig(
        architecture=MemoryArchitecture.TRANSCRIPT_REPLAY,
        lm_studio_url="",
    )
    system = TranscriptReplaySystem(config)
    await system.process_turn(user_input="first message")
    await system.process_turn(user_input="second message")
    assert system.get_history_length() == 4


async def test_system_builds_context_with_last_n():
    config = BenchmarkConfig(
        architecture=MemoryArchitecture.TRANSCRIPT_REPLAY,
        truncation=ContextTruncation.LAST_N,
        last_n_messages=2,
        lm_studio_url="",
    )
    system = TranscriptReplaySystem(config)
    for i in range(10):
        await system.process_turn(user_input=f"message {i}")
    assert system.get_history_length() == 20


async def test_system_context_token_limit():
    config = BenchmarkConfig(
        architecture=MemoryArchitecture.TRANSCRIPT_REPLAY,
        truncation=ContextTruncation.TOKEN_BUDGET,
        token_limit=50,
        lm_studio_url="",
    )
    system = TranscriptReplaySystem(config)
    context, tokens = system._build_context()
    assert tokens <= 50


async def test_system_reset():
    config = BenchmarkConfig(
        architecture=MemoryArchitecture.TRANSCRIPT_REPLAY,
        lm_studio_url="",
    )
    system = TranscriptReplaySystem(config)
    await system.process_turn(user_input="hello")
    assert system.get_history_length() > 0
    system.reset()
    assert system.get_history_length() == 0


async def test_system_rolling_summary_truncation():
    config = BenchmarkConfig(
        architecture=MemoryArchitecture.TRANSCRIPT_REPLAY,
        truncation=ContextTruncation.ROLLING_SUMMARY,
        last_n_messages=2,
        lm_studio_url="",
    )
    system = TranscriptReplaySystem(config)
    await system.process_turn(user_input="long message " * 50)
    summary = system._generate_summary()
    assert len(summary) > 0
