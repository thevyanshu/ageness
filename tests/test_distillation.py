
from ageness.cognition.distillation.pipeline import AsyncDistillationPipeline
from ageness.cognition.salience.engine import MemorySalienceEngine
from ageness.memory.models import DistillationResult, MemoryType


async def test_distill_empty_state_returns_empty_result():
    pipeline = AsyncDistillationPipeline(salience_engine=MemorySalienceEngine())
    result = await pipeline.distill(thread_id="test", state={})
    assert len(result.decisions) == 0
    assert len(result.unresolved_goals) == 0
    assert result.compressed_trace is None


async def test_extract_decisions_from_explicit_list():
    pipeline = AsyncDistillationPipeline(salience_engine=MemorySalienceEngine())
    state = {
        "decisions": [
            {"decision": "use postgres", "rationale": "better consistency"},
            {"decision": "drop redis cache", "rationale": "not needed"},
        ]
    }
    decisions = await pipeline._extract_decisions(state)
    assert len(decisions) == 2
    assert decisions[0]["decision"] == "use postgres"


async def test_extract_decisions_from_metadata():
    pipeline = AsyncDistillationPipeline(salience_engine=MemorySalienceEngine())
    state = {
        "metadata": [
            {"decision": "upgrade python 3.12", "rationale": "performance"},
        ]
    }
    decisions = await pipeline._extract_decisions(state)
    assert len(decisions) == 1
    assert decisions[0]["decision"] == "upgrade python 3.12"


async def test_extract_decisions_from_messages():
    pipeline = AsyncDistillationPipeline(salience_engine=MemorySalienceEngine())
    state = {
        "messages": [
            {"role": "assistant", "content": "I decided to use FastAPI for the API layer"},
            {"role": "user", "content": "what do you think?"},
        ]
    }
    decisions = await pipeline._extract_decisions(state)
    assert len(decisions) == 1
    assert "FastAPI" in decisions[0]["decision"]


async def test_extract_decisions_deduplicates():
    pipeline = AsyncDistillationPipeline(salience_engine=MemorySalienceEngine())
    state = {
        "decisions": [
            {"decision": "use postgres", "rationale": "good"},
            {"decision": "use postgres", "rationale": "duplicate"},
        ]
    }
    decisions = await pipeline._extract_decisions(state)
    assert len(decisions) == 1


async def test_identify_unresolved_goals_from_strings():
    pipeline = AsyncDistillationPipeline(salience_engine=MemorySalienceEngine())
    state = {"goals": ["fix login", "deploy to prod", "write docs"]}
    goals = await pipeline._identify_unresolved_goals(state)
    assert len(goals) == 3


async def test_identify_unresolved_goals_filters_completed():
    pipeline = AsyncDistillationPipeline(salience_engine=MemorySalienceEngine())
    state = {
        "goals": [
            {"goal": "fix login", "status": "active"},
            {"goal": "deploy", "status": "completed"},
            {"goal": "write docs", "status": "resolved"},
            {"goal": "refactor", "status": "cancelled"},
        ]
    }
    goals = await pipeline._identify_unresolved_goals(state)
    assert len(goals) == 1
    assert goals[0]["goal"] == "fix login"


async def test_compress_trace_with_input_output():
    pipeline = AsyncDistillationPipeline(salience_engine=MemorySalienceEngine())
    state = {"input": "hello", "output": "hi there"}
    trace = await pipeline._compress_trace(state)
    assert trace is not None
    assert "Input: hello" in trace
    assert "Output: hi there" in trace


async def test_compress_trace_with_messages():
    pipeline = AsyncDistillationPipeline(salience_engine=MemorySalienceEngine())
    state = {
        "messages": [
            {"role": "user", "content": "q1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a1"},
        ]
    }
    trace = await pipeline._compress_trace(state)
    assert trace is not None
    assert "2 user" in trace
    assert "1 assistant" in trace


async def test_compress_trace_empty_returns_none():
    pipeline = AsyncDistillationPipeline(salience_engine=MemorySalienceEngine())
    trace = await pipeline._compress_trace({})
    assert trace is None


async def test_build_memories_creates_correct_types():
    pipeline = AsyncDistillationPipeline(salience_engine=MemorySalienceEngine())
    result = DistillationResult(
        decisions=[{"decision": "use python", "rationale": "ecosystem"}],
        unresolved_goals=[{"goal": "finish project", "status": "active"}],
        compressed_trace="Input: test",
    )
    memories = await pipeline._build_memories(result)
    types = [m.memory_type for m in memories]
    assert MemoryType.DECISION in types
    assert MemoryType.GOAL in types
    assert MemoryType.EPISODIC in types


async def test_build_memories_empty_result():
    pipeline = AsyncDistillationPipeline(salience_engine=MemorySalienceEngine())
    memories = await pipeline._build_memories(DistillationResult())
    assert len(memories) == 0


async def test_distill_full_pipeline():
    pipeline = AsyncDistillationPipeline(salience_engine=MemorySalienceEngine())
    state = {
        "input": "help me plan the project",
        "output": "here is the plan",
        "decisions": [
            {"decision": "use python", "rationale": "team expertise"},
        ],
        "goals": ["build feature x", "write tests"],
        "messages": [
            {"role": "user", "content": "help"},
            {"role": "assistant", "content": "I decided to start with planning"},
        ],
    }
    result = await pipeline.distill(thread_id="t1", state=state)
    assert len(result.decisions) >= 1
    assert len(result.unresolved_goals) == 2
    assert result.compressed_trace is not None
    assert len(result.memory_items) >= 2
