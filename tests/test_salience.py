from datetime import datetime, timedelta, timezone

from ageness.cognition.salience.engine import MemorySalienceEngine
from ageness.memory.models import MemoryItem, MemoryType, RawMemory, SalienceScore


def test_score_goal_memory_high_relevance():
    engine = MemorySalienceEngine()
    memory = RawMemory(content="complete project report", memory_type=MemoryType.GOAL)
    score = engine.score_sync(memory)
    assert score.relevance >= 0.8
    assert score.importance >= 0.5
    assert score.novelty == 1.0


def test_score_decision_memory_high_importance():
    engine = MemorySalienceEngine()
    memory = RawMemory(content="chose Python over Java", memory_type=MemoryType.DECISION)
    score = engine.score_sync(memory)
    assert score.importance >= 0.5


def test_score_semantic_lowest_relevance():
    engine = MemorySalienceEngine()
    memory = RawMemory(content="general fact", memory_type=MemoryType.SEMANTIC)
    score = engine.score_sync(memory)
    assert score.relevance <= 0.6


def test_novelty_drops_with_similar_existing():
    engine = MemorySalienceEngine()
    memory = RawMemory(content="deploy to production", memory_type=MemoryType.EPISODIC)
    existing = [
        MemoryItem(
            key="prev",
            namespace=("user", "episodic"),
            value={"content": "deploy to production server"},
        )
    ]
    score = engine.score_sync(memory, existing_items=existing)
    assert score.novelty < 0.5


def test_active_metadata_boosts_relevance():
    engine = MemorySalienceEngine()
    active = RawMemory(content="fix login bug", memory_type=MemoryType.GOAL, metadata={"active": True})
    inactive = RawMemory(content="fix login bug", memory_type=MemoryType.GOAL)
    active_score = engine.score_sync(active)
    inactive_score = engine.score_sync(inactive)
    assert active_score.relevance > inactive_score.relevance


def test_error_metadata_boosts_importance():
    engine = MemorySalienceEngine()
    error = RawMemory(content="payment crash", memory_type=MemoryType.EPISODIC, metadata={"error": True})
    normal = RawMemory(content="payment ok", memory_type=MemoryType.EPISODIC)
    error_score = engine.score_sync(error)
    normal_score = engine.score_sync(normal)
    assert error_score.importance > normal_score.importance


def test_user_priority_boost():
    engine = MemorySalienceEngine()
    high = RawMemory(content="urgent task", memory_type=MemoryType.GOAL, metadata={"user_priority": True})
    normal = RawMemory(content="normal task", memory_type=MemoryType.GOAL)
    high_score = engine.score_sync(high)
    normal_score = engine.score_sync(normal)
    assert high_score.importance >= normal_score.importance


def test_freshness_decays_over_time():
    engine = MemorySalienceEngine()
    old = RawMemory(
        content="stale memory",
        memory_type=MemoryType.EPISODIC,
        timestamp=datetime.now(timezone.utc) - timedelta(days=30),
    )
    fresh = RawMemory(
        content="fresh memory",
        memory_type=MemoryType.EPISODIC,
    )
    old_score = engine.score_sync(old)
    fresh_score = engine.score_sync(fresh)
    assert old_score.relevance < fresh_score.relevance


def test_weights_normalized():
    engine = MemorySalienceEngine(relevance_weight=1, importance_weight=1, novelty_weight=1)
    assert abs(engine.relevance_weight + engine.importance_weight + engine.novelty_weight - 1.0) < 1e-6


def test_score_returns_bounded_values():
    engine = MemorySalienceEngine()
    memory = RawMemory(content="test", memory_type=MemoryType.GOAL, metadata={"active": True, "error": True})
    score = engine.score_sync(memory)
    assert 0.0 <= score.relevance <= 1.0
    assert 0.0 <= score.importance <= 1.0
    assert 0.0 <= score.novelty <= 1.0
    assert 0.0 <= score.composite <= 1.0
