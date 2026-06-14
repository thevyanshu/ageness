from langgraph.store.memory import InMemoryStore

from ageness.cognition.active_context.builder import ActiveContextBuilder
from ageness.cognition.retrieval.hybrid import HybridRetrievalSystem
from ageness.memory.models import MemoryItem, SalienceScore
from ageness.memory.models import MemoryType as MT


def _make_item(
    key: str,
    memory_type: str,
    content: str,
    composite: float = 0.5,
) -> MemoryItem:
    return MemoryItem(
        key=key,
        namespace=("*", memory_type),
        value={
            "content": content,
            "memory_type": memory_type,
        },
        salience=SalienceScore(
            relevance=composite,
            importance=composite,
            novelty=composite,
            composite=composite,
        ),
    )


def test_classify_items_by_memory_type():
    builder = ActiveContextBuilder(retrieval=HybridRetrievalSystem(InMemoryStore()))
    items = [
        _make_item("g1", MT.GOAL.value, "goal task"),
        _make_item("d1", MT.DECISION.value, "decision made"),
        _make_item("e1", MT.EPISODIC.value, "something happened"),
        _make_item("f1", "semantic", "general fact"),
    ]
    classified = builder._classify_items(items)
    assert len(classified["goals"]) == 1
    assert len(classified["decisions"]) == 1
    assert len(classified["episodes"]) == 1
    assert len(classified["facts"]) == 1


def test_classify_unknown_type_goes_to_facts():
    builder = ActiveContextBuilder(retrieval=HybridRetrievalSystem(InMemoryStore()))
    items = [_make_item("x1", "unknown_type", "something")]
    classified = builder._classify_items(items)
    assert len(classified["facts"]) == 1


def test_sort_by_priority_order():
    builder = ActiveContextBuilder(retrieval=HybridRetrievalSystem(InMemoryStore()))
    items = [
        _make_item("e1", MT.EPISODIC.value, "episodic"),
        _make_item("g1", MT.GOAL.value, "goal"),
        _make_item("f1", "semantic", "fact"),
        _make_item("d1", MT.DECISION.value, "decision"),
    ]
    classified = builder._classify_items(items)
    sorted_items = builder._sort_by_priority(classified)
    keys = [i.key for i in sorted_items]
    assert keys.index("g1") < keys.index("d1")
    assert keys.index("d1") < keys.index("e1")
    assert keys.index("e1") < keys.index("f1")


def test_sort_within_category_by_salience():
    builder = ActiveContextBuilder(retrieval=HybridRetrievalSystem(InMemoryStore()))
    items = [
        _make_item("g2", MT.GOAL.value, "low goal", composite=0.3),
        _make_item("g1", MT.GOAL.value, "high goal", composite=0.9),
    ]
    classified = builder._classify_items(items)
    sorted_items = builder._sort_by_priority(classified)
    assert sorted_items[0].key == "g1"
    assert sorted_items[1].key == "g2"


def test_assemble_window_respects_token_budget():
    builder = ActiveContextBuilder(
        retrieval=HybridRetrievalSystem(InMemoryStore()), max_tokens=10
    )
    items = [
        _make_item("g1", MT.GOAL.value, "a" * 100),
        _make_item("g2", MT.GOAL.value, "b"),
    ]
    window = builder._assemble_window(items)
    assert window.total_tokens <= 10
    assert len(window.goals) <= 1


def test_assemble_fills_categories_correctly():
    builder = ActiveContextBuilder(retrieval=HybridRetrievalSystem(InMemoryStore()))
    items = [
        _make_item("g1", MT.GOAL.value, "goal"),
        _make_item("d1", MT.DECISION.value, "decision"),
        _make_item("e1", MT.EPISODIC.value, "episode"),
        _make_item("f1", "semantic", "fact"),
    ]
    window = builder._assemble_window(items)
    assert len(window.goals) == 1
    assert len(window.decisions) == 1
    assert len(window.episodes) == 1
    assert len(window.facts) == 1


def test_empty_items_returns_empty_window():
    builder = ActiveContextBuilder(retrieval=HybridRetrievalSystem(InMemoryStore()))
    window = builder._assemble_window([])
    assert window.total_tokens == 0
    assert len(window.goals) == 0
    assert len(window.decisions) == 0
    assert len(window.episodes) == 0
    assert len(window.facts) == 0


def test_estimate_tokens_rough_4_chars_per_token():
    builder = ActiveContextBuilder(retrieval=HybridRetrievalSystem(InMemoryStore()))
    item = _make_item("t1", MT.GOAL.value, "a" * 40)
    tokens = builder._estimate_tokens(item)
    assert tokens == 10


def test_estimate_tokens_minimum_1():
    builder = ActiveContextBuilder(retrieval=HybridRetrievalSystem(InMemoryStore()))
    item = _make_item("t1", MT.GOAL.value, "")
    tokens = builder._estimate_tokens(item)
    assert tokens == 1


def test_budget_drops_lowest_priority_first():
    builder = ActiveContextBuilder(
        retrieval=HybridRetrievalSystem(InMemoryStore()), max_tokens=3
    )
    items = [
        _make_item("g1", MT.GOAL.value, "important goal", composite=0.9),
        _make_item("f1", "semantic", "low priority fact that takes many tokens", composite=0.1),
    ]
    window = builder._assemble_window(items)
    assert "g1" in [i.key for i in window.goals]
