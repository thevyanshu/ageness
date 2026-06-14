from langgraph.store.memory import InMemoryStore

from ageness.cognition.reconstruction.reconstructor import CognitiveStateReconstruction
from ageness.cognition.retrieval.hybrid import HybridRetrievalSystem
from ageness.memory.models import CognitiveState


def _populate_store(store: InMemoryStore) -> None:
    store.put(
        ("*", "goal"),
        "goal-1",
        {
            "content": "fix login bug",
            "memory_type": "goal",
            "status": "active",
            "salience": {"composite": 0.9},
        },
    )
    store.put(
        ("*", "goal"),
        "goal-2",
        {
            "content": "deploy to production",
            "memory_type": "goal",
            "status": "active",
            "salience": {"composite": 0.7},
        },
    )
    store.put(
        ("*", "goal"),
        "goal-3",
        {
            "content": "write tests",
            "memory_type": "goal",
            "status": "completed",
            "salience": {"composite": 0.5},
        },
    )
    store.put(
        ("*", "decision"),
        "dec-1",
        {
            "content": "use PostgreSQL for data layer",
            "memory_type": "decision",
            "metadata": {"rationale": "better consistency"},
            "salience": {"composite": 0.8},
        },
    )
    store.put(
        ("*", "decision"),
        "dec-2",
        {
            "content": "adopt FastAPI for API",
            "memory_type": "decision",
            "metadata": {"rationale": "async support"},
            "salience": {"composite": 0.6},
        },
    )
    store.put(
        ("*", "episodic"),
        "ep-1",
        {
            "content": "deployed staging environment",
            "memory_type": "episodic",
            "status": "completed",
            "salience": {"composite": 0.4},
        },
    )
    store.put(
        ("*", "episodic"),
        "ep-2",
        {
            "content": "need to fix memory leak",
            "memory_type": "episodic",
            "status": "active",
            "salience": {"composite": 0.3},
        },
    )


async def test_reconstruct_returns_cognitive_state():
    store = InMemoryStore()
    _populate_store(store)
    recon = CognitiveStateReconstruction(HybridRetrievalSystem(store))
    state = await recon.reconstruct(thread_id="t1", query="fix bugs")
    assert isinstance(state, CognitiveState)


async def test_reconstruct_finds_active_goals():
    store = InMemoryStore()
    _populate_store(store)
    recon = CognitiveStateReconstruction(HybridRetrievalSystem(store))
    state = await recon.reconstruct(thread_id="t1", query="fix bugs")
    goals = state.active_goals
    assert len(goals) >= 1
    assert any("login" in g["goal"] for g in goals)


async def test_reconstruct_filters_completed_goals():
    store = InMemoryStore()
    _populate_store(store)
    recon = CognitiveStateReconstruction(HybridRetrievalSystem(store))
    state = await recon.reconstruct(thread_id="t1", query="")
    goals = state.active_goals
    for g in goals:
        assert g["status"] == "active"


async def test_reconstruct_surfaces_decisions():
    store = InMemoryStore()
    _populate_store(store)
    recon = CognitiveStateReconstruction(HybridRetrievalSystem(store))
    state = await recon.reconstruct(thread_id="t1", query="")
    decisions = state.relevant_decisions
    assert len(decisions) >= 1
    assert any("PostgreSQL" in d["decision"] for d in decisions)
    assert any("FastAPI" in d["decision"] for d in decisions)


async def test_reconstruct_decisions_include_rationale():
    store = InMemoryStore()
    _populate_store(store)
    recon = CognitiveStateReconstruction(HybridRetrievalSystem(store))
    state = await recon.reconstruct(thread_id="t1", query="")
    for d in state.relevant_decisions:
        if "PostgreSQL" in d["decision"]:
            assert d["rationale"] == "better consistency"


async def test_reconstruct_finds_unresolved_tasks():
    store = InMemoryStore()
    _populate_store(store)
    recon = CognitiveStateReconstruction(HybridRetrievalSystem(store))
    state = await recon.reconstruct(thread_id="t1", query="")
    assert any("memory leak" in t["task"] for t in state.unresolved_tasks)


async def test_reconstruct_excludes_completed_tasks():
    store = InMemoryStore()
    _populate_store(store)
    recon = CognitiveStateReconstruction(HybridRetrievalSystem(store))
    state = await recon.reconstruct(thread_id="t1", query="")
    for t in state.unresolved_tasks:
        assert t["status"] != "completed"


async def test_reconstruct_resolves_dependencies():
    store = InMemoryStore()
    _populate_store(store)
    recon = CognitiveStateReconstruction(HybridRetrievalSystem(store))
    state = await recon.reconstruct(thread_id="t1", query="")
    deps = state.contextual_dependencies
    assert "related_goal_keys" in deps
    assert "related_decision_keys" in deps
    assert len(deps["related_goal_keys"]) >= 1
    assert len(deps["related_decision_keys"]) >= 1


async def test_reconstruct_confidence_empty():
    recon = CognitiveStateReconstruction(HybridRetrievalSystem(InMemoryStore()))
    state = await recon.reconstruct(thread_id="t1", query="")
    assert state.reconstruction_confidence == 0.0


async def test_reconstruct_confidence_non_empty():
    store = InMemoryStore()
    _populate_store(store)
    recon = CognitiveStateReconstruction(HybridRetrievalSystem(store))
    state = await recon.reconstruct(thread_id="t1", query="")
    assert state.reconstruction_confidence > 0.0


async def test_reconstruct_goals_sorted_by_salience():
    store = InMemoryStore()
    _populate_store(store)
    recon = CognitiveStateReconstruction(HybridRetrievalSystem(store))
    state = await recon.reconstruct(thread_id="t1", query="")
    goals = state.active_goals
    for i in range(len(goals) - 1):
        assert goals[i]["salience"] >= goals[i + 1]["salience"]
