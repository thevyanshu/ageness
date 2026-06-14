from datetime import datetime, timedelta, timezone

from langgraph.store.memory import InMemoryStore

from ageness.cognition.retrieval.hybrid import HybridRetrievalSystem, RetrievalStrategy
from ageness.memory.models import MemoryType, RetrievalQuery


def _populate_store(store: InMemoryStore) -> None:
    store.put(
        ("*", "episodic"),
        "mem-1",
        {
            "content": "deployed application to production server",
            "memory_type": "episodic",
            "salience": {"composite": 0.8, "relevance": 0.7, "importance": 0.6, "novelty": 0.5},
            "created_at": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
            "last_accessed": datetime.now(timezone.utc).isoformat(),
        },
    )
    store.put(
        ("*", "goal"),
        "mem-2",
        {
            "content": "fix login bug before next release",
            "memory_type": "goal",
            "salience": {"composite": 0.95, "relevance": 0.9, "importance": 0.8, "novelty": 0.3},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_accessed": datetime.now(timezone.utc).isoformat(),
        },
    )
    store.put(
        ("*", "semantic"),
        "mem-3",
        {
            "content": "Python is a programming language",
            "memory_type": "semantic",
            "salience": {"composite": 0.3, "relevance": 0.2, "importance": 0.2, "novelty": 0.7},
            "created_at": (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
            "last_accessed": (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
        },
    )
    store.put(
        ("*", "decision"),
        "mem-4",
        {
            "content": "chose PostgreSQL over MySQL for data layer",
            "memory_type": "decision",
            "salience": {"composite": 0.7, "relevance": 0.6, "importance": 0.9, "novelty": 0.4},
            "created_at": (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat(),
            "last_accessed": (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat(),
        },
    )


async def test_semantic_retrieval_returns_matching_items():
    store = InMemoryStore()
    _populate_store(store)
    retrieval = HybridRetrievalSystem(store)

    query = RetrievalQuery(text="deploy production server", max_results=5)
    results = await retrieval.retrieve(query, strategies=[RetrievalStrategy.SEMANTIC])

    keys = [r.key for r in results]
    assert "mem-1" in keys


async def test_semantic_retrieval_ranks_by_similarity():
    store = InMemoryStore()
    _populate_store(store)
    retrieval = HybridRetrievalSystem(store)

    query = RetrievalQuery(text="login bug fix release", max_results=2)
    results = await retrieval.retrieve(query, strategies=[RetrievalStrategy.SEMANTIC])

    assert results[0].key == "mem-2"


async def test_temporal_retrieval_filters_by_range():
    store = InMemoryStore()
    _populate_store(store)
    retrieval = HybridRetrievalSystem(store)

    now = datetime.now(timezone.utc)
    query = RetrievalQuery(
        text="",
        temporal_range=(now - timedelta(hours=6), now),
        max_results=10,
    )
    results = await retrieval.retrieve(query, strategies=[RetrievalStrategy.TEMPORAL])

    keys = [r.key for r in results]
    assert "mem-3" not in keys


async def test_temporal_retrieval_excludes_old_items():
    store = InMemoryStore()
    _populate_store(store)
    retrieval = HybridRetrievalSystem(store)

    last_week = datetime.now(timezone.utc) - timedelta(days=7)
    query = RetrievalQuery(
        text="",
        temporal_range=(last_week - timedelta(days=60), last_week),
        max_results=10,
    )
    results = await retrieval.retrieve(query, strategies=[RetrievalStrategy.TEMPORAL])

    keys = [r.key for r in results]
    assert "mem-3" in keys
    assert "mem-2" not in keys


async def test_fusion_deduplicates():
    store = InMemoryStore()
    _populate_store(store)
    retrieval = HybridRetrievalSystem(store)

    query = RetrievalQuery(text="deploy production", max_results=10)
    semantic = await retrieval.retrieve(query, strategies=[RetrievalStrategy.SEMANTIC])
    temporal = await retrieval.retrieve(
        RetrievalQuery(text="", temporal_range=(datetime.now(timezone.utc) - timedelta(days=365), datetime.now(timezone.utc))),
        strategies=[RetrievalStrategy.TEMPORAL],
    )
    fused = retrieval._fusion({"semantic": semantic, "temporal": temporal})

    keys = [r.key for r in fused]
    assert len(keys) == len(set(keys))


async def test_fusion_sorts_by_salience_desc():
    store = InMemoryStore()
    _populate_store(store)
    retrieval = HybridRetrievalSystem(store)

    query = RetrievalQuery(text="", max_results=10)
    results = await retrieval.retrieve(query)

    for i in range(len(results) - 1):
        s1 = results[i].salience.composite if results[i].salience else 0
        s2 = results[i + 1].salience.composite if results[i + 1].salience else 0
        assert s1 >= s2


async def test_memory_type_filtering():
    store = InMemoryStore()
    _populate_store(store)
    retrieval = HybridRetrievalSystem(store)

    query = RetrievalQuery(text="", memory_types=[MemoryType.GOAL], max_results=10)
    results = await retrieval.retrieve(query, strategies=[RetrievalStrategy.SEMANTIC])

    assert all(r.key == "mem-2" for r in results)


async def test_min_salience_filter():
    store = InMemoryStore()
    _populate_store(store)
    retrieval = HybridRetrievalSystem(store)

    query = RetrievalQuery(text="", min_salience=0.6, max_results=10)
    results = await retrieval.retrieve(query, strategies=[RetrievalStrategy.SEMANTIC])

    for r in results:
        assert r.salience is None or r.salience.composite >= 0.6


async def test_empty_store_returns_empty():
    store = InMemoryStore()
    retrieval = HybridRetrievalSystem(store)

    query = RetrievalQuery(text="anything", max_results=10)
    results = await retrieval.retrieve(query)

    assert results == []


async def test_retrieve_default_strategies():
    store = InMemoryStore()
    _populate_store(store)
    retrieval = HybridRetrievalSystem(store)

    query = RetrievalQuery(text="deploy", max_results=10)
    results = await retrieval.retrieve(query)

    assert len(results) > 0
