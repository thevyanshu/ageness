from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from langgraph.store.memory import InMemoryStore

from ageness.memory.models import MemoryItem, MemoryType, RetrievalQuery


class RetrievalStrategy(str, Enum):
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    CHECKPOINT = "checkpoint"
    GRAPH = "graph"


class HybridRetrievalSystem:
    def __init__(
        self,
        store: InMemoryStore,
        embedding_model: Any | None = None,
        checkpointer: Any | None = None,
    ) -> None:
        self.store = store
        self.embedding_model = embedding_model
        self.checkpointer = checkpointer

    async def retrieve(
        self,
        query: RetrievalQuery,
        strategies: list[RetrievalStrategy] | None = None,
    ) -> list[MemoryItem]:
        if strategies is None:
            strategies = [RetrievalStrategy.SEMANTIC, RetrievalStrategy.TEMPORAL]

        results: dict[str, list[MemoryItem]] = {}

        for strategy in strategies:
            if strategy == RetrievalStrategy.SEMANTIC:
                results["semantic"] = await self._semantic_retrieval(query)
            elif strategy == RetrievalStrategy.TEMPORAL:
                results["temporal"] = await self._temporal_retrieval(query)
            elif strategy == RetrievalStrategy.CHECKPOINT:
                results["checkpoint"] = await self._checkpoint_traversal(query)
            elif strategy == RetrievalStrategy.GRAPH:
                results["graph"] = await self._graph_retrieval(query)

        return self._fusion(results)

    async def _semantic_retrieval(self, query: RetrievalQuery) -> list[MemoryItem]:
        all_items: list[MemoryItem] = []
        query_words = set(query.text.lower().split())

        namespaces_to_search = self._namespaces_for_query(query)
        for ns in namespaces_to_search:
            for item in self.store.search(ns):
                parsed = self._store_item_to_memory_item(item, query)
                if parsed is None:
                    continue
                if query.memory_types and parsed.value.get("memory_type") not in query.memory_types:
                    continue
                all_items.append(parsed)

        if not query_words:
            return all_items[: query.max_results]

        scored = []
        for item in all_items:
            content = item.value.get("content", "") or ""
            item_words = set(content.lower().split())
            if not item_words:
                overlap = 0.0
            else:
                overlap = len(query_words & item_words) / len(query_words | item_words)
            score = overlap if not self.embedding_model else overlap
            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[: query.max_results]]

    async def _temporal_retrieval(self, query: RetrievalQuery) -> list[MemoryItem]:
        if not query.temporal_range:
            return []

        start, end = query.temporal_range
        results: list[MemoryItem] = []

        namespaces_to_search = self._namespaces_for_query(query)
        for ns in namespaces_to_search:
            for item in self.store.search(ns):
                parsed = self._store_item_to_memory_item(item, query)
                if parsed is None:
                    continue
                ts = parsed.created_at
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if start <= ts <= end:
                    results.append(parsed)

        results.sort(key=lambda x: x.created_at, reverse=True)
        return results[: query.max_results]

    async def _checkpoint_traversal(self, query: RetrievalQuery) -> list[MemoryItem]:
        if not self.checkpointer or not query.metadata.get("thread_id"):
            return []
        thread_id = query.metadata["thread_id"]
        config = {"configurable": {"thread_id": thread_id}}
        results: list[MemoryItem] = []
        for state in self.checkpointer.list(config):
            ts = state.created_at if hasattr(state, "created_at") else datetime.now(timezone.utc)
            results.append(
                MemoryItem(
                    key=f"ckpt_{state.checkpoint_id}",
                    namespace=("checkpoint", thread_id),
                    value={"state": state.values, "checkpoint_id": state.checkpoint_id},
                    created_at=ts,
                    last_accessed=ts,
                )
            )
        results.sort(key=lambda x: x.created_at, reverse=True)
        return results[: query.max_results]

    async def _graph_retrieval(self, query: RetrievalQuery) -> list[MemoryItem]:
        return []

    def _fusion(self, results: dict[str, list[MemoryItem]]) -> list[MemoryItem]:
        seen: set[str] = set()
        merged: list[MemoryItem] = []

        for strategy_name, items in results.items():
            for item in items:
                if item.key in seen:
                    continue
                seen.add(item.key)
                score = item.salience
                if score is None or score.composite < 0.0:
                    merged.append(item)
                elif score.composite >= 0.0:
                    merged.append(item)

        merged.sort(
            key=lambda x: x.salience.composite if x.salience else 0.0,
            reverse=True,
        )
        return merged

    def _namespaces_for_query(self, query: RetrievalQuery) -> list[tuple[str, ...]]:
        user_id = query.metadata.get("user_id", "*")
        mts = query.memory_types or list(MemoryType)
        return [(user_id, mt.value) for mt in mts]

    def _store_item_to_memory_item(
        self, store_item: Any, query: RetrievalQuery
    ) -> MemoryItem | None:
        try:
            key = store_item.key
            namespace = store_item.namespace
            value = store_item.value if isinstance(store_item.value, dict) else {}
            if not value:
                return None
            from ageness.memory.models import SalienceScore

            salience_data = value.get("salience")
            salience = None
            if isinstance(salience_data, dict):
                salience = SalienceScore(
                    relevance=salience_data.get("relevance", 0.5),
                    importance=salience_data.get("importance", 0.5),
                    novelty=salience_data.get("novelty", 0.5),
                    composite=salience_data.get("composite", 0.5),
                )

            if query.min_salience > 0 and salience and salience.composite < query.min_salience:
                return None

            ts = value.get("created_at")
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            elif ts is None:
                ts = datetime.now(timezone.utc)

            return MemoryItem(
                key=key,
                namespace=namespace,
                value=value,
                salience=salience,
                created_at=ts,
                last_accessed=value.get("last_accessed", ts),
            )
        except Exception:
            return None
