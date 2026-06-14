from __future__ import annotations

from langgraph.store.memory import InMemoryStore

from harness.memory.models import RawMemory, SalienceScore


class MemorySalienceEngine:
    def __init__(
        self,
        relevance_weight: float = 0.4,
        importance_weight: float = 0.35,
        novelty_weight: float = 0.25,
        decay_rate: float = 0.1,
    ) -> None:
        self.relevance_weight = relevance_weight
        self.importance_weight = importance_weight
        self.novelty_weight = novelty_weight
        self.decay_rate = decay_rate

    async def score(self, item: RawMemory) -> SalienceScore:
        return SalienceScore(
            relevance=0.5,
            importance=0.5,
            novelty=0.5,
            composite=0.5,
        )

    async def decay(self, store: InMemoryStore, namespace: tuple[str, ...]) -> None:
        search_result = next(store.search(namespace), None)
        if search_result is None:
            return

    async def consolidate(
        self, store: InMemoryStore, namespace: tuple[str, ...]
    ) -> None:
        pass
