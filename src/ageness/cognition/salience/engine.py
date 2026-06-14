from __future__ import annotations

import math
from datetime import datetime, timezone

from langgraph.store.base import Item
from langgraph.store.memory import InMemoryStore

from ageness.memory.models import MemoryItem, MemoryType, RawMemory, SalienceScore


class MemorySalienceEngine:
    def __init__(
        self,
        relevance_weight: float = 0.4,
        importance_weight: float = 0.35,
        novelty_weight: float = 0.25,
        decay_rate: float = 0.05,
        decay_threshold: float = 0.1,
        consolidation_similarity: float = 0.6,
    ) -> None:
        self.relevance_weight = relevance_weight
        self.importance_weight = importance_weight
        self.novelty_weight = novelty_weight
        self.decay_rate = decay_rate
        self.decay_threshold = decay_threshold
        self.consolidation_similarity = consolidation_similarity

        _total = relevance_weight + importance_weight + novelty_weight
        if abs(_total - 1.0) > 1e-6:
            self.relevance_weight /= _total
            self.importance_weight /= _total
            self.novelty_weight /= _total

    async def score(
        self,
        item: RawMemory,
        existing_items: list[MemoryItem] | None = None,
    ) -> SalienceScore:
        relevance = self._compute_relevance(item)
        importance = self._compute_importance(item)
        novelty = self._compute_novelty(item, existing_items or [])
        composite = (
            self.relevance_weight * relevance
            + self.importance_weight * importance
            + self.novelty_weight * novelty
        )
        return SalienceScore(
            relevance=round(relevance, 4),
            importance=round(importance, 4),
            novelty=round(novelty, 4),
            composite=round(composite, 4),
        )

    def _compute_relevance(self, item: RawMemory) -> float:
        base = {
            MemoryType.GOAL: 0.9,
            MemoryType.DECISION: 0.8,
            MemoryType.EPISODIC: 0.6,
            MemoryType.SEMANTIC: 0.5,
        }.get(item.memory_type, 0.5)
        freshness = self._freshness_factor(item.timestamp, half_life_hours=24)
        boost = 0.0
        if item.metadata.get("active"):
            boost += 0.15
        if item.metadata.get("direct_match"):
            boost += 0.2
        if item.metadata.get("error"):
            boost += 0.1
        return min(base * freshness + boost, 1.0)

    def _compute_importance(self, item: RawMemory) -> float:
        score = 0.3
        type_boost = {
            MemoryType.DECISION: 0.3,
            MemoryType.GOAL: 0.25,
            MemoryType.EPISODIC: 0.1,
            MemoryType.SEMANTIC: 0.05,
        }.get(item.memory_type, 0.1)
        score += type_boost
        if item.metadata.get("error"):
            score += 0.2
        if item.metadata.get("user_priority"):
            score += 0.2
        if item.metadata.get("task_count", 0) > 0:
            score += min(item.metadata["task_count"] * 0.05, 0.15)
        return min(score, 1.0)

    def _compute_novelty(
        self, item: RawMemory, existing_items: list[MemoryItem]
    ) -> float:
        if not existing_items:
            return 1.0
        max_overlap = 0.0
        item_words = set(item.content.lower().split())
        for existing in existing_items:
            existing_content = existing.value.get("content", "")
            if not existing_content:
                continue
            existing_words = set(existing_content.lower().split())
            if not item_words or not existing_words:
                continue
            overlap = len(item_words & existing_words) / len(item_words | existing_words)
            max_overlap = max(max_overlap, overlap)
        return round(1.0 - max_overlap, 4)

    def _freshness_factor(
        self, timestamp: datetime, half_life_hours: float = 24
    ) -> float:
        ts = timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
        if age_hours <= 0:
            return 1.0
        return 2.0 ** (-age_hours / half_life_hours)

    async def decay(
        self, store: InMemoryStore, namespace: tuple[str, ...]
    ) -> int:
        now = datetime.now(timezone.utc)
        removed = 0
        for item in store.search(namespace):
            last_accessed = item.value.get("last_accessed")
            if not last_accessed:
                continue
            if isinstance(last_accessed, str):
                last_accessed = datetime.fromisoformat(last_accessed)
            hours_since = (now - last_accessed).total_seconds() / 3600
            current_salience = item.value.get("salience", {}).get("composite", 0.5)
            decayed = current_salience * math.exp(-self.decay_rate * hours_since)
            if decayed < self.decay_threshold:
                store.delete(namespace, item.key)
                removed += 1
            else:
                payload = dict(item.value)
                payload.setdefault("salience", {})["composite"] = round(decayed, 4)
                store.put(namespace, item.key, payload)
        return removed

    async def consolidate(
        self, store: InMemoryStore, namespace: tuple[str, ...]
    ) -> int:
        items: list[Item] = list(store.search(namespace))
        if len(items) < 2:
            return 0

        merged_count = 0
        used: set[str] = set()

        for i, a in enumerate(items):
            if a.key in used:
                continue
            for j, b in enumerate(items):
                if j <= i or b.key in used:
                    continue
                similarity = self._text_similarity(
                    a.value.get("content", ""), b.value.get("content", "")
                )
                if similarity >= self.consolidation_similarity:
                    merged = self._merge_items(a, b)
                    store.put(namespace, a.key, merged)
                    store.delete(namespace, b.key)
                    used.add(b.key)
                    merged_count += 1

        return merged_count

    def _text_similarity(self, text_a: str, text_b: str) -> float:
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)

    def _merge_items(self, a: Item, b: Item) -> dict:
        a_content = a.value.get("content", "")
        b_content = b.value.get("content", "")
        merged_content = f"{a_content}\n---\n{b_content}" if a_content != b_content else a_content

        a_sal = a.value.get("salience", {}) or {}
        b_sal = b.value.get("salience", {}) or {}
        merged_salience = {
            "composite": round(
                max(
                    a_sal.get("composite", 0),
                    b_sal.get("composite", 0),
                ),
                4,
            )
        }

        a_ts = a.value.get("created_at")
        b_ts = b.value.get("created_at")
        if isinstance(a_ts, str):
            a_ts = datetime.fromisoformat(a_ts)
        if isinstance(b_ts, str):
            b_ts = datetime.fromisoformat(b_ts)
        earliest = min(
            t for t in (a_ts, b_ts) if t is not None
        ) or datetime.now(timezone.utc)

        return {
            "content": merged_content,
            "memory_type": a.value.get("memory_type", "episodic"),
            "salience": merged_salience,
            "created_at": earliest.isoformat(),
            "last_accessed": datetime.now(timezone.utc).isoformat(),
            "merged_from": [a.key, b.key],
            "source_thread_ids": list(
                set(
                    a.value.get("source_thread_ids", [a.value.get("source_thread_id")]) or []
                    + b.value.get("source_thread_ids", [b.value.get("source_thread_id")]) or []
                )
            ),
        }

    def score_sync(
        self,
        item: RawMemory,
        existing_items: list[MemoryItem] | None = None,
    ) -> SalienceScore:
        relevance = self._compute_relevance(item)
        importance = self._compute_importance(item)
        novelty = self._compute_novelty(item, existing_items or [])
        composite = (
            self.relevance_weight * relevance
            + self.importance_weight * importance
            + self.novelty_weight * novelty
        )
        return SalienceScore(
            relevance=round(relevance, 4),
            importance=round(importance, 4),
            novelty=round(novelty, 4),
            composite=round(composite, 4),
        )
