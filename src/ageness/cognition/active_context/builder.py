from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ageness.cognition.retrieval.hybrid import HybridRetrievalSystem
from ageness.memory.models import ContextWindow, MemoryItem, MemoryType, RetrievalQuery


@dataclass
class ActiveContextBuilder:
    retrieval: HybridRetrievalSystem
    max_tokens: int = 4000
    _priority_order: list[str] = field(
        default_factory=lambda: ["goals", "decisions", "episodes", "facts"]
    )

    async def build_context(
        self,
        current_state: dict[str, Any],
        thread_id: str,
        goals: list[str],
    ) -> ContextWindow:
        query = RetrievalQuery(
            text=" ".join(goals),
            max_results=40,
            metadata={"thread_id": thread_id},
        )
        items = await self.retrieval.retrieve(query)
        return self._assemble_window(items)

    def _assemble_window(self, items: list[MemoryItem]) -> ContextWindow:
        classified = self._classify_items(items)
        sorted_items = self._sort_by_priority(classified)
        window = ContextWindow(token_budget=self.max_tokens)

        for item in sorted_items:
            item_tokens = self._estimate_tokens(item)
            if window.total_tokens + item_tokens > self.max_tokens:
                continue
            self._add_to_window(window, item)
            window.total_tokens += item_tokens

        return window

    def _classify_items(
        self, items: list[MemoryItem]
    ) -> dict[str, list[MemoryItem]]:
        classified: dict[str, list[MemoryItem]] = {
            "goals": [],
            "decisions": [],
            "episodes": [],
            "facts": [],
        }
        for item in items:
            mt = item.value.get("memory_type", "")
            if mt == MemoryType.GOAL.value:
                classified["goals"].append(item)
            elif mt == MemoryType.DECISION.value:
                classified["decisions"].append(item)
            elif mt == MemoryType.EPISODIC.value:
                classified["episodes"].append(item)
            else:
                classified["facts"].append(item)
        return classified

    def _sort_by_priority(
        self, classified: dict[str, list[MemoryItem]]
    ) -> list[MemoryItem]:
        result: list[MemoryItem] = []
        for category in self._priority_order:
            items = classified.get(category, [])
            items.sort(
                key=lambda x: x.salience.composite if x.salience else 0.0,
                reverse=True,
            )
            result.extend(items)
        return result

    def _add_to_window(
        self, window: ContextWindow, item: MemoryItem
    ) -> None:
        mt = item.value.get("memory_type", "")
        if mt == MemoryType.GOAL.value:
            window.goals.append(item)
        elif mt == MemoryType.DECISION.value:
            window.decisions.append(item)
        elif mt == MemoryType.EPISODIC.value:
            window.episodes.append(item)
        else:
            window.facts.append(item)

    def _estimate_tokens(self, item: MemoryItem) -> int:
        content = item.value.get("content", "") or ""
        return max(len(content) // 4, 1)


