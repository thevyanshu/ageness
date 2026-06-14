from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ageness.cognition.retrieval.hybrid import HybridRetrievalSystem
from ageness.memory.models import ContextWindow, MemoryItem, RetrievalQuery


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
        items = await self.retrieval.retrieve(
            query=RetrievalQuery(
                text=" ".join(goals),
                max_results=20,
            )
        )
        return self._assemble_window(items)

    def _assemble_window(self, items: list[MemoryItem]) -> ContextWindow:
        return ContextWindow(token_budget=self.max_tokens)
