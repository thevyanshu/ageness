from __future__ import annotations

from typing import Any

from ageness.cognition.salience.engine import MemorySalienceEngine
from ageness.memory.models import DistillationResult, RawMemory


class AsyncDistillationPipeline:
    def __init__(self, salience_engine: MemorySalienceEngine) -> None:
        self.salience_engine = salience_engine

    async def distill(
        self, thread_id: str, checkpoint_id: str
    ) -> DistillationResult:
        return DistillationResult()

    async def _extract_decisions(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        return []

    async def _identify_unresolved_goals(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        return []

    async def _compress_trace(self, state: dict[str, Any]) -> str | None:
        return None

    async def _build_memories(
        self, result: DistillationResult
    ) -> list[RawMemory]:
        return []
