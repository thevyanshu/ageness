from __future__ import annotations

from typing import Any

from ageness.cognition.retrieval.hybrid import HybridRetrievalSystem
from ageness.memory.models import CognitiveState, MemoryType, RetrievalQuery


class CognitiveStateReconstruction:
    def __init__(self, retrieval: HybridRetrievalSystem) -> None:
        self.retrieval = retrieval

    async def reconstruct(
        self, thread_id: str, query: str
    ) -> CognitiveState:
        goals = await self._reconstruct_goals(thread_id)
        decisions = await self._surface_decisions(thread_id)
        unresolved = await self._find_unresolved_tasks(thread_id)
        dependencies = await self._resolve_dependencies(thread_id)

        confidence = self._compute_confidence(goals, decisions, unresolved)
        return CognitiveState(
            active_goals=goals,
            relevant_decisions=decisions,
            unresolved_tasks=unresolved,
            contextual_dependencies=dependencies,
            reconstruction_confidence=confidence,
        )

    async def _reconstruct_goals(self, thread_id: str) -> list[dict[str, Any]]:
        query = RetrievalQuery(
            text="",
            memory_types=[MemoryType.GOAL],
            max_results=20,
            metadata={"thread_id": thread_id},
        )
        items = await self.retrieval.retrieve(query)
        resolved = {"completed", "resolved", "done", "cancelled", "failed"}
        goals: list[dict[str, Any]] = []
        for item in items:
            status = item.value.get("status", "active")
            if status.lower() in resolved:
                continue
            content = item.value.get("content", "") or ""
            salience = item.value.get("salience", {}) or {}
            goals.append(
                {
                    "goal": content,
                    "status": status,
                    "salience": salience.get("composite", 0.5),
                    "source_key": item.key,
                    "updated_at": (
                        item.last_accessed.isoformat()
                        if item.last_accessed
                        else ""
                    ),
                }
            )
        goals.sort(key=lambda g: g["salience"], reverse=True)
        return goals

    async def _surface_decisions(self, thread_id: str) -> list[dict[str, Any]]:
        query = RetrievalQuery(
            text="",
            memory_types=[MemoryType.DECISION],
            max_results=15,
            metadata={"thread_id": thread_id},
        )
        items = await self.retrieval.retrieve(query)
        decisions: list[dict[str, Any]] = []
        for item in items:
            content = item.value.get("content", "") or ""
            metadata = item.value.get("metadata", {}) or {}
            salience = item.value.get("salience", {}) or {}
            decisions.append(
                {
                    "decision": content,
                    "rationale": metadata.get("rationale", ""),
                    "salience": salience.get("composite", 0.5),
                    "source_key": item.key,
                    "timestamp": (
                        item.created_at.isoformat() if item.created_at else ""
                    ),
                }
            )
        decisions.sort(key=lambda d: d["salience"], reverse=True)
        return decisions

    async def _find_unresolved_tasks(self, thread_id: str) -> list[dict[str, Any]]:
        query = RetrievalQuery(
            text="",
            memory_types=[MemoryType.EPISODIC, MemoryType.GOAL],
            max_results=15,
            metadata={"thread_id": thread_id},
        )
        items = await self.retrieval.retrieve(query)
        tasks: list[dict[str, Any]] = []
        for item in items:
            content = item.value.get("content", "") or ""
            status = item.value.get("status", "active")
            if status in ("completed", "resolved", "done", "cancelled", "failed"):
                continue
            salience = item.value.get("salience", {}) or {}
            tasks.append(
                {
                    "task": content,
                    "status": status,
                    "salience": salience.get("composite", 0.5),
                    "source_key": item.key,
                    "memory_type": item.value.get("memory_type", ""),
                }
            )
        tasks.sort(key=lambda t: t["salience"], reverse=True)
        return tasks

    async def _resolve_dependencies(
        self, thread_id: str
    ) -> dict[str, Any]:
        query = RetrievalQuery(
            text="",
            max_results=30,
            metadata={"thread_id": thread_id},
        )
        items = await self.retrieval.retrieve(query)
        if not items:
            return {}

        decision_keys: set[str] = set()
        goal_keys: set[str] = set()

        for item in items:
            mt = item.value.get("memory_type", "")
            if mt == MemoryType.DECISION.value:
                decision_keys.add(item.key)
            elif mt == MemoryType.GOAL.value:
                goal_keys.add(item.key)

        decisions_with_rationale: list[dict[str, Any]] = []
        for item in items:
            if item.key in decision_keys:
                metadata = item.value.get("metadata", {}) or {}
                decisions_with_rationale.append(
                    {
                        "decision": (item.value.get("content", "") or "")[:100],
                        "rationale": metadata.get("rationale", ""),
                        "key": item.key,
                    }
                )

        return {
            "related_goal_keys": list(goal_keys),
            "related_decision_keys": list(decision_keys),
            "decisions_with_rationale": decisions_with_rationale,
            "total_memories_scanned": len(items),
        }

    def _compute_confidence(
        self,
        goals: list[dict[str, Any]],
        decisions: list[dict[str, Any]],
        tasks: list[dict[str, Any]],
    ) -> float:
        total_items = len(goals) + len(decisions) + len(tasks)
        if total_items == 0:
            return 0.0

        has_goals = min(len(goals) / 5, 1.0) * 0.35
        has_decisions = min(len(decisions) / 5, 1.0) * 0.30
        has_tasks = min(len(tasks) / 5, 1.0) * 0.35
        return round(min(has_goals + has_decisions + has_tasks, 1.0), 4)
