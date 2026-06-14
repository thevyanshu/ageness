from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ageness.cognition.salience.engine import MemorySalienceEngine
from ageness.memory.models import DistillationResult, MemoryType, RawMemory


class AsyncDistillationPipeline:
    def __init__(
        self,
        salience_engine: MemorySalienceEngine,
        checkpointer: Any | None = None,
    ) -> None:
        self.salience_engine = salience_engine
        self.checkpointer = checkpointer

    async def distill(
        self,
        thread_id: str,
        checkpoint_id: str | None = None,
        state: dict[str, Any] | None = None,
    ) -> DistillationResult:
        if state is None and self.checkpointer and thread_id:
            state = await self._fetch_state(thread_id, checkpoint_id)
        if state is None:
            state = {}

        decisions = await self._extract_decisions(state)
        unresolved_goals = await self._identify_unresolved_goals(state)
        compressed_trace = await self._compress_trace(state)

        result = DistillationResult(
            decisions=decisions,
            unresolved_goals=unresolved_goals,
            compressed_trace=compressed_trace,
        )
        result.memory_items = await self._build_memories(result)
        return result

    async def _fetch_state(
        self, thread_id: str, checkpoint_id: str | None = None
    ) -> dict[str, Any]:
        if not self.checkpointer:
            return {}
        config = {"configurable": {"thread_id": thread_id}}
        try:
            if checkpoint_id:
                for state_snapshot in self.checkpointer.list(config):
                    if str(state_snapshot.checkpoint_id) == checkpoint_id:
                        return dict(state_snapshot.values)
                return {}

            latest = self.checkpointer.get(config)
            if latest is not None:
                return dict(latest.values)
            return {}
        except Exception:
            return {}

    async def _extract_decisions(
        self, state: dict[str, Any]
    ) -> list[dict[str, Any]]:
        decisions: list[dict[str, Any]] = []

        explicit = state.get("decisions") or state.get("decisions_made") or []
        if isinstance(explicit, list):
            decisions.extend(explicit)

        metadata = state.get("metadata") or []
        if isinstance(metadata, list):
            for entry in metadata:
                if isinstance(entry, dict) and entry.get("decision"):
                    decisions.append(
                        {
                            "decision": entry["decision"],
                            "rationale": entry.get("rationale", ""),
                            "timestamp": entry.get(
                                "timestamp", datetime.now(timezone.utc).isoformat()
                            ),
                        }
                    )

        messages = state.get("messages") or []
        if isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, dict):
                    content = str(msg.get("content", ""))
                    role = msg.get("role", "")
                    if role == "assistant" and any(
                        keyword in content.lower()
                        for keyword in [
                            "i decided",
                            "i choose",
                            "selected",
                            "opted for",
                            "decision:",
                        ]
                    ):
                        decisions.append(
                            {
                                "decision": content[:200],
                                "rationale": "extracted from assistant response",
                                "timestamp": msg.get(
                                    "timestamp",
                                    datetime.now(timezone.utc).isoformat(),
                                ),
                                "source": "message",
                            }
                        )

        seen = set()
        unique: list[dict[str, Any]] = []
        for d in decisions:
            key = d.get("decision", "")[:80]
            if key not in seen:
                seen.add(key)
                unique.append(d)
        return unique

    async def _identify_unresolved_goals(
        self, state: dict[str, Any]
    ) -> list[dict[str, Any]]:
        goals = state.get("goals") or state.get("active_goals") or []
        if not isinstance(goals, list):
            goals = [goals] if isinstance(goals, dict) else []

        resolved_statuses = {"completed", "resolved", "done", "cancelled", "failed"}
        unresolved: list[dict[str, Any]] = []
        for goal in goals:
            if isinstance(goal, str):
                unresolved.append(
                    {
                        "goal": goal,
                        "status": "active",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
            elif isinstance(goal, dict):
                status = str(goal.get("status", "active")).lower()
                if status not in resolved_statuses:
                    unresolved.append(
                        {
                            "goal": goal.get("goal") or goal.get("description", ""),
                            "status": status,
                            "timestamp": goal.get(
                                "timestamp",
                                datetime.now(timezone.utc).isoformat(),
                            ),
                        }
                    )

        return unresolved

    async def _compress_trace(
        self, state: dict[str, Any]
    ) -> str | None:
        parts: list[str] = []

        inp = state.get("input")
        if inp:
            inp_str = str(inp)[:300]
            parts.append(f"Input: {inp_str}")

        out = state.get("output")
        if out:
            out_str = str(out)[:300]
            parts.append(f"Output: {out_str}")

        messages = state.get("messages") or []
        if isinstance(messages, list) and messages:
            msg_summary = self._summarize_messages(messages)
            if msg_summary:
                parts.append(msg_summary)

        for key in ("context", "result", "summary", "reasoning"):
            val = state.get(key)
            if val and key not in ("messages",):
                val_str = str(val)[:200]
                parts.append(f"{key.capitalize()}: {val_str}")

        if not parts:
            return None

        return " | ".join(parts)

    def _summarize_messages(self, messages: list) -> str:
        user_msgs = 0
        asst_msgs = 0
        tool_msgs = 0
        for msg in messages:
            role = ""
            if isinstance(msg, dict):
                role = msg.get("role", "")
            elif hasattr(msg, "type"):
                role = msg.type
            else:
                role = str(type(msg).__name__).lower()
            if role == "user":
                user_msgs += 1
            elif role == "assistant":
                asst_msgs += 1
            elif role == "tool":
                tool_msgs += 1

        summary = f"Messages: {user_msgs} user, {asst_msgs} assistant"
        if tool_msgs:
            summary += f", {tool_msgs} tool"
        return summary

    async def _build_memories(
        self, result: DistillationResult
    ) -> list[RawMemory]:
        memories: list[RawMemory] = []

        for decision in result.decisions:
            memory = RawMemory(
                content=str(decision.get("decision", "")),
                memory_type=MemoryType.DECISION,
                metadata={
                    "rationale": decision.get("rationale", ""),
                    "timestamp": decision.get("timestamp", ""),
                    "source": decision.get("source", "distillation"),
                },
                source_thread_id=decision.get("thread_id"),
            )
            memories.append(memory)

        for goal in result.unresolved_goals:
            memory = RawMemory(
                content=str(goal.get("goal", "")),
                memory_type=MemoryType.GOAL,
                metadata={
                    "status": goal.get("status", "active"),
                    "timestamp": goal.get("timestamp", ""),
                },
                source_thread_id=goal.get("thread_id"),
            )
            memories.append(memory)

        if result.compressed_trace:
            memory = RawMemory(
                content=result.compressed_trace,
                memory_type=MemoryType.EPISODIC,
                metadata={"type": "compressed_trace"},
            )
            memories.append(memory)

        return memories
