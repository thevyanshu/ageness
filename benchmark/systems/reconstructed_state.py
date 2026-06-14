from __future__ import annotations

from datetime import datetime, timezone

from langgraph.store.memory import InMemoryStore

from ageness.cognition.active_context.builder import ActiveContextBuilder
from ageness.cognition.distillation.pipeline import AsyncDistillationPipeline
from ageness.cognition.reconstruction.reconstructor import CognitiveStateReconstruction
from ageness.cognition.retrieval.hybrid import HybridRetrievalSystem
from ageness.cognition.salience.engine import MemorySalienceEngine
from ageness.memory.models import MemoryItem
from benchmark.models import (
    BenchmarkConfig,
    MetricsSnapshot,
    SystemAResult,
)


class ReconstructedStateSystem:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.store = InMemoryStore()
        self._turn_count = 0

        def _embed(text: str) -> list[float]:
            if not config.lm_studio_url:
                return []
            import requests
            try:
                r = requests.post(
                    f"{config.lm_studio_url}/v1/embeddings",
                    json={"model": config.embedding_model, "input": text},
                    timeout=15,
                )
                r.raise_for_status()
                return r.json()["data"][0]["embedding"]
            except Exception:
                return []

        self.retrieval = HybridRetrievalSystem(
            self.store, embedding_model=_embed,
        )
        self.salience = MemorySalienceEngine()
        self.context_builder = ActiveContextBuilder(self.retrieval)
        self.distillation = AsyncDistillationPipeline(self.salience)
        self.reconstruction = CognitiveStateReconstruction(self.retrieval)

        self._goals: list[str] = []
        self._decisions: list[str] = []
        self._messages: list[dict] = []

    @property
    def name(self) -> str:
        return "reconstructed_state"

    def reset(self) -> None:
        self.store = InMemoryStore()
        self._turn_count = 0
        self._goals.clear()
        self._decisions.clear()
        self._messages.clear()

        def _embed(text: str) -> list[float]:
            if not self.config.lm_studio_url:
                return []
            import requests
            try:
                r = requests.post(
                    f"{self.config.lm_studio_url}/v1/embeddings",
                    json={
                        "model": self.config.embedding_model,
                        "input": text,
                    },
                    timeout=15,
                )
                r.raise_for_status()
                return r.json()["data"][0]["embedding"]
            except Exception:
                return []

        self.retrieval = HybridRetrievalSystem(
            self.store, embedding_model=_embed,
        )
        self.context_builder = ActiveContextBuilder(self.retrieval)
        self.reconstruction = CognitiveStateReconstruction(self.retrieval)

    def _count_tokens(self, text: str) -> int:
        return max(len(text) // 4, 1)

    def _call_llm(self, prompt: str) -> str:
        if self.config.lm_studio_url:
            import requests
            try:
                r = requests.post(
                    f"{self.config.lm_studio_url}/v1/chat/completions",
                    json={
                        "model": self.config.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": self.config.temperature,
                        "max_tokens": self.config.max_output_tokens,
                    },
                    timeout=30,
                )
                r.raise_for_status()
                msg = r.json()["choices"][0]["message"]
                content = (msg.get("content") or "").strip()
                return content or (msg.get("reasoning_content") or "").strip()
            except Exception:
                pass
        return f"Mock response for: {prompt[:50]}..."

    async def process_turn(self, user_input: str) -> SystemAResult:
        self._turn_count += 1
        start = datetime.now(timezone.utc)

        retrieval_start = datetime.now(timezone.utc)
        context_window = await self.context_builder.build_context(
            current_state={},
            thread_id="bench",
            goals=self._goals or [user_input],
        )
        retrieval_end = datetime.now(timezone.utc)
        retrieval_ms = (retrieval_end - retrieval_start).total_seconds() * 1000

        context_parts: list[str] = []
        for item in context_window.goals:
            context_parts.append(f"[Goal] {item.value.get('content', '')}")
        for item in context_window.decisions:
            context_parts.append(f"[Decision] {item.value.get('content', '')}")
        for item in context_window.episodes:
            context_parts.append(f"[Episode] {item.value.get('content', '')}")
        for item in context_window.facts:
            context_parts.append(f"[Fact] {item.value.get('content', '')}")
        context = "\n".join(context_parts) if context_parts else "No prior context."

        context_tokens = self._count_tokens(context)

        prompt = (
            f"Current cognitive state:\n{context}\n\n"
            f"User: {user_input}\n"
            f"Assistant:"
        )

        inference_start = datetime.now(timezone.utc)
        output = self._call_llm(prompt)
        inference_end = datetime.now(timezone.utc)
        inference_ms = (inference_end - inference_start).total_seconds() * 1000

        self._messages.append({"role": "user", "content": user_input})
        self._messages.append({"role": "assistant", "content": output})

        distill_start = datetime.now(timezone.utc)
        state = {
            "input": user_input,
            "output": output,
            "messages": list(self._messages),
            "goals": list(self._goals),
            "metadata": [],
            "thread_id": "bench",
        }
        distilled = await self.distillation.distill(
            thread_id="bench", state=state,
        )

        if self._goals:
            existing_goals = set(g.lower() for g in self._goals)
        else:
            existing_goals = set()
        for g in distilled.unresolved_goals:
            gt = g.get("goal", "")
            if gt and gt.lower() not in existing_goals:
                self._goals.append(gt)
                existing_goals.add(gt.lower())

        for raw in distilled.memory_items:
            existing_items = [
                MemoryItem(key=item.key, namespace=item.namespace, value=dict(item.value))
                for item in self.store.search(("*", raw.memory_type.value))
            ]
            scored = await self.salience.score(raw, existing_items)
            ns = ("*", raw.memory_type.value)
            self.store.put(ns, raw.memory_type.value, {
                "content": raw.content,
                "memory_type": raw.memory_type.value,
                "salience": {
                    "relevance": scored.relevance,
                    "importance": scored.importance,
                    "novelty": scored.novelty,
                    "composite": scored.composite,
                },
                "metadata": raw.metadata,
                "source_thread_id": raw.source_thread_id,
                "created_at": raw.timestamp.isoformat(),
                "last_accessed": raw.timestamp.isoformat(),
            })
        distill_end = datetime.now(timezone.utc)
        distill_ms = (distill_end - distill_start).total_seconds() * 1000

        stored_memories = sum(
            1 for _ in self.store.search(("*", "decision"))
        ) + sum(
            1 for _ in self.store.search(("*", "goal"))
        ) + sum(
            1 for _ in self.store.search(("*", "episodic"))
        )

        total_end = datetime.now(timezone.utc)
        total_ms = (total_end - start).total_seconds() * 1000

        metrics = MetricsSnapshot(
            turn_id=self._turn_count,
            input_tokens=self._count_tokens(user_input),
            output_tokens=self._count_tokens(output),
            context_size_tokens=context_tokens,
            retrieval_latency_ms=retrieval_ms,
            inference_latency_ms=inference_ms + distill_ms,
            total_latency_ms=total_ms,
            memories_retrieved=len(context_window.goals)
                + len(context_window.decisions)
                + len(context_window.episodes)
                + len(context_window.facts),
            active_memory_count=stored_memories,
            output=output,
        )

        return SystemAResult(output=output, metrics=metrics)
