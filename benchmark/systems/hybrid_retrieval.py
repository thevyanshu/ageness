from __future__ import annotations

from datetime import datetime, timezone

from benchmark.models import (
    BenchmarkConfig,
    ContextTruncation,
    MetricsSnapshot,
    SystemAResult,
)


class HybridRetrievalSystemBench:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self._history: list[dict[str, str]] = []
        self._vector_store: list[dict] = []
        self._summary: str | None = None

    @property
    def name(self) -> str:
        return "hybrid_retrieval"

    def reset(self) -> None:
        self._history.clear()
        self._vector_store.clear()
        self._summary = None

    def _count_tokens(self, text: str) -> int:
        return max(len(text) // 4, 1)

    def _get_embedding(self, text: str) -> list[float]:
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

    def _cosine_sim(self, a: list[float], b: list[float]) -> float:
        if not a or not b:
            return 0.0
        import math
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def _vector_retrieve(self, query: str, top_k: int = 5) -> list[str]:
        q_emb = self._get_embedding(query)
        if not q_emb or not self._vector_store:
            return []

        scored = []
        for entry in self._vector_store:
            if entry.get("embedding"):
                sim = self._cosine_sim(q_emb, entry["embedding"])
                scored.append((sim, entry["text"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in scored[:top_k]]

    def _build_context(self) -> tuple[str, int]:
        if self.config.truncation == ContextTruncation.LAST_N:
            recent = self._history[-self.config.last_n_messages:]
            context = "\n".join(
                f"{m['role']}: {m['content']}" for m in recent
            )
            return context, self._count_tokens(context)

        recent = self._history[-self.config.last_n_messages:]
        recent_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in recent
        )

        retrieval_text = ""
        if recent:
            retrieval_text = "\n".join(
                self._vector_retrieve(recent[-1]["content"], top_k=3)
            )

        summary_part = ""
        if self.config.truncation == ContextTruncation.ROLLING_SUMMARY \
                and self._summary:
            summary_part = f"[Summary of earlier conversation]: {self._summary}\n"

        retrieval_part = ""
        if retrieval_text:
            retrieval_part = f"[Retrieved memories]:\n{retrieval_text}\n"

        context = summary_part + retrieval_part + recent_text
        return context, self._count_tokens(context)

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
        start = datetime.now(timezone.utc)

        retrieval_start = datetime.now(timezone.utc)
        retrieved_chunks = self._vector_retrieve(user_input, top_k=3)
        retrieval_end = datetime.now(timezone.utc)
        retrieval_ms = (retrieval_end - retrieval_start).total_seconds() * 1000

        self._history.append({"role": "user", "content": user_input})

        context, context_tokens = self._build_context()

        prompt = (
            f"Conversation history and memories:\n{context}\n\n"
            f"User: {user_input}\n"
            f"Assistant:"
        )

        inference_start = datetime.now(timezone.utc)
        output = self._call_llm(prompt)
        inference_end = datetime.now(timezone.utc)

        self._history.append({"role": "assistant", "content": output})

        emb = self._get_embedding(user_input + " " + output)
        self._vector_store.append({
            "text": f"user: {user_input}\nassistant: {output}",
            "embedding": emb,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        if self.config.truncation == ContextTruncation.ROLLING_SUMMARY \
                and self._summary is None:
            self._summary = self._generate_summary()

        total_end = datetime.now(timezone.utc)

        total_ms = (total_end - start).total_seconds() * 1000
        inference_ms = (inference_end - inference_start).total_seconds() * 1000

        import json
        checkpoint_bytes = (
            len(json.dumps(self._history))
            + len(json.dumps(self._vector_store))
        )

        metrics = MetricsSnapshot(
            turn_id=len(self._history) // 2,
            input_tokens=self._count_tokens(user_input),
            output_tokens=self._count_tokens(output),
            context_size_tokens=context_tokens,
            retrieval_latency_ms=retrieval_ms,
            inference_latency_ms=inference_ms,
            total_latency_ms=total_ms,
            memories_retrieved=len(retrieved_chunks),
            active_memory_count=len(self._history) + len(self._vector_store),
            checkpoint_size_bytes=checkpoint_bytes,
            output=output,
        )

        return SystemAResult(output=output, metrics=metrics)

    def _generate_summary(self) -> str:
        full = "\n".join(
            f"{m['role']}: {m['content']}" for m in self._history
        )
        if len(full) > 1000:
            return full[:1000] + "..."
        return full
