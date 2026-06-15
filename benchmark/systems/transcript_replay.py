from __future__ import annotations

from datetime import datetime, timezone

from benchmark.models import (
    BenchmarkConfig,
    ContextTruncation,
    MetricsSnapshot,
    SystemAResult,
)


class TranscriptReplaySystem:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self._history: list[dict[str, str]] = []
        self._summary: str | None = None

    @property
    def name(self) -> str:
        return "transcript_replay"

    def reset(self) -> None:
        self._history.clear()
        self._summary = None

    def _count_tokens(self, text: str) -> int:
        return max(len(text) // 4, 1)

    def _build_context(self) -> tuple[str, int]:
        if self.config.truncation == ContextTruncation.LAST_N:
            recent = self._history[-self.config.last_n_messages:]
            context = "\n".join(
                f"{m['role']}: {m['content']}" for m in recent
            )
            return context, self._count_tokens(context)

        elif self.config.truncation == ContextTruncation.ROLLING_SUMMARY:
            if not self._history:
                return "", 0
            recent = self._history[-self.config.last_n_messages:]
            recent_text = "\n".join(
                f"{m['role']}: {m['content']}" for m in recent
            )
            summary_part = ""
            if self._summary:
                summary_part = f"[Summary of earlier conversation]: {self._summary}\n"
            context = summary_part + recent_text
            return context, self._count_tokens(context)

        elif self.config.truncation == ContextTruncation.TOKEN_BUDGET:
            context_parts: list[str] = []
            token_count = 0
            for msg in reversed(self._history):
                line = f"{msg['role']}: {msg['content']}"
                tokens = self._count_tokens(line)
                if token_count + tokens > self.config.token_limit:
                    break
                context_parts.insert(0, line)
                token_count += tokens
            context = "\n".join(context_parts)
            return context, token_count

        return "", 0

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

        self._history.append({"role": "user", "content": user_input})

        context, context_tokens = self._build_context()

        prompt = (
            f"Conversation history:\n{context}\n\n"
            f"User: {user_input}\n"
            f"Assistant:"
        )

        inference_start = datetime.now(timezone.utc)
        output = self._call_llm(prompt)
        inference_end = datetime.now(timezone.utc)

        self._history.append({"role": "assistant", "content": output})

        if self.config.truncation == ContextTruncation.ROLLING_SUMMARY \
                and self._summary is None:
            self._summary = self._generate_summary()

        total_end = datetime.now(timezone.utc)

        total_ms = (total_end - start).total_seconds() * 1000
        inference_ms = (inference_end - inference_start).total_seconds() * 1000

        import json
        checkpoint_bytes = len(json.dumps(self._history))

        metrics = MetricsSnapshot(
            turn_id=len(self._history) // 2,
            input_tokens=self._count_tokens(user_input),
            output_tokens=self._count_tokens(output),
            context_size_tokens=context_tokens,
            inference_latency_ms=inference_ms,
            total_latency_ms=total_ms,
            active_memory_count=len(self._history),
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

    def get_history_length(self) -> int:
        return len(self._history)
