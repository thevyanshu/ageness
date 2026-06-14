from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MemoryArchitecture(str, Enum):
    TRANSCRIPT_REPLAY = "transcript_replay"
    HYBRID_RETRIEVAL = "hybrid_retrieval"
    RECONSTRUCTED_STATE = "reconstructed_state"


class ContextTruncation(str, Enum):
    LAST_N = "last_n"
    ROLLING_SUMMARY = "rolling_summary"
    TOKEN_BUDGET = "token_budget"


class ConversationTurn(BaseModel):
    turn_id: int
    role: str
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class InjectedFact(BaseModel):
    fact_id: str
    turn_id: int
    content: str
    category: str
    expected_recall_turns: list[int] = Field(default_factory=list)


class Scenario(BaseModel):
    scenario_id: str
    title: str
    description: str
    turns: list[ConversationTurn]
    facts: list[InjectedFact] = Field(default_factory=list)
    expected_decisions: list[str] = Field(default_factory=list)
    expected_goals: list[str] = Field(default_factory=list)
    query_pairs: list[dict[str, Any]] = Field(default_factory=list)


class MetricsSnapshot(BaseModel):
    turn_id: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    input_tokens: int = 0
    output_tokens: int = 0
    context_size_tokens: int = 0
    retrieval_latency_ms: float = 0.0
    inference_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    memories_retrieved: int = 0
    hallucination_indicators: list[str] = Field(default_factory=list)
    contradiction_indicators: list[str] = Field(default_factory=list)
    retrieval_confidence: float = 0.0
    active_memory_count: int = 0


class BenchmarkConfig(BaseModel):
    architecture: MemoryArchitecture
    truncation: ContextTruncation = ContextTruncation.LAST_N
    max_turns: int = 100
    last_n_messages: int = 20
    token_limit: int = 4096
    model: str = "qwen/qwen3.5-9b"
    embedding_model: str = "text-embedding-nomic-embed-text-v1.5"
    lm_studio_url: str = "http://localhost:1234"
    temperature: float = 0.1
    max_output_tokens: int = 256
    seed: int = 42


class BenchmarkRun(BaseModel):
    run_id: str
    scenario_id: str
    config: BenchmarkConfig
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    metrics: list[MetricsSnapshot] = Field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_ms: float = 0.0
    decisions_matched: int = 0
    decisions_total: int = 0
    facts_recalled: int = 0
    facts_total: int = 0
    contradictions_detected: int = 0
    hallucination_events: int = 0


class SystemAResult(BaseModel):
    output: str
    metrics: MetricsSnapshot | None = None


class EvaluationScore(BaseModel):
    name: str
    value: float
    description: str
    metadata: dict[str, Any] = Field(default_factory=dict)
