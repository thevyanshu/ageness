from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    DECISION = "decision"
    GOAL = "goal"


class SalienceScore(BaseModel):
    relevance: float = Field(ge=0.0, le=1.0, description="Relevance to current context")
    importance: float = Field(ge=0.0, le=1.0, description="Inherent importance")
    novelty: float = Field(ge=0.0, le=1.0, description="Novelty relative to existing memories")
    composite: float = Field(ge=0.0, le=1.0, description="Weighted composite score")


class RawMemory(BaseModel):
    content: str
    memory_type: MemoryType
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_thread_id: str | None = None
    source_checkpoint_id: str | None = None


class MemoryItem(BaseModel):
    key: str
    namespace: tuple[str, ...]
    value: dict[str, Any]
    salience: SalienceScore | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)


class ContextWindow(BaseModel):
    episodes: list[MemoryItem] = Field(default_factory=list)
    decisions: list[MemoryItem] = Field(default_factory=list)
    goals: list[MemoryItem] = Field(default_factory=list)
    facts: list[MemoryItem] = Field(default_factory=list)
    token_budget: int = 4000
    total_tokens: int = 0


class CognitiveState(BaseModel):
    active_goals: list[dict[str, Any]] = Field(default_factory=list)
    relevant_decisions: list[dict[str, Any]] = Field(default_factory=list)
    unresolved_tasks: list[dict[str, Any]] = Field(default_factory=list)
    contextual_dependencies: dict[str, Any] = Field(default_factory=dict)
    reconstruction_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class DistillationResult(BaseModel):
    decisions: list[dict[str, Any]] = Field(default_factory=list)
    unresolved_goals: list[dict[str, Any]] = Field(default_factory=list)
    compressed_trace: str | None = None
    memory_items: list[RawMemory] = Field(default_factory=list)


class RetrievalQuery(BaseModel):
    text: str
    memory_types: list[MemoryType] | None = None
    max_results: int = 10
    min_salience: float = 0.0
    temporal_range: tuple[datetime, datetime] | None = None
