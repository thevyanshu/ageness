from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest
from langgraph.store.memory import InMemoryStore

from ageness.cognition.active_context.builder import ActiveContextBuilder
from ageness.cognition.distillation.pipeline import AsyncDistillationPipeline
from ageness.cognition.reconstruction.reconstructor import CognitiveStateReconstruction
from ageness.cognition.retrieval.hybrid import HybridRetrievalSystem, RetrievalQuery
from ageness.cognition.salience.engine import MemorySalienceEngine
from ageness.memory.models import MemoryType

LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://localhost:1234")


def _warm_up_llm() -> None:
    import requests
    try:
        requests.post(
            f"{LM_STUDIO_URL}/v1/chat/completions",
            json={
                "model": "qwen/qwen3.5-9b",
                "messages": [{"role": "user", "content": "ping"}],
                "temperature": 0.1,
                "max_tokens": 5,
            },
            timeout=120,
        )
    except Exception:
        pass


def _call_llm(prompt: str) -> str:
    import requests
    payload = {
        "model": "qwen/qwen3.5-9b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 512,
    }
    r = requests.post(f"{LM_STUDIO_URL}/v1/chat/completions", json=payload, timeout=120)
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    if not content:
        content = (msg.get("reasoning_content") or "").strip()
    return content


def _get_embedding(text: str) -> list[float]:
    import requests
    payload = {
        "model": "text-embedding-nomic-embed-text-v1.5",
        "input": text,
    }
    r = requests.post(f"{LM_STUDIO_URL}/v1/embeddings", json=payload, timeout=15)
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_REAL_WORLD"),
    reason="Set RUN_REAL_WORLD=1 to run real-world LM Studio tests",
)


async def test_retrieval_with_real_embeddings():
    store = InMemoryStore()
    store.put(
        ("*", "episodic"), "e1",
        {
            "content": "User reported login page crashes on submit",
            "memory_type": "episodic",
            "salience": {"composite": 0.7},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_accessed": datetime.now(timezone.utc).isoformat(),
        },
    )
    store.put(
        ("*", "episodic"), "e2",
        {
            "content": "Deployed hotfix for payment gateway timeout",
            "memory_type": "episodic",
            "salience": {"composite": 0.5},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_accessed": datetime.now(timezone.utc).isoformat(),
        },
    )
    store.put(
        ("*", "goal"), "g1",
        {
            "content": "Fix login page crash",
            "memory_type": "goal",
            "status": "active",
            "salience": {"composite": 0.9},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_accessed": datetime.now(timezone.utc).isoformat(),
        },
    )

    retrieval = HybridRetrievalSystem(store, embedding_model=_get_embedding)
    query = RetrievalQuery(
        text="login crash",
        memory_types=[MemoryType.GOAL, MemoryType.EPISODIC],
        max_results=5,
    )
    items = await retrieval.retrieve(query)
    assert len(items) >= 1
    contents = [i.value.get("content", "") for i in items]
    assert any("login" in c.lower() for c in contents)


async def test_full_pipeline_with_real_llm():
    _warm_up_llm()
    store = InMemoryStore()
    salience = MemorySalienceEngine()
    retrieval = HybridRetrievalSystem(store, embedding_model=_get_embedding)
    context_builder = ActiveContextBuilder(retrieval)
    distillation = AsyncDistillationPipeline(salience)
    reconstruction = CognitiveStateReconstruction(retrieval)

    store.put(
        ("*", "goal"), "g1",
        {
            "content": "Fix login page crash on submit button",
            "memory_type": "goal",
            "status": "active",
            "salience": {"composite": 0.9},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_accessed": datetime.now(timezone.utc).isoformat(),
        },
    )
    store.put(
        ("*", "episodic"), "e1",
        {
            "content": "User reported login page crashes when clicking submit with valid credentials",
            "memory_type": "episodic",
            "salience": {"composite": 0.7},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_accessed": datetime.now(timezone.utc).isoformat(),
        },
    )

    context = await context_builder.build_context(
        current_state={},
        thread_id="real-test-1",
        goals=["Fix login page crash on submit button"],
    )
    assert len(context.goals) >= 1
    assert len(context.episodes) >= 1

    context_text = "\n".join(
        f"[Goal] {g.value.get('content', '')}" for g in context.goals
    ) + "\n" + "\n".join(
        f"[Episode] {e.value.get('content', '')}" for e in context.episodes
    )

    prompt = (
        f"Context from memory:\n{context_text}\n\n"
        f"Given the context above, decide how to fix the login page crash. "
        f"State your decision clearly starting with 'Decision:' and provide a rationale."
    )
    llm_output = _call_llm(prompt)
    assert "decision" in llm_output.lower(), f"No 'decision' in LLM output: {llm_output[:300]}"

    messages = [
        {"role": "user", "content": "fix login page crash"},
        {"role": "assistant", "content": llm_output},
    ]

    state = {
        "input": "fix login page crash",
        "output": llm_output,
        "messages": messages,
        "goals": ["Fix login page crash on submit button"],
        "metadata": [],
        "thread_id": "real-test-1",
    }

    result = await distillation.distill(thread_id="real-test-1", state=state)
    assert len(result.decisions) >= 1, f"No decisions extracted. LLM said: {llm_output[:200]}"
    assert result.compressed_trace is not None

    for raw_memory in result.memory_items:
        existing_items = []
        for item in store.search(("*", raw_memory.memory_type.value)):
            from ageness.memory.models import MemoryItem
            existing_items.append(
                MemoryItem(key=item.key, namespace=item.namespace, value=dict(item.value))
            )
        scored = await salience.score(raw_memory, existing_items)
        ns = ("*", raw_memory.memory_type.value)
        store.put(
            ns,
            raw_memory.memory_type.value,
            {
                "content": raw_memory.content,
                "memory_type": raw_memory.memory_type.value,
                "salience": {
                    "relevance": scored.relevance,
                    "importance": scored.importance,
                    "novelty": scored.novelty,
                    "composite": scored.composite,
                },
                "metadata": raw_memory.metadata,
                "source_thread_id": raw_memory.source_thread_id,
                "created_at": raw_memory.timestamp.isoformat(),
                "last_accessed": raw_memory.timestamp.isoformat(),
            },
        )

    cognitive = await reconstruction.reconstruct(thread_id="real-test-1", query="login crash")
    assert len(cognitive.active_goals) >= 1
    assert len(cognitive.relevant_decisions) >= 1
    assert cognitive.reconstruction_confidence > 0.0
    assert any("login" in g.get("goal", "").lower() for g in cognitive.active_goals)
