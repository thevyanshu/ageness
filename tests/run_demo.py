from __future__ import annotations

from datetime import datetime, timezone

import requests
from langgraph.store.memory import InMemoryStore

from ageness.cognition.active_context.builder import ActiveContextBuilder
from ageness.cognition.distillation.pipeline import AsyncDistillationPipeline
from ageness.cognition.reconstruction.reconstructor import CognitiveStateReconstruction
from ageness.cognition.retrieval.hybrid import HybridRetrievalSystem
from ageness.cognition.salience.engine import MemorySalienceEngine
from ageness.memory.models import MemoryItem

LM_STUDIO_URL = "http://localhost:1234"
MODEL = "qwen/qwen3.5-9b"
EMBED = "text-embedding-nomic-embed-text-v1.5"


def _llm(prompt: str) -> str:
    r = requests.post(
        f"{LM_STUDIO_URL}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 512,
        },
        timeout=120,
    )
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    return content or (msg.get("reasoning_content") or "").strip()


def _embed(text: str) -> list[float]:
    r = requests.post(
        f"{LM_STUDIO_URL}/v1/embeddings",
        json={"model": EMBED, "input": text},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


def say(label: str, *args) -> None:
    parts = " ".join(str(a) for a in args)
    print(f"[{label}] {parts}\n")


async def main() -> None:
    say("WARMUP", "Loading model...")
    try:
        requests.post(
            f"{LM_STUDIO_URL}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "ping"}],
                "temperature": 0.1,
                "max_tokens": 5,
            },
            timeout=120,
        )
    except Exception:
        pass

    # --- 1. Setup store with seed memories ---
    say("SETUP", "Creating store and cognitive modules")
    store = InMemoryStore()
    salience_engine = MemorySalienceEngine()
    retrieval = HybridRetrievalSystem(store, embedding_model=_embed)
    ctx_builder = ActiveContextBuilder(retrieval)
    distill_pipeline = AsyncDistillationPipeline(salience_engine)
    reconstruction = CognitiveStateReconstruction(retrieval)

    say("SETUP", "Storing seed memories (Goal: fix login, Decision: JWT, Episode: crash report)")

    store.put(
        ("*", "goal"),
        "g1",
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
        ("*", "goal"),
        "g2",
        {
            "content": "Migrate database to PostgreSQL",
            "memory_type": "goal",
            "status": "completed",
            "salience": {"composite": 0.5},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_accessed": datetime.now(timezone.utc).isoformat(),
        },
    )
    store.put(
        ("*", "decision"),
        "d1",
        {
            "content": "Use JWT for authentication tokens",
            "memory_type": "decision",
            "salience": {"composite": 0.85},
            "metadata": {
                "rationale": "JWT is stateless and works well with our microservices architecture"
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_accessed": datetime.now(timezone.utc).isoformat(),
        },
    )
    store.put(
        ("*", "episodic"),
        "e1",
        {
            "content": "User reported login page crashes on submit. TypeError in validateForm().",
            "memory_type": "episodic",
            "salience": {"composite": 0.75},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_accessed": datetime.now(timezone.utc).isoformat(),
        },
    )

    # --- 2. Active Context ---
    say("CONTEXT", "Building active context from store...")
    context_window = await ctx_builder.build_context(
        current_state={},
        thread_id="demo-1",
        goals=["Fix login page crash on submit button"],
    )

    ng = len(context_window.goals)
    nd = len(context_window.decisions)
    ne = len(context_window.episodes)
    say("CONTEXT", f"Retrieved: {ng} goals, {nd} decisions, {ne} episodes")

    context_text = ""
    for g in context_window.goals:
        context_text += f"[Goal] {g.value.get('content', '')}\n"
    for d in context_window.decisions:
        context_text += f"[Decision] {d.value.get('content', '')}\n"
    for e in context_window.episodes:
        context_text += f"[Episode] {e.value.get('content', '')}\n"

    say("CONTEXT", f"Context text fed to LLM:\n{context_text}")

    # --- 3. LLM Agent ---
    prompt = (
        "Context from memory:\n"
        + context_text
        + "\n"
        + "Given the context above, decide how to fix the login page crash. "
        + "State your decision clearly starting with 'Decision:' and provide your rationale."
    )

    say("LLM", f"Querying {MODEL}...")
    llm_output = _llm(prompt)
    say("LLM", f"Response ({len(llm_output)} chars):\n{llm_output}")

    # --- 4. Distillation ---
    state = {
        "input": "fix login page crash",
        "output": llm_output,
        "messages": [
            {"role": "user", "content": "fix login page crash"},
            {"role": "assistant", "content": llm_output},
        ],
        "goals": ["Fix login page crash on submit button"],
        "metadata": [],
        "thread_id": "demo-1",
    }

    say("DISTILL", "Running distillation pipeline...")
    distilled = await distill_pipeline.distill(thread_id="demo-1", state=state)
    say("DISTILL", f"Decisions extracted: {len(distilled.decisions)}")
    for i, dec in enumerate(distilled.decisions):
        say("DISTILL", f"  Decision {i+1}: {dec.get('decision', '')[:100]}")
    say("DISTILL", f"Unresolved goals: {len(distilled.unresolved_goals)}")
    say("DISTILL", f"Has compressed trace: {distilled.compressed_trace is not None}")
    if distilled.compressed_trace:
        say("DISTILL", f"  Trace: {distilled.compressed_trace[:200]}")

    # --- 5. Salience scoring & storage ---
    say("STORE", "Scoring and storing memories...")
    for raw in distilled.memory_items:
        existing = []
        for item in store.search(("*", raw.memory_type.value)):
            existing.append(
                MemoryItem(key=item.key, namespace=item.namespace, value=dict(item.value))
            )
        scored = await salience_engine.score(raw, existing)
        ns = ("*", raw.memory_type.value)
        store.put(
            ns,
            raw.memory_type.value,
            {
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
            },
        )
        say(
            "STORE",
            f"  Stored {raw.memory_type.value}:"
            f" salience={scored.composite:.3f}, content={raw.content[:60]}...",
        )

    # --- 6. Cognitive state reconstruction ---
    say("RECONSTRUCT", "Rebuilding cognitive state...")
    cognitive = await reconstruction.reconstruct(thread_id="demo-1", query="login crash")

    say("RECONSTRUCT", f"Active goals ({len(cognitive.active_goals)}):")
    for g in cognitive.active_goals:
        say("RECONSTRUCT", f"  - {g['goal']} (salience={g['salience']}, status={g['status']})")

    say("RECONSTRUCT", f"Relevant decisions ({len(cognitive.relevant_decisions)}):")
    for d in cognitive.relevant_decisions:
        say("RECONSTRUCT", f"  - {d['decision'][:80]} (salience={d['salience']})")

    say("RECONSTRUCT", f"Unresolved tasks ({len(cognitive.unresolved_tasks)}):")
    for t in cognitive.unresolved_tasks:
        say("RECONSTRUCT", f"  - {t['task'][:80]} (status={t['status']})")

    say("RECONSTRUCT", f"Reconstruction confidence: {cognitive.reconstruction_confidence}")

    say("DONE", "Full pipeline completed successfully")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
