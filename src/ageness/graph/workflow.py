from __future__ import annotations

import operator
from typing import Annotated, Any, Callable

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore
from typing_extensions import TypedDict

from ageness.cognition.active_context.builder import ActiveContextBuilder
from ageness.cognition.distillation.pipeline import AsyncDistillationPipeline
from ageness.cognition.retrieval.hybrid import HybridRetrievalSystem
from ageness.cognition.salience.engine import MemorySalienceEngine
from ageness.memory.models import ContextWindow, DistillationResult, MemoryItem


class AgentState(TypedDict):
    input: str
    context: str | None
    output: str | None
    thread_id: str
    metadata: Annotated[list[dict[str, Any]], operator.add]
    goals: list[str]
    messages: list[dict[str, Any]]
    distilled: DistillationResult | None
    context_window: ContextWindow | None


async def retrieve_context_node(
    state: AgentState,
    retrieval: HybridRetrievalSystem,
    context_builder: ActiveContextBuilder,
) -> dict[str, Any]:
    goals = state.get("goals") or []
    if not goals and state.get("input"):
        goals = [state["input"]]

    context_window = await context_builder.build_context(
        current_state=dict(state),
        thread_id=state.get("thread_id", ""),
        goals=goals,
    )

    context_parts: list[str] = []
    for item in context_window.goals:
        context_parts.append(f"[Goal] {item.value.get('content', '')}")
    for item in context_window.decisions:
        context_parts.append(f"[Decision] {item.value.get('content', '')}")
    for item in context_window.episodes:
        context_parts.append(f"[Episode] {item.value.get('content', '')}")
    for item in context_window.facts:
        context_parts.append(f"[Fact] {item.value.get('content', '')}")

    return {
        "context": "\n".join(context_parts) if context_parts else None,
        "context_window": context_window,
    }


async def agent_step_node(
    state: AgentState,
    predict: Callable[[str, str | None], str] | None = None,
) -> dict[str, Any]:
    user_input = state.get("input", "")
    context = state.get("context")

    if predict is not None:
        output = predict(user_input, context)
    else:
        output = f"Processed: {user_input}"

    new_messages: list[dict[str, Any]] = list(state.get("messages", []))
    new_messages.append({"role": "assistant", "content": output})

    return {"output": output, "messages": new_messages}


async def distill_node(
    state: AgentState,
    pipeline: AsyncDistillationPipeline,
) -> dict[str, Any]:
    thread_id = state.get("thread_id", "")
    agent_state_dict: dict[str, Any] = dict(state)

    result = await pipeline.distill(
        thread_id=thread_id,
        state=agent_state_dict,
    )

    new_goals: list[str] = []
    existing_goals = set(g.lower() for g in (state.get("goals") or []))
    for g in result.unresolved_goals:
        goal_text = g.get("goal", "")
        if goal_text and goal_text.lower() not in existing_goals:
            new_goals.append(goal_text)

    new_decisions: list[dict[str, Any]] = []
    existing_decisions: set[str] = set()
    for d in state.get("decisions", []):
        key = str(d.get("decision", ""))[:80]
        existing_decisions.add(key)
    for d in result.decisions:
        key = str(d.get("decision", ""))[:80]
        if key not in existing_decisions:
            new_decisions.append(d)

    return {
        "distilled": result,
        "goals": (state.get("goals") or []) + new_goals,
        "decisions": (state.get("decisions", [])) + new_decisions,
    }


async def store_memories_node(
    state: AgentState,
    store: InMemoryStore,
    salience: MemorySalienceEngine,
) -> dict[str, Any]:
    distilled = state.get("distilled")
    if not distilled or not distilled.memory_items:
        return {}

    stored_count = 0
    for raw_memory in distilled.memory_items:
        existing_items: list[MemoryItem] = []
        for item in store.search(("*", raw_memory.memory_type.value)):
            existing_items.append(
                MemoryItem(
                    key=item.key,
                    namespace=item.namespace,
                    value=dict(item.value),
                )
            )

        scored = await salience.score(raw_memory, existing_items)

        ns = ("*", raw_memory.memory_type.value)
        payload = {
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
            "source_checkpoint_id": raw_memory.source_checkpoint_id,
            "created_at": raw_memory.timestamp.isoformat(),
            "last_accessed": raw_memory.timestamp.isoformat(),
        }
        store.put(ns, raw_memory.memory_type.value, payload)
        stored_count += 1

    decayed = await salience.decay(store, ("*", "episodic"))
    consolidated = await salience.consolidate(store, ("*", "episodic"))

    return {
        "metadata": [
            {
                "stored_memories": stored_count,
                "decayed": decayed,
                "consolidated": consolidated,
            }
        ],
    }


def build_workflow(
    *,
    retrieval: HybridRetrievalSystem | None = None,
    context_builder: ActiveContextBuilder | None = None,
    salience: MemorySalienceEngine | None = None,
    distillation: AsyncDistillationPipeline | None = None,
    predict: Callable[[str, str | None], str] | None = None,
) -> StateGraph:
    retrieval = retrieval or HybridRetrievalSystem(InMemoryStore())
    salience_engine = salience or MemorySalienceEngine()
    distill_pipeline = distillation or AsyncDistillationPipeline(salience_engine)
    ctx_builder = context_builder or ActiveContextBuilder(retrieval)

    builder = StateGraph(AgentState)

    async def retrieve_wrapper(state: AgentState) -> dict[str, Any]:
        return await retrieve_context_node(state, retrieval, ctx_builder)

    async def agent_wrapper(state: AgentState) -> dict[str, Any]:
        return await agent_step_node(state, predict)

    async def distill_wrapper(state: AgentState) -> dict[str, Any]:
        return await distill_node(state, distill_pipeline)

    async def store_wrapper(state: AgentState) -> dict[str, Any]:
        return await store_memories_node(state, retrieval.store, salience_engine)

    builder.add_node("retrieve_context", retrieve_wrapper)
    builder.add_node("agent_step", agent_wrapper)
    builder.add_node("distill", distill_wrapper)
    builder.add_node("store_memories", store_wrapper)

    builder.add_edge(START, "retrieve_context")
    builder.add_edge("retrieve_context", "agent_step")
    builder.add_edge("agent_step", "distill")
    builder.add_edge("distill", "store_memories")
    builder.add_edge("store_memories", END)

    return builder


def compile_workflow(
    graph: StateGraph,
    checkpointer: Any | None = None,
    store: Any | None = None,
) -> Any:
    return graph.compile(
        checkpointer=checkpointer or InMemorySaver(),
        store=store or InMemoryStore(),
    )
