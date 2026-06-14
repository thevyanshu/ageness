from __future__ import annotations

import operator
from typing import Annotated, Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore
from typing_extensions import TypedDict


class AgentState(TypedDict):
    input: str
    context: dict[str, Any] | None
    output: str | None
    metadata: Annotated[list[dict[str, Any]], operator.add]


def build_workflow() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_edge(START, END)

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
