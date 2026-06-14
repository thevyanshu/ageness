from __future__ import annotations

from langgraph.store.memory import InMemoryStore

from ageness.cognition.active_context.builder import ActiveContextBuilder
from ageness.cognition.distillation.pipeline import AsyncDistillationPipeline
from ageness.cognition.retrieval.hybrid import HybridRetrievalSystem
from ageness.cognition.salience.engine import MemorySalienceEngine
from ageness.graph.workflow import build_workflow, compile_workflow


async def test_workflow_builds_and_compiles():
    graph = build_workflow()
    app = compile_workflow(graph)
    assert app is not None


async def test_workflow_runs_end_to_end():
    store = InMemoryStore()
    retrieval = HybridRetrievalSystem(store)
    salience = MemorySalienceEngine()
    context_builder = ActiveContextBuilder(retrieval)
    distillation = AsyncDistillationPipeline(salience)

    graph = build_workflow(
        retrieval=retrieval,
        context_builder=context_builder,
        salience=salience,
        distillation=distillation,
    )
    app = compile_workflow(graph, store=store)

    result = await app.ainvoke(
        {
            "input": "fix login bug",
            "thread_id": "test-1",
            "goals": ["fix login bug"],
            "messages": [],
            "metadata": [],
            "context": None,
            "output": None,
            "context_window": None,
            "distilled": None,
            "decisions": [],
        },
        {"configurable": {"thread_id": "test-1"}},
    )

    assert result.get("output") is not None
    assert "fix login bug" in result["output"]


async def test_workflow_stores_memories_in_store():
    store = InMemoryStore()
    retrieval = HybridRetrievalSystem(store)
    salience = MemorySalienceEngine()
    context_builder = ActiveContextBuilder(retrieval)
    distillation = AsyncDistillationPipeline(salience)

    def mock_predict(user_input: str, context: str | None) -> str:
        return "I decided to fix the bug. Decision: use two-factor auth."

    graph = build_workflow(
        retrieval=retrieval,
        context_builder=context_builder,
        salience=salience,
        distillation=distillation,
        predict=mock_predict,
    )
    app = compile_workflow(graph, store=store)

    await app.ainvoke(
        {
            "input": "fix login bug",
            "thread_id": "test-2",
            "goals": ["fix login bug"],
            "messages": [],
            "metadata": [],
            "context": None,
            "output": None,
            "context_window": None,
            "distilled": None,
            "decisions": [],
        },
        {"configurable": {"thread_id": "test-2"}},
    )

    results = list(store.search(("*", "episodic")))
    assert len(results) > 0

    decision_results = list(store.search(("*", "decision")))
    assert len(decision_results) > 0
