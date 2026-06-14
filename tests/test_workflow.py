from __future__ import annotations

from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

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


async def test_workflow_interrupts_for_human_review():
    store = InMemoryStore()
    retrieval = HybridRetrievalSystem(store)
    salience = MemorySalienceEngine()
    context_builder = ActiveContextBuilder(retrieval)
    distillation = AsyncDistillationPipeline(salience)

    def mock_predict(user_input: str, context: str | None) -> str:
        return "I decided to refactor the auth module."

    graph = build_workflow(
        retrieval=retrieval,
        context_builder=context_builder,
        salience=salience,
        distillation=distillation,
        predict=mock_predict,
        enable_review=True,
    )
    app = compile_workflow(graph, store=store)

    config = {"configurable": {"thread_id": "review-test-1"}}

    first = await app.ainvoke(
        {
            "input": "fix login bug",
            "thread_id": "review-test-1",
            "goals": ["fix login bug"],
            "messages": [],
            "metadata": [],
            "context": None,
            "output": None,
            "context_window": None,
            "distilled": None,
            "decisions": [],
        },
        config,
    )

    assert "__interrupt__" in first

    interrupt_val = first["__interrupt__"][0].value
    assert interrupt_val["action"] == "review_agent_output"
    assert "refactor" in interrupt_val["output"]

    resumed = await app.ainvoke(
        Command(resume={"approved": True}),
        config,
    )

    assert resumed.get("output") == "I decided to refactor the auth module."
    assert len(resumed.get("messages", [])) >= 1


async def test_workflow_discards_on_review_reject():
    store = InMemoryStore()
    retrieval = HybridRetrievalSystem(store)
    salience = MemorySalienceEngine()
    context_builder = ActiveContextBuilder(retrieval)
    distillation = AsyncDistillationPipeline(salience)

    def mock_predict(user_input: str, context: str | None) -> str:
        return "This is wrong."

    graph = build_workflow(
        retrieval=retrieval,
        context_builder=context_builder,
        salience=salience,
        distillation=distillation,
        predict=mock_predict,
        enable_review=True,
    )
    app = compile_workflow(graph, store=store)

    config = {"configurable": {"thread_id": "review-test-2"}}

    first = await app.ainvoke(
        {
            "input": "fix login bug",
            "thread_id": "review-test-2",
            "goals": ["fix login bug"],
            "messages": [],
            "metadata": [],
            "context": None,
            "output": None,
            "context_window": None,
            "distilled": None,
            "decisions": [],
        },
        config,
    )

    assert "__interrupt__" in first

    resumed = await app.ainvoke(
        Command(resume={"approved": False}),
        config,
    )

    assert resumed.get("distilled") is None
    assert "__interrupt__" not in resumed


async def test_workflow_accepts_edited_output():
    store = InMemoryStore()
    retrieval = HybridRetrievalSystem(store)
    salience = MemorySalienceEngine()
    context_builder = ActiveContextBuilder(retrieval)
    distillation = AsyncDistillationPipeline(salience)

    def mock_predict(user_input: str, context: str | None) -> str:
        return "Original response."

    graph = build_workflow(
        retrieval=retrieval,
        context_builder=context_builder,
        salience=salience,
        distillation=distillation,
        predict=mock_predict,
        enable_review=True,
    )
    app = compile_workflow(graph, store=store)

    config = {"configurable": {"thread_id": "review-test-3"}}

    first = await app.ainvoke(
        {
            "input": "fix login bug",
            "thread_id": "review-test-3",
            "goals": ["fix login bug"],
            "messages": [],
            "metadata": [],
            "context": None,
            "output": None,
            "context_window": None,
            "distilled": None,
            "decisions": [],
        },
        config,
    )

    assert "__interrupt__" in first

    resumed = await app.ainvoke(
        Command(resume={"approved": True, "edited_output": "Edited: use JWT tokens."}),
        config,
    )

    assert resumed.get("output") == "Edited: use JWT tokens."
    assert any(
        "Edited: use JWT tokens." in m.get("content", "")
        for m in resumed.get("messages", [])
    )
