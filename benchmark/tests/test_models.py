from __future__ import annotations

from benchmark.models import (
    BenchmarkConfig,
    BenchmarkRun,
    ContextTruncation,
    ConversationTurn,
    EvaluationScore,
    InjectedFact,
    MemoryArchitecture,
    MetricsSnapshot,
    Scenario,
)


def test_conversation_turn_creation():
    t = ConversationTurn(turn_id=1, role="user", content="hello")
    assert t.turn_id == 1
    assert t.role == "user"
    assert t.content == "hello"
    assert t.timestamp is not None


def test_injected_fact_creation():
    f = InjectedFact(
        fact_id="f1",
        turn_id=5,
        content="Use Redis",
        category="decision",
        expected_recall_turns=[10, 20],
    )
    assert f.fact_id == "f1"
    assert f.category == "decision"
    assert 20 in f.expected_recall_turns


def test_scenario_creation():
    turns = [
        ConversationTurn(turn_id=1, role="user", content="hello"),
        ConversationTurn(turn_id=2, role="assistant", content="hi"),
    ]
    facts = [InjectedFact(
        fact_id="f1", turn_id=1, content="fact", category="general"
    )]
    s = Scenario(
        scenario_id="test_001",
        title="Test",
        description="A test scenario",
        turns=turns,
        facts=facts,
        expected_decisions=["Use Redis"],
        expected_goals=["Scalability"],
    )
    assert len(s.turns) == 2
    assert len(s.facts) == 1
    assert s.expected_decisions == ["Use Redis"]


def test_metrics_snapshot_defaults():
    m = MetricsSnapshot(turn_id=1)
    assert m.input_tokens == 0
    assert m.total_latency_ms == 0.0
    assert m.hallucination_indicators == []


def test_benchmark_config_defaults():
    c = BenchmarkConfig(architecture=MemoryArchitecture.TRANSCRIPT_REPLAY)
    assert c.max_turns == 100
    assert c.last_n_messages == 20
    assert c.seed == 42


def test_context_truncation_enum():
    assert ContextTruncation.LAST_N.value == "last_n"
    assert ContextTruncation.ROLLING_SUMMARY.value == "rolling_summary"
    assert ContextTruncation.TOKEN_BUDGET.value == "token_budget"


def test_memory_architecture_enum():
    assert MemoryArchitecture.TRANSCRIPT_REPLAY.value == "transcript_replay"
    assert MemoryArchitecture.RECONSTRUCTED_STATE.value == "reconstructed_state"


def test_benchmark_run_aggregation():
    cfg = BenchmarkConfig(architecture=MemoryArchitecture.TRANSCRIPT_REPLAY)
    run = BenchmarkRun(run_id="r1", scenario_id="s1", config=cfg)
    m1 = MetricsSnapshot(
        turn_id=1, input_tokens=10, output_tokens=20, total_latency_ms=100.0,
    )
    m2 = MetricsSnapshot(
        turn_id=2, input_tokens=15, output_tokens=25, total_latency_ms=150.0,
    )
    run.metrics = [m1, m2]
    run.total_input_tokens = sum(m.input_tokens for m in run.metrics)
    run.total_output_tokens = sum(m.output_tokens for m in run.metrics)
    run.total_latency_ms = sum(m.total_latency_ms for m in run.metrics)
    assert run.total_input_tokens == 25
    assert run.total_output_tokens == 45
    assert run.total_latency_ms == 250.0


def test_evaluation_score_creation():
    s = EvaluationScore(name="test", value=0.85, description="A test score")
    assert s.name == "test"
    assert s.value == 0.85
    assert isinstance(s.metadata, dict)
