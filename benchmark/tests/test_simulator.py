from __future__ import annotations

from benchmark.models import Scenario
from benchmark.simulator.generator import ConversationGenerator


def test_generator_creates_scenario():
    gen = ConversationGenerator(seed=42)
    scenario = gen.generate_architecture_planning()
    assert scenario.scenario_id == "arch_planning_001"
    assert len(scenario.turns) > 0
    assert len(scenario.facts) > 0
    assert len(scenario.expected_decisions) > 0
    assert len(scenario.expected_goals) > 0


def test_generator_architecture_planning_has_key_decisions():
    gen = ConversationGenerator(seed=42)
    scenario = gen.generate_architecture_planning()
    decisions = [d.lower() for d in scenario.expected_decisions]
    assert any("kafka" in d for d in decisions)
    assert any("saga" in d for d in decisions)
    assert any("domain-driven" in d for d in decisions)
    assert any("strangler" in d for d in decisions)


def test_generator_architecture_planning_has_query_pairs():
    gen = ConversationGenerator(seed=42)
    scenario = gen.generate_architecture_planning()
    assert len(scenario.query_pairs) > 0
    for qp in scenario.query_pairs:
        assert "turn" in qp
        assert "query" in qp
        assert "expected" in qp


def test_generator_debugging_session():
    gen = ConversationGenerator(seed=42)
    scenario = gen.generate_debugging_session(num_bugs=5)
    assert scenario.scenario_id == "debug_session_001"
    assert len(scenario.turns) >= 6
    assert len(scenario.facts) == 5


def test_generator_long_session():
    gen = ConversationGenerator(seed=42)
    scenario = gen.generate_long_session(num_turns=20)
    assert scenario.scenario_id == "long_session_20"
    assert len(scenario.turns) == 20


def test_generator_reset():
    gen = ConversationGenerator(seed=42)
    s1 = gen.generate_architecture_planning()
    gen.reset()
    s2 = gen.generate_architecture_planning()
    assert s1.scenario_id == s2.scenario_id
    assert len(s1.turns) == len(s2.turns)


def test_generator_returns_scenario_type():
    gen = ConversationGenerator(seed=42)
    scenario = gen.generate_architecture_planning()
    assert isinstance(scenario, Scenario)
