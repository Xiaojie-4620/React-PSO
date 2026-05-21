"""Integration tests for Phase 1 new modules: llm_react, intervention_policy, config, experiments.

Uses a MockLLM to avoid real API calls.
"""

import json
from pathlib import Path
import sys

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


# ---------------------------------------------------------------------------
# Mock LLM for testing ReAct controller without API calls
# ---------------------------------------------------------------------------

class MockLLM:
    """A fake LLM that returns pre-scripted JSON responses for ReAct testing."""

    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0
        self.last_messages = None

    def getResponse(self, message):
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
        else:
            resp = '{"thought": "No intervention needed.", "action": "none", "params": {}, "done": true}'
        self.call_count += 1
        self.last_messages = message
        return resp

    def chat(self, messages, tools=None, tool_choice=None):
        return {"content": self.getResponse(str(messages))}


# ---------------------------------------------------------------------------
# Test: PSOConfig
# ---------------------------------------------------------------------------

def test_config_presets():
    from LLM4PSO.config import PSOConfig

    c10 = PSOConfig.for_cec2017(dim=10)
    assert c10.dim == 10
    assert c10.max_iter == 100_000
    assert c10.stagnation_threshold == 100
    assert c10.flag == "cec"

    c30 = PSOConfig.for_cec2017(dim=30)
    assert c30.stagnation_threshold == 300
    assert c30.max_iter == 300_000

    qt = PSOConfig.for_quick_test()
    assert qt.max_iter == 50
    assert qt.pop_size == 20

    d = c10.to_dict()
    assert d["dim"] == 10
    assert "w" in d


# ---------------------------------------------------------------------------
# Test: InterventionScheduler
# ---------------------------------------------------------------------------

def test_scheduler_budget():
    from LLM4PSO.intervention_policy import InterventionScheduler, PolicyConfig

    config = PolicyConfig(budget=3, use_cooldown=False, use_escalation=False, use_confidence_gate=False)
    scheduler = InterventionScheduler(config)

    class FakeState:
        state_label = "normal_search"

    # First 3 calls should be allowed
    assert scheduler.should_call_llm(FakeState(), 0)
    scheduler.record_call(0)
    assert scheduler.should_call_llm(FakeState(), 1)
    scheduler.record_call(1)
    assert scheduler.should_call_llm(FakeState(), 2)
    scheduler.record_call(2)
    # 4th should be blocked by budget
    assert not scheduler.should_call_llm(FakeState(), 3)
    assert scheduler.stats["call_count"] == 3


def test_scheduler_cooldown():
    from LLM4PSO.intervention_policy import InterventionScheduler, PolicyConfig

    config = PolicyConfig(
        cooldown_iters=10, use_cooldown=True,
        use_escalation=False, use_budget=True, use_confidence_gate=False,
    )
    scheduler = InterventionScheduler(config)

    class FakeState:
        state_label = "normal_search"

    assert scheduler.should_call_llm(FakeState(), 0)
    scheduler.record_call(0)
    assert not scheduler.should_call_llm(FakeState(), 5)   # cooldown
    assert scheduler.should_call_llm(FakeState(), 15)        # cooldown passed


def test_scheduler_confidence_gate():
    from LLM4PSO.intervention_policy import InterventionScheduler, PolicyConfig

    config = PolicyConfig(
        confidence_gate=True, use_confidence_gate=True,
        use_cooldown=False, use_escalation=False, use_budget=True,
    )
    scheduler = InterventionScheduler(config)

    class PrematureState:
        state_label = "premature_convergence"

    class MultimodalState:
        state_label = "multimodal_trap"

    # Clear-cut cases (rule controller handles well) → gated
    assert not scheduler.should_call_llm(PrematureState(), 0)
    # Ambiguous cases (LLM adds value) → allowed
    assert scheduler.should_call_llm(MultimodalState(), 0)


# ---------------------------------------------------------------------------
# Test: LLMReActController (with MockLLM)
# ---------------------------------------------------------------------------

def test_react_controller_single_turn_done():
    from LLM4PSO.llm_react import LLMReActController
    from LLM4PSO.actions import StrategyToolbox
    from LLM4PSO.compare.my_pso.Pso import PSO
    from LLM4PSO.compare.my_pso.function import Func

    np.random.seed(42)
    dim = 5
    lower = np.full(dim, -100.0)
    upper = np.full(dim, 100.0)
    vel = 0.2 * (upper - lower)

    pso = PSO(Func(), 10, dim, 1.5, 1.5, [lower, upper], [-vel, vel], "else", 0.9, 0.99)
    pso.pop_init()
    pso.evaluation("else")
    pso.update_p_best_cost()
    pso.update_g_best_cost()

    toolbox = StrategyToolbox([lower, upper], [-vel, vel], dim)

    # Mock LLM responds with gaussian_mutation then done
    mock = MockLLM(responses=[
        json.dumps({
            "thought": "Velocity has collapsed, need local perturbation.",
            "action": "gaussian_mutation",
            "params": {"ratio": 0.3, "scale": 0.05, "target": "worst"},
            "done": True,
        }),
    ])

    from LLM4PSO.state import SwarmStateAnalyzer
    analyzer = SwarmStateAnalyzer([lower, upper], [-vel, vel], dim=dim)
    state = analyzer.analyze(pso, 100, [pso.get_gBest()], 101)

    controller = LLMReActController(mock, toolbox, max_turns=3, verbose=False)
    result = controller.decide_and_act(pso, state, 100, [pso.get_gBest()])

    assert result.applied
    assert len(result.turns) == 1
    assert result.turns[0].action == "gaussian_mutation"
    assert result.turns[0].done


def test_react_controller_multi_turn():
    from LLM4PSO.llm_react import LLMReActController
    from LLM4PSO.actions import StrategyToolbox
    from LLM4PSO.compare.my_pso.Pso import PSO
    from LLM4PSO.compare.my_pso.function import Func

    np.random.seed(99)
    dim = 5
    lower = np.full(dim, -100.0)
    upper = np.full(dim, 100.0)
    vel = 0.2 * (upper - lower)

    pso = PSO(Func(), 20, dim, 1.5, 1.5, [lower, upper], [-vel, vel], "else", 0.9, 0.99)
    pso.pop_init()
    pso.evaluation("else")
    pso.update_p_best_cost()
    pso.update_g_best_cost()

    toolbox = StrategyToolbox([lower, upper], [-vel, vel], dim)

    # Two-turn conversation
    mock = MockLLM(responses=[
        json.dumps({
            "thought": "Low diversity, reset worst particles first.",
            "action": "reset_worst_particles",
            "params": {"ratio": 0.25, "reset_velocity": True},
            "done": False,
        }),
        json.dumps({
            "thought": "Good reset, now adjust inertia to explore.",
            "action": "adjust_parameters",
            "params": {"inertia_weight": 1.2},
            "done": True,
        }),
    ])

    from LLM4PSO.state import SwarmStateAnalyzer
    analyzer = SwarmStateAnalyzer([lower, upper], [-vel, vel], dim=dim)
    state = analyzer.analyze(pso, 50, [pso.get_gBest()], 51)

    controller = LLMReActController(mock, toolbox, max_turns=3, verbose=False)
    result = controller.decide_and_act(pso, state, 50, [pso.get_gBest()])

    assert result.applied
    assert len(result.turns) == 2
    assert result.turns[0].action == "reset_worst_particles"
    assert not result.turns[0].done
    assert result.turns[1].action == "adjust_parameters"
    assert result.turns[1].done
    assert result.final_inertia_weight == 1.2


def test_react_controller_parse_fallback():
    from LLM4PSO.llm_react import LLMReActController
    from LLM4PSO.actions import StrategyToolbox
    from LLM4PSO.compare.my_pso.Pso import PSO
    from LLM4PSO.compare.my_pso.function import Func

    np.random.seed(7)
    dim = 3
    lower = np.full(dim, -100.0)
    upper = np.full(dim, 100.0)
    vel = 0.2 * (upper - lower)

    pso = PSO(Func(), 8, dim, 1.5, 1.5, [lower, upper], [-vel, vel], "else", 0.9, 0.99)
    pso.pop_init()
    pso.evaluation("else")
    pso.update_p_best_cost()
    pso.update_g_best_cost()

    toolbox = StrategyToolbox([lower, upper], [-vel, vel], dim)

    # LLM returns markdown-wrapped JSON (common in practice)
    mock = MockLLM(responses=[
        '```json\n{"thought": "Swarm looks fine.", "action": "none", "params": {}, "done": true}\n```',
    ])

    from LLM4PSO.state import SwarmStateAnalyzer
    analyzer = SwarmStateAnalyzer([lower, upper], [-vel, vel], dim=dim)
    state = analyzer.analyze(pso, 10, [pso.get_gBest()], 0)

    controller = LLMReActController(mock, toolbox, max_turns=3, verbose=False)
    result = controller.decide_and_act(pso, state, 10, [pso.get_gBest()])

    assert not result.applied
    assert len(result.turns) == 1
    assert result.turns[0].action == "none"
    assert result.turns[0].done


# ---------------------------------------------------------------------------
# Test: LLM4PSO with llm_react mode (mock LLM, no real API)
# ---------------------------------------------------------------------------

def test_llm4pso_react_mode_with_mock():
    from LLM4PSO.LLMs4PSO import LLM4PSO
    from LLM4PSO.llm_react import LLMReActController
    from LLM4PSO.compare.my_pso.function import Func

    np.random.seed(123)
    dim = 5
    iterations = 15
    lower = np.full(dim, -100.0)
    upper = np.full(dim, 100.0)
    velocity_span = 0.2 * (upper - lower)

    # Stagnation quickly by disabling velocity (w=0, c1=0, c2=0)
    optimizer = LLM4PSO(
        dim=dim,
        flag="else",
        func=Func(),
        w=0.0,
        pop_size=10,
        iterations=iterations,
        wdamp=1.0,
        c1=0.0,
        c2=0.0,
        position_bounds=[lower, upper],
        velocity_bounds=[-velocity_span, velocity_span],
        stagnation_threshold=2,
        intervention_mode="llm_react",
    )

    # Replace the react controller's LLM with a mock
    mock = MockLLM(responses=[
        json.dumps({
            "thought": "Premature convergence detected, resetting worst particles.",
            "action": "reset_worst_particles",
            "params": {"ratio": 0.3, "reset_velocity": True, "inertia_weight": 1.1},
            "done": True,
        }),
    ] * 20)  # enough for multiple interventions
    optimizer._react_controller.llm = mock

    history = optimizer.run()

    assert history.shape == (iterations,)
    assert np.all(np.isfinite(history))
    assert optimizer.action_history
    react_actions = [a for a in optimizer.action_history if a["mode"] == "llm_react"]
    assert len(react_actions) > 0


# ---------------------------------------------------------------------------
# Test: New module imports work correctly
# ---------------------------------------------------------------------------

def test_imports():
    from LLM4PSO.llm_react import LLMReActController, ReActTurn, ReActResult
    from LLM4PSO.intervention_policy import InterventionScheduler, PolicyConfig
    from LLM4PSO.config import PSOConfig
    from llms.base import BaseLLM, LLMError
    from llms.factory import create_llm
    from experiments.runner import ExperimentRunner, TrialResult
    from experiments.metrics import convergence_auc, final_statistics, wilcoxon_test
    from experiments.configs import ExperimentConfig

    assert ReActTurn is not None
    assert InterventionScheduler is not None
    assert PSOConfig is not None
    assert BaseLLM is not None


# ---------------------------------------------------------------------------
# Test: Prompt template contains required placeholders
# ---------------------------------------------------------------------------

def test_prompt_template():
    from llms.prompt import PROMPT_REACT_TOOLS

    assert "{tools_json}" in PROMPT_REACT_TOOLS
    assert "{func_description}" in PROMPT_REACT_TOOLS
    assert "{dim}" in PROMPT_REACT_TOOLS
    assert "thought" in PROMPT_REACT_TOOLS
    assert "action" in PROMPT_REACT_TOOLS


if __name__ == "__main__":
    test_config_presets()
    print("[PASS] test_config_presets")

    test_scheduler_budget()
    print("[PASS] test_scheduler_budget")

    test_scheduler_cooldown()
    print("[PASS] test_scheduler_cooldown")

    test_scheduler_confidence_gate()
    print("[PASS] test_scheduler_confidence_gate")

    test_react_controller_single_turn_done()
    print("[PASS] test_react_controller_single_turn_done")

    test_react_controller_multi_turn()
    print("[PASS] test_react_controller_multi_turn")

    test_react_controller_parse_fallback()
    print("[PASS] test_react_controller_parse_fallback")

    test_llm4pso_react_mode_with_mock()
    print("[PASS] test_llm4pso_react_mode_with_mock")

    test_imports()
    print("[PASS] test_imports")

    test_prompt_template()
    print("[PASS] test_prompt_template")

    print("\nAll Phase 1 tests passed!")
