"""Integration tests for Phase 4 (Advanced ReAct) and Phase 5 (Comparison)."""

import sys
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


def sphere(x):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return np.sum(x**2, axis=1)


# ---------------------------------------------------------------------------
# Phase 4.1: DeepReActController
# ---------------------------------------------------------------------------

def test_deep_react_controller():
    from LLM4PSO.react_deep import DeepReActController, DeepReActTurn, DEEP_REACT_SYSTEM_PROMPT
    from LLM4PSO.actions import StrategyToolbox
    from LLM4PSO.state import SwarmStateAnalyzer
    from LLM4PSO.compare.my_pso.Pso import PSO
    from LLM4PSO.compare.my_pso.function import Func
    import json

    np.random.seed(88)
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
    analyzer = SwarmStateAnalyzer([lower, upper], [-vel, vel], dim=dim)
    state = analyzer.analyze(pso, 50, [pso.get_gBest()], 51)

    mock = type("MockLLM", (), {
        "getResponse": lambda self, msg: json.dumps({
            "observe": "Diversity is 0.01, velocity near zero.",
            "diagnose": "Premature convergence, swarm collapsed to a local basin.",
            "strategize": "Reset worst particles to restore exploration.",
            "action": "reset_worst_particles",
            "params": {"ratio": 0.3, "inertia_weight": 1.1},
            "done": True,
        }),
        "chat": lambda self, msg, **kw: {"content": self.getResponse(str(msg))},
    })()

    ctrl = DeepReActController(mock, toolbox, max_turns=2, verbose=False,
                               landscape_prior="Test: unimodal, low ruggedness")
    result = ctrl.decide_and_act(pso, state, 50, [pso.get_gBest()])

    assert result.applied
    assert len(result.turns) == 1
    assert isinstance(result.turns[0], DeepReActTurn)
    turn = result.turns[0]
    assert "Diversity is 0.01" in turn.observe
    assert "Premature convergence" in turn.diagnose
    assert "Reset worst" in turn.strategize
    assert len(ctrl.diagnosis_history) >= 0


def test_deep_react_multi_turn():
    from LLM4PSO.react_deep import DeepReActController
    from LLM4PSO.actions import StrategyToolbox
    from LLM4PSO.state import SwarmStateAnalyzer
    from LLM4PSO.compare.my_pso.Pso import PSO
    from LLM4PSO.compare.my_pso.function import Func
    import json

    np.random.seed(77)
    dim = 4
    lower = np.full(dim, -100.0)
    upper = np.full(dim, 100.0)
    vel = 0.2 * (upper - lower)

    pso = PSO(Func(), 8, dim, 1.5, 1.5, [lower, upper], [-vel, vel], "else", 0.9, 0.99)
    pso.pop_init()
    pso.evaluation("else")
    pso.update_p_best_cost()
    pso.update_g_best_cost()

    toolbox = StrategyToolbox([lower, upper], [-vel, vel], dim)
    analyzer = SwarmStateAnalyzer([lower, upper], [-vel, vel], dim=dim, stagnation_window=3)
    state = analyzer.analyze(pso, 60, [pso.get_gBest(), pso.get_gBest()], 4)

    responses = iter([
        json.dumps({
            "observe": "Swarm stuck, low velocity.",
            "diagnose": "Velocity collapse after convergence.",
            "strategize": "First: reset particles to inject diversity.",
            "action": "reset_worst_particles",
            "params": {"ratio": 0.25},
            "done": False,
        }),
        json.dumps({
            "reflect": "Reset worked, now fine-tune parameters.",
            "revised_diagnose": "Diversity restored, adjust for exploitation.",
            "action": "adjust_parameters",
            "params": {"inertia_weight": 0.8},
            "done": True,
        }),
    ])

    mock = type("MockLLM", (), {
        "getResponse": lambda self, msg: next(responses),
        "chat": lambda self, msg, **kw: {"content": self.getResponse(str(msg))},
    })()

    ctrl = DeepReActController(mock, toolbox, max_turns=2, verbose=False)
    result = ctrl.decide_and_act(pso, state, 60, [pso.get_gBest()])

    assert result.applied
    assert len(result.turns) == 2


# ---------------------------------------------------------------------------
# Phase 4.3: Feature function-aware prompts
# ---------------------------------------------------------------------------

def test_feature_landscape_prior():
    from LLM4PSO.feature import Feature, CEC2017_META

    feat = Feature()
    feat.getAllFuncName()

    # Test known function
    prior = feat.get_landscape_prior("f1")
    assert "f1" in prior
    assert "unimodal" in prior
    assert "low" in prior
    assert "broad_valley" in prior

    # Test unknown function
    prior_unknown = feat.get_landscape_prior("unknown_func")
    assert "unknown" in prior_unknown.lower()

    # Test meta access
    meta = feat.get_function_meta("f7")
    assert meta["modality"] == "multimodal_many"
    assert meta["ruggedness"] == "high"

    assert "f1" in CEC2017_META
    assert "f30" in CEC2017_META


# ---------------------------------------------------------------------------
# Phase 5.4: Comparison algorithms
# ---------------------------------------------------------------------------

def test_clpso_runs():
    from LLM4PSO.compare.clpso import CLPSO

    np.random.seed(1)
    opt = CLPSO(sphere, n_pop=20, dim=10, refreshing_gap=3)
    history = opt.optimize(max_iter=30, verbose=False)

    assert len(history) == 31  # initial + 30 iter
    assert np.isfinite(history[-1])
    assert history[0] >= history[-1]  # monotonic improvement or plateau


def test_jde_runs():
    from LLM4PSO.compare.jde import JDE

    np.random.seed(2)
    opt = JDE(sphere, n_pop=20, dim=10)
    history = opt.optimize(max_iter=30, verbose=False)

    assert len(history) == 31
    assert np.isfinite(history[-1])
    assert history[0] >= history[-1]


def test_react_deep_import_in_llm4pso():
    """Verify DeepReActController is properly importable and extends correct base."""
    from LLM4PSO.react_deep import DeepReActController
    from LLM4PSO.llm_react import LLMReActController

    assert issubclass(DeepReActController, LLMReActController)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_deep_react_controller()
    print("[PASS] test_deep_react_controller")
    test_deep_react_multi_turn()
    print("[PASS] test_deep_react_multi_turn")
    test_feature_landscape_prior()
    print("[PASS] test_feature_landscape_prior")
    test_clpso_runs()
    print("[PASS] test_clpso_runs")
    test_jde_runs()
    print("[PASS] test_jde_runs")
    test_react_deep_import_in_llm4pso()
    print("[PASS] test_react_deep_import_in_llm4pso")

    print("\nAll Phase 4+5 tests passed!")
