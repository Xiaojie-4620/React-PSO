"""Integration tests for Phase 2 (Memory) and Phase 3 (Landscape) modules."""

import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


# ---------------------------------------------------------------------------
# Phase 3.1: LandscapeAnalyzer
# ---------------------------------------------------------------------------

def sphere_func(x: np.ndarray) -> np.ndarray:
    """Vectorized Sphere function: f(x) = sum(x^2)."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return np.sum(x ** 2, axis=1)


def rastrigin_func(x: np.ndarray) -> np.ndarray:
    """Vectorized Rastrigin function."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return 10 * x.shape[1] + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x), axis=1)


def test_landscape_profile_sphere():
    from LLM4PSO.landscape import LandscapeAnalyzer, LandscapeProfile

    dim = 5
    analyzer = LandscapeAnalyzer(sphere_func, dim=dim, bounds=(-10, 10), n_samples=300)
    best_pos = np.ones(dim) * 2.0  # away from optimum at 0
    profile = analyzer.analyze(best_pos)

    assert isinstance(profile, LandscapeProfile)
    assert 0.0 <= profile.ruggedness <= 1.0
    assert 0.0 <= profile.information_content <= 1.0
    assert profile.gradient_magnitude_mean >= 0.0
    # Sphere is smooth → low ruggedness
    assert profile.ruggedness < 0.7
    # Sphere has reliable gradients
    assert profile.landscape_label in {
        "broad_valley", "moderate_hills", "flat_plain", "unclassified",
    }


def test_landscape_profile_rastrigin():
    from LLM4PSO.landscape import LandscapeAnalyzer

    dim = 5
    analyzer = LandscapeAnalyzer(rastrigin_func, dim=dim, bounds=(-5.12, 5.12), n_samples=200)
    best_pos = np.ones(dim) * 2.0
    profile = analyzer.analyze(best_pos)

    # Rastrigin is highly multimodal → high ruggedness, high information content
    assert profile.ruggedness > 0.3 or profile.information_content > 0.3
    # Basin radius should be small (many local optima)
    assert profile.estimated_basin_radius < 20.0
    assert profile.modality_estimate in {"multimodal_few", "multimodal_many"}


def test_landscape_cache():
    from LLM4PSO.landscape import LandscapeAnalyzer

    analyzer = LandscapeAnalyzer(sphere_func, dim=3, bounds=(-10, 10), n_samples=50)
    pos = np.array([1.0, 2.0, 3.0])
    p1 = analyzer.analyze(pos, gbest=10.0)
    p2 = analyzer.analyze(pos, gbest=10.0)
    # Same inputs → same profile (cached)
    assert p1.ruggedness == p2.ruggedness


def test_landscape_to_dict():
    from LLM4PSO.landscape import LandscapeAnalyzer

    analyzer = LandscapeAnalyzer(sphere_func, dim=2, bounds=(-5, 5), n_samples=30)
    profile = analyzer.analyze(np.array([1.0, 2.0]))
    d = profile.to_dict()
    assert "ruggedness" in d
    assert "landscape_label" in d
    assert isinstance(profile.summary(), str)


# ---------------------------------------------------------------------------
# Phase 3.2: Landscape-aware state classification
# ---------------------------------------------------------------------------

def test_landscape_aware_state_labels():
    from LLM4PSO.landscape import LandscapeProfile
    from LLM4PSO.state import SwarmStateAnalyzer
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

    analyzer = SwarmStateAnalyzer([lower, upper], [-vel, vel], dim=dim, stagnation_window=3)

    # Rugged plateau landscape profile
    lp = LandscapeProfile(
        ruggedness=0.8, information_content=0.7,
        gradient_magnitude_mean=0.03, gradient_magnitude_std=0.01,
        landscape_label="rugged_plateau",
    )
    state = analyzer.analyze(pso, 5, [pso.get_gBest()], 4, landscape=lp)
    assert state.state_label == "rugged_plateau_trap"

    # Deceptive basin
    lp2 = LandscapeProfile(
        ruggedness=0.4, information_content=0.3,
        gradient_magnitude_mean=0.1, gradient_magnitude_std=0.05,
        deceptiveness=0.7, landscape_label="deceptive_valley",
    )
    pso.position = np.zeros((10, dim))
    pso.velocity = np.zeros((10, dim))
    pso.evaluation("else")
    pso.update_p_best_cost()
    pso.update_g_best_cost()
    state2 = analyzer.analyze(pso, 6, [pso.get_gBest(), pso.get_gBest()], 5, landscape=lp2)
    assert state2.state_label == "deceptive_basin"


# ---------------------------------------------------------------------------
# Phase 3.3: Landscape-adaptive actions
# ---------------------------------------------------------------------------

def test_landscape_actions_in_toolbox():
    from LLM4PSO.actions import StrategyToolbox
    from LLM4PSO.compare.my_pso.Pso import PSO
    from LLM4PSO.compare.my_pso.function import Func

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
    actions = toolbox.actions

    # All 4 new landscape actions should be available
    assert "basin_hopping" in actions
    assert "gradient_descent_step" in actions
    assert "landscape_adaptive_mutation" in actions
    assert "landscape_adaptive_restart" in actions

    # Each should work without error
    for name in ["basin_hopping", "gradient_descent_step",
                 "landscape_adaptive_mutation", "landscape_adaptive_restart"]:
        old_pos = pso.position.copy()
        result = toolbox.apply(pso, name)
        assert result.applied
        assert result.changed_particles > 0


# ---------------------------------------------------------------------------
# Phase 2.1: InterventionMemory
# ---------------------------------------------------------------------------

def test_memory_record_and_query():
    from LLM4PSO.memory import InterventionMemory
    from LLM4PSO.state import SwarmStateAnalyzer
    from LLM4PSO.compare.my_pso.Pso import PSO
    from LLM4PSO.compare.my_pso.function import Func

    np.random.seed(7)
    dim = 5
    lower = np.full(dim, -100.0)
    upper = np.full(dim, 100.0)
    vel = 0.2 * (upper - lower)

    pso = PSO(Func(), 10, dim, 1.5, 1.5, [lower, upper], [-vel, vel], "else", 0.9, 0.99)
    pso.pop_init()
    pso.evaluation("else")
    pso.update_p_best_cost()
    pso.update_g_best_cost()

    analyzer = SwarmStateAnalyzer([lower, upper], [-vel, vel], dim=dim)
    state = analyzer.analyze(pso, 0, [pso.get_gBest()], 0)

    memory = InterventionMemory()

    # Record some interventions
    memory.record(state, "reset_worst_particles", {"ratio": 0.3}, improvement_delta=0.01,
                  function_name="sphere", dim=5, iteration=10)
    memory.record(state, "gaussian_mutation", {"ratio": 0.2}, improvement_delta=-0.001,
                  function_name="sphere", dim=5, iteration=50)
    memory.record(state, "levy_flight", {"ratio": 0.25}, improvement_delta=0.05,
                  function_name="rastrigin", dim=5, iteration=100)

    assert len(memory._entries) == 3
    assert memory._entries[0].success  # improved
    assert not memory._entries[1].success  # did not improve
    assert memory._entries[2].success

    # Query
    results = memory.query(state, k=2)
    assert 1 <= len(results) <= 2

    # Success rate
    rate = memory.get_success_rate("reset_worst_particles")
    assert rate == 1.0  # all reset_worst_particles entries succeeded

    # Best action for label
    best = memory.best_action_for_label("normal_search", min_samples=1)
    assert best is not None


def _make_fake_entry(state_label, action, success):
    from LLM4PSO.memory import MemoryEntry
    return MemoryEntry(
        state_label=state_label,
        state_features={"normalized_diversity": 0.01, "velocity_zero_ratio": 0.8},
        action=action,
        params={},
        improvement_delta=0.01 if success else -0.001,
        success=success,
        function_name="test_func",
        dim=10,
        iteration=50,
        timestamp="",
    )


def test_memory_persistence():
    from LLM4PSO.memory import InterventionMemory
    import tempfile, os, shutil

    memory = InterventionMemory()
    memory._entries = [
        _make_fake_entry("premature_convergence", "reset_worst_particles", True),
        _make_fake_entry("multimodal_trap", "levy_flight", False),
    ]

    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test_memory.json")

    memory.save(path)
    memory2 = InterventionMemory()
    memory2.load(path)
    assert len(memory2._entries) == 2

    shutil.rmtree(tmpdir, ignore_errors=True)


def test_memory_context_for_prompt():
    from LLM4PSO.memory import InterventionMemory
    from LLM4PSO.state import SwarmStateAnalyzer
    from LLM4PSO.compare.my_pso.Pso import PSO
    from LLM4PSO.compare.my_pso.function import Func

    np.random.seed(3)
    dim = 5
    lower = np.full(dim, -100.0)
    upper = np.full(dim, 100.0)
    vel = 0.2 * (upper - lower)

    pso = PSO(Func(), 10, dim, 1.5, 1.5, [lower, upper], [-vel, vel], "else", 0.9, 0.99)
    pso.pop_init()
    pso.evaluation("else")
    pso.update_p_best_cost()
    pso.update_g_best_cost()

    analyzer = SwarmStateAnalyzer([lower, upper], [-vel, vel], dim=dim)
    state = analyzer.analyze(pso, 0, [pso.get_gBest()], 0)

    memory = InterventionMemory()
    memory.record(state, "reset_worst_particles", {"ratio": 0.3}, improvement_delta=0.05,
                  function_name="sphere", dim=5, iteration=100)

    ctx = memory.build_memory_context(state, k=3)
    assert "Past similar intervention" in ctx
    assert "reset_worst_particles" in ctx


# ---------------------------------------------------------------------------
# Phase 2.3: AdaptiveRuleController
# ---------------------------------------------------------------------------

def test_adaptive_controller_default_behavior():
    from LLM4PSO.controller_adaptive import AdaptiveRuleController
    from LLM4PSO.state import SwarmStateAnalyzer
    from LLM4PSO.compare.my_pso.Pso import PSO
    from LLM4PSO.compare.my_pso.function import Func

    np.random.seed(13)
    dim = 4
    lower = np.full(dim, -100.0)
    upper = np.full(dim, 100.0)
    vel = 0.2 * (upper - lower)

    pso = PSO(Func(), 8, dim, 1.5, 1.5, [lower, upper], [-vel, vel], "else", 0.9, 0.99)
    pso.position = np.zeros((8, dim))
    pso.velocity = np.zeros((8, dim))
    pso.evaluation("else")
    pso.update_p_best_cost()
    pso.update_g_best_cost()

    analyzer = SwarmStateAnalyzer([lower, upper], [-vel, vel], dim=dim, stagnation_window=3)
    state = analyzer.analyze(pso, 4, [0.0, 0.0], 4)

    ctrl = AdaptiveRuleController()
    decision = ctrl.decide(state)

    # Without learning, should behave identically to RuleBasedReActController
    assert decision.action == "reset_worst_particles"
    assert state.state_label == "premature_convergence"


def test_adaptive_controller_learns():
    from LLM4PSO.controller_adaptive import AdaptiveRuleController

    ctrl = AdaptiveRuleController()
    # Simulate history where gaussian_mutation works better than reset_worst for PC
    history = [
        {
            "mode": "rule",
            "state": {"state_label": "premature_convergence"},
            "decision": {"action": "gaussian_mutation", "params": {"ratio": 0.3, "scale": 0.05}},
            "result": {"applied": True},
        },
        {
            "mode": "rule",
            "state": {"state_label": "premature_convergence"},
            "decision": {"action": "gaussian_mutation", "params": {"ratio": 0.3, "scale": 0.05}},
            "result": {"applied": True},
        },
        {
            "mode": "rule",
            "state": {"state_label": "premature_convergence"},
            "decision": {"action": "gaussian_mutation", "params": {"ratio": 0.3, "scale": 0.05}},
            "result": {"applied": True},
        },
        {
            "mode": "rule",
            "state": {"state_label": "premature_convergence"},
            "decision": {"action": "reset_worst_particles", "params": {"ratio": 0.2}},
            "result": {"applied": True},
        },
    ]
    ctrl.learn_from_history(history, min_samples=3)

    label = "premature_convergence"
    assert label in ctrl._custom_rules
    learned_action, thought, params = ctrl._custom_rules[label]
    assert learned_action == "gaussian_mutation"


def test_adaptive_controller_save_load():
    from LLM4PSO.controller_adaptive import AdaptiveRuleController
    import tempfile, os

    ctrl = AdaptiveRuleController()
    ctrl._custom_rules["premature_convergence"] = (
        "gaussian_mutation", "Learned best", {"ratio": 0.35},
    )

    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "policy.json")
    ctrl.save_policy(path)

    ctrl2 = AdaptiveRuleController()
    ctrl2.load_policy(path)
    assert "premature_convergence" in ctrl2._custom_rules

    import shutil
    shutil.rmtree(tmpdir)


# ---------------------------------------------------------------------------
# Phase 2.2: Memory-augmented ReAct prompt
# ---------------------------------------------------------------------------

def test_react_with_memory_prompt_template():
    from llms.prompt import PROMPT_REACT_WITH_MEMORY

    assert "{memory_context}" in PROMPT_REACT_WITH_MEMORY
    assert "{tools_json}" in PROMPT_REACT_WITH_MEMORY
    assert "Historical Experience" in PROMPT_REACT_WITH_MEMORY
    assert "landscape_adaptive_mutation" in PROMPT_REACT_WITH_MEMORY


def test_react_controller_with_memory():
    from LLM4PSO.llm_react import LLMReActController
    from LLM4PSO.memory import InterventionMemory
    from LLM4PSO.actions import StrategyToolbox
    from LLM4PSO.state import SwarmStateAnalyzer
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
    analyzer = SwarmStateAnalyzer([lower, upper], [-vel, vel], dim=dim)
    state = analyzer.analyze(pso, 0, [pso.get_gBest()], 0)
    memory = InterventionMemory()
    memory.record(state, "reset_worst_particles", {"ratio": 0.3}, improvement_delta=0.01,
                  function_name="sphere", dim=5, iteration=10)

    import json
    mock = type("MockLLM", (), {
        "call_count": 0,
        "getResponse": lambda self, msg: (
            setattr(self, "call_count", self.call_count + 1)
            or '{"thought": "No intervention.", "action": "none", "params": {}, "done": true}'
        ),
        "chat": lambda self, msg, **kw: {"content": self.getResponse(str(msg))},
    })()

    ctrl = LLMReActController(mock, toolbox, max_turns=3, memory=memory, verbose=False)
    result = ctrl.decide_and_act(pso, state, 10, [pso.get_gBest()])

    assert not result.applied
    assert len(result.turns) == 1
    assert result.turns[0].action == "none"


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Phase 3.1
    test_landscape_profile_sphere()
    print("[PASS] test_landscape_profile_sphere")
    test_landscape_profile_rastrigin()
    print("[PASS] test_landscape_profile_rastrigin")
    test_landscape_cache()
    print("[PASS] test_landscape_cache")
    test_landscape_to_dict()
    print("[PASS] test_landscape_to_dict")

    # Phase 3.2
    test_landscape_aware_state_labels()
    print("[PASS] test_landscape_aware_state_labels")

    # Phase 3.3
    test_landscape_actions_in_toolbox()
    print("[PASS] test_landscape_actions_in_toolbox")

    # Phase 2.1
    test_memory_record_and_query()
    print("[PASS] test_memory_record_and_query")
    test_memory_persistence()
    print("[PASS] test_memory_persistence")
    test_memory_context_for_prompt()
    print("[PASS] test_memory_context_for_prompt")

    # Phase 2.2
    test_react_with_memory_prompt_template()
    print("[PASS] test_react_with_memory_prompt_template")
    test_react_controller_with_memory()
    print("[PASS] test_react_controller_with_memory")

    # Phase 2.3
    test_adaptive_controller_default_behavior()
    print("[PASS] test_adaptive_controller_default_behavior")
    test_adaptive_controller_learns()
    print("[PASS] test_adaptive_controller_learns")
    test_adaptive_controller_save_load()
    print("[PASS] test_adaptive_controller_save_load")

    print("\nAll Phase 2+3 tests passed!")
