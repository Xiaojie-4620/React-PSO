from pathlib import Path
import sys

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from LLM4PSO.LLMs4PSO import LLM4PSO
from LLM4PSO.actions import StrategyToolbox
from LLM4PSO.controller import RuleBasedReActController
from LLM4PSO.state import SwarmStateAnalyzer
from LLM4PSO.compare.my_pso.Pso import PSO
from LLM4PSO.compare.my_pso.function import Func


def test_llm4pso_run_sphere_without_llm_call():
    np.random.seed(42)
    dim = 10
    iterations = 20
    lower = np.full(dim, -100.0)
    upper = np.full(dim, 100.0)
    velocity_span = 0.2 * (upper - lower)

    optimizer = LLM4PSO(
        dim=dim,
        flag="else",
        func=Func(),
        w=0.9,
        pop_size=30,
        iterations=iterations,
        wdamp=0.99,
        c1=1.5,
        c2=1.5,
        position_bounds=[lower, upper],
        velocity_bounds=[-velocity_span, velocity_span],
        stagnation_threshold=1000,
    )

    history = optimizer.run()

    assert history.shape == (iterations,)
    assert np.all(np.isfinite(history))
    assert np.all(np.diff(history) <= 1e-12)
    assert len(optimizer.state_history) == iterations


def test_pso_run_sphere():
    np.random.seed(42)
    dim = 10
    iterations = 20
    lower = np.full(dim, -100.0)
    upper = np.full(dim, 100.0)
    velocity_span = 0.2 * (upper - lower)

    optimizer = PSO(
        func=Func(),
        n_pop=30,
        dim=dim,
        c1=1.5,
        c2=1.5,
        position_bound=[lower, upper],
        velocity_bound=[-velocity_span, velocity_span],
        flag="else",
        w=0.9,
        wdamp=0.99,
    )

    best_pos, best_cost = optimizer.run(max_iter=iterations, verbose=False)

    assert best_pos.shape == (dim,)
    assert np.isfinite(best_cost)


def test_swarm_state_analyzer_reports_structured_state():
    np.random.seed(7)
    dim = 5
    lower = np.full(dim, -100.0)
    upper = np.full(dim, 100.0)
    velocity_span = 0.2 * (upper - lower)
    pso = PSO(
        func=Func(),
        n_pop=10,
        dim=dim,
        c1=1.5,
        c2=1.5,
        position_bound=[lower, upper],
        velocity_bound=[-velocity_span, velocity_span],
        flag="else",
        w=0.9,
        wdamp=0.99,
    )
    pso.pop_init()
    pso.evaluation("else")
    pso.update_p_best_cost()
    pso.update_g_best_cost()

    analyzer = SwarmStateAnalyzer([lower, upper], [-velocity_span, velocity_span], dim=dim)
    state = analyzer.analyze(pso, iteration=0, history=[pso.get_gBest()], no_improve_iters=0)
    # print("\n" + "="*30 + "\n")
    # print(state)
    # print("="*30)
    assert state.state_label in {
        "normal_search",
        "normal_convergence",
        "slow_stagnation",
        "premature_convergence",
        "velocity_collapse",
        "boundary_stagnation",
        "multimodal_trap",
    }
    assert state.to_prompt_dict()["iteration"] == 0
    assert state.normalized_diversity >= 0.0


def test_strategy_toolbox_applies_whitelisted_action():
    np.random.seed(11)
    dim = 5
    lower = np.full(dim, -100.0)
    upper = np.full(dim, 100.0)
    velocity_span = 0.2 * (upper - lower)
    pso = PSO(
        func=Func(),
        n_pop=10,
        dim=dim,
        c1=1.5,
        c2=1.5,
        position_bound=[lower, upper],
        velocity_bound=[-velocity_span, velocity_span],
        flag="else",
        w=0.9,
        wdamp=0.99,
    )
    pso.pop_init()
    pso.evaluation("else")
    pso.update_p_best_cost()
    pso.update_g_best_cost()
    old_position = pso.position.copy()

    toolbox = StrategyToolbox([lower, upper], [-velocity_span, velocity_span], dim=dim)
    result = toolbox.apply(pso, "reset_worst_particles", {"ratio": 0.3, "reset_velocity": True})

    assert result.applied
    assert result.changed_particles > 0
    assert not np.allclose(old_position, pso.position)
    assert np.all(pso.position >= lower)
    assert np.all(pso.position <= upper)


def test_rule_controller_selects_action_for_premature_convergence():
    np.random.seed(13)
    dim = 4
    lower = np.full(dim, -100.0)
    upper = np.full(dim, 100.0)
    velocity_span = 0.2 * (upper - lower)
    pso = PSO(
        func=Func(),
        n_pop=8,
        dim=dim,
        c1=1.5,
        c2=1.5,
        position_bound=[lower, upper],
        velocity_bound=[-velocity_span, velocity_span],
        flag="else",
        w=0.9,
        wdamp=0.99,
    )
    pso.position = np.zeros((8, dim))
    pso.velocity = np.zeros((8, dim))
    pso.evaluation("else")
    pso.update_p_best_cost()
    pso.update_g_best_cost()

    analyzer = SwarmStateAnalyzer([lower, upper], [-velocity_span, velocity_span], dim=dim, stagnation_window=3)
    state = analyzer.analyze(pso, iteration=4, history=[0.0, 0.0], no_improve_iters=4)
    decision = RuleBasedReActController().decide(state)

    assert state.state_label == "premature_convergence"
    assert decision.action == "reset_worst_particles"
    assert "diversity" in decision.thought.lower()


def test_llm4pso_rule_intervention_runs_without_llm_call():
    np.random.seed(17)
    dim = 5
    iterations = 12
    lower = np.full(dim, -100.0)
    upper = np.full(dim, 100.0)
    velocity_span = 0.2 * (upper - lower)

    optimizer = LLM4PSO(
        dim=dim,
        flag="else",
        func=Func(),
        w=0.0,
        pop_size=12,
        iterations=iterations,
        wdamp=1.0,
        c1=0.0,
        c2=0.0,
        position_bounds=[lower, upper],
        velocity_bounds=[-velocity_span, velocity_span],
        stagnation_threshold=2,
        intervention_mode="rule",
    )

    history = optimizer.run()

    assert history.shape == (iterations,)
    assert np.all(np.isfinite(history))
    assert optimizer.action_history
    assert all(item["mode"] == "rule" for item in optimizer.action_history)



if __name__ == "__main__":
    print(test_swarm_state_analyzer_reports_structured_state())
