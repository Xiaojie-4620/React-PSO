from pathlib import Path
import sys

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from LLM4PSO.LLMs4PSO import LLM4PSO
from LLM4PSO.state import SwarmStateAnalyzer
from my_pso.Pso import PSO
from my_pso.function import Func


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



if __name__ == "__main__":
    print(test_swarm_state_analyzer_reports_structured_state())

