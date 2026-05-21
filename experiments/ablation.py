"""Automated ablation experiment runner.

Defines ablation matrices and runs them via ExperimentRunner.
Produces comparison tables and convergence plots for each ablation dimension.
"""

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from experiments.runner import ExperimentRunner, TrialResult
from experiments.configs import ExperimentConfig
from experiments.metrics import (
    convergence_auc, final_statistics, wilcoxon_test, friedman_ranking,
)
from experiments.reporting import generate_latex_table, plot_convergence_curves


def _get_pso_defaults(dim: int, mode: str) -> dict:
    """Get default PSO parameters for a configuration."""
    defaults = {
        "dim": dim,
        "flag": "else",
        "w": 0.9,
        "pop_size": max(20, min(50, dim * 2)),
        "iterations": max(50, dim * 100),
        "wdamp": 0.99,
        "c1": 1.5,
        "c2": 1.5,
        "stagnation_threshold": max(20, dim * 5),
        "improvement_tolerance": 1e-6,
        "intervention_mode": mode,
    }
    pos_bound = (-100.0, 100.0)
    vel_bound = (-20.0, 20.0)
    defaults["position_bounds"] = pos_bound
    defaults["velocity_bounds"] = vel_bound
    return defaults


def build_optimizer_factory_for_mode(mode: str):
    """Create a build_optimizer callable for a specific intervention mode.

    The returned callable has signature: (func, dim, mode_label, seed) -> optimizer.
    """
    from LLM4PSO.LLMs4PSO import LLM4PSO

    def builder(func, dim, mode_label, seed):
        np.random.seed(seed)
        kwargs = _get_pso_defaults(dim, mode)
        kwargs["func"] = func
        # "none" mode: disable interventions by setting infinite threshold
        if mode == "none":
            kwargs["intervention_mode"] = "rule"
            kwargs["stagnation_threshold"] = kwargs["iterations"] * 10
        return LLM4PSO(**kwargs)

    return builder


def run_ablation_intervention_modes(
    functions: List[tuple],
    dims: List[int] = None,
    n_trials: int = 30,
    output_dir: str = "./experiments/results/ablation_modes",
):
    """Ablation: compare intervention modes (none vs rule vs llm_react).

    This is the primary ablation experiment — it measures the contribution
    of each intervention layer to the overall optimization performance.
    """
    dims = dims or [10, 30]
    config = ExperimentConfig(
        name="ablation_modes",
        description="Ablation: intervention mode comparison",
        functions=functions,
        dimensions=dims,
        modes=["rule"],  # placeholdexr, builder handles variation
        n_trials=n_trials,
        output_dir=output_dir,
    )

    modes_to_test = ["none", "rule"]
    all_reports = []

    for mode_name in modes_to_test:
        print(f"\n{'='*50}\nRunning mode: {mode_name}\n{'='*50}")
        builder = build_optimizer_factory_for_mode(mode_name)
        config.modes = [mode_name]
        runner = ExperimentRunner(config, builder, n_workers=1, verbose=False)
        reports = runner.run()
        runner.save_results()
        all_reports.extend(reports)

    # Print summary
    print(f"\n{'='*50}")
    print("Ablation Results (mean final cost, lower is better):")
    print(f"{'='*50}")
    groups: Dict[tuple, Dict[str, ExperimentRunner.ExperimentReport]] = {}
    for r in all_reports:
        groups.setdefault((r.func_name, r.dim), {})[r.mode] = r

    for (func_name, dim), mode_map in sorted(groups.items()):
        print(f"\n  {func_name} (D={dim}):")
        for mode in modes_to_test:
            rep = mode_map.get(mode)
            if rep:
                print(f"    {mode:12s}: mean={rep.final_mean:.4e}  std={rep.final_std:.4e}  "
                      f"interventions={rep.avg_interventions:.1f}")

    return all_reports


def run_ablation_react_configs(
    functions: List[tuple],
    dims: List[int] = None,
    n_trials: int = 30,
    output_dir: str = "./experiments/results/ablation_react",
):
    """Ablation: compare basic ReAct vs Deep ReAct with/without memory."""
    dims = dims or [30]
    from LLM4PSO.LLMs4PSO import LLM4PSO

    configs_to_test = {
        "react_basic": {"intervention_mode": "llm_react", "react_mode": "basic"},
        "react_deep": {"intervention_mode": "llm_react", "react_mode": "deep"},
    }

    all_reports = []

    for label, overrides in configs_to_test.items():
        print(f"\n{'='*50}\nRunning config: {label}\n{'='*50}")

        def builder(func, dim, mode_label, seed, cfg=overrides):
            np.random.seed(seed)
            kwargs = _get_pso_defaults(dim, cfg["intervention_mode"])
            kwargs["func"] = func
            kwargs["react_mode"] = cfg.get("react_mode", "basic")
            return LLM4PSO(**kwargs)

        config = ExperimentConfig(
            name=f"ablation_{label}",
            description=f"Ablation: {label}",
            functions=functions,
            dimensions=dims,
            modes=[label],
            n_trials=n_trials,
            output_dir=output_dir,
        )
        runner = ExperimentRunner(config, builder, n_workers=1, verbose=False)
        reports = runner.run()
        runner.save_results()
        all_reports.extend(reports)

    # Print summary
    print(f"\n{'='*50}")
    print("ReAct Config Ablation Results:")
    print(f"{'='*50}")
    groups: Dict[tuple, Dict[str, Any]] = {}
    for r in all_reports:
        groups.setdefault((r.func_name, r.dim), {})[r.mode] = r

    for (func_name, dim), mode_map in sorted(groups.items()):
        print(f"\n  {func_name} (D={dim}):")
        for label in configs_to_test:
            rep = mode_map.get(label)
            if rep:
                print(f"    {label:15s}: mean={rep.final_mean:.4e}  std={rep.final_std:.4e}  "
                      f"llm_calls={rep.avg_llm_calls:.1f}")

    return all_reports


# ---------------------------------------------------------------------------
# Quick smoke-test: run a tiny ablation on Sphere to verify pipeline
# ---------------------------------------------------------------------------

def smoke_test():
    """Run a minimal ablation (2 trials, 10D Sphere) to verify the pipeline."""
    from LLM4PSO.compare.my_pso.function import Func

    print("Running smoke-test ablation...")
    reports = run_ablation_intervention_modes(
        functions=[("Sphere", Func())],
        dims=[5],
        n_trials=2,
        output_dir="./experiments/results/smoke_test",
    )
    print(f"Generated {len(reports)} reports.")

    # Verify reports have expected data
    for r in reports:
        assert r.n_trials == 2
        assert r.final_mean > 0
    print("Smoke test passed.")


if __name__ == "__main__":
    smoke_test()
