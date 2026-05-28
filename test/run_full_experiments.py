#!/usr/bin/env python
# run_full_experiments.py
r"""
Comprehensive experiment suite for LLM4PSO.

Experiments:
  1. Ablation study: none vs rule vs llm_react vs llm_react_deep
  2. SOTA comparison: PSO vs HPSOSCAC vs CLPSO vs jDE vs LLM4PSO(full)

Dimensions: 10D, 30D  |  Full parameter settings (no reduction).

Usage:
  python test/run_full_experiments.py --mode ablation --trials 30
  python test/run_full_experiments.py --mode comparison --trials 30
  python test/run_full_experiments.py --mode all --trials 30
  python test/run_full_experiments.py --mode quick            # 2-trial smoke test

Output structure:
  results/
    ablation/     — data/, iteration_data/, curves/
    comparison/   — data/, iteration_data/, curves/
    llm_react_thoughts/ — {func}/d{dim}/trial_{n}/intervention_{iter}.json
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Load .env for API keys if present
try:
    from llms.env_loader import load_env
    load_env(REPO_ROOT / ".env")
except Exception:
    pass

# ---------------------------------------------------------------------------
# matplotlib  setup
# ---------------------------------------------------------------------------
from matplotlib import pyplot as plt

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "legend.fontsize": 8,
})

# ---------------------------------------------------------------------------
# Full PSO parameters (no reduction)
# ---------------------------------------------------------------------------
FULL_PARAMS = {
    "pop_size": 30,
    "max_iter": 500,
    "w": 0.729,
    "wdamp": 0.99,
    "c1": 1.5,
    "c2": 1.5,
    "stagnation_threshold": 100,
    "improvement_tolerance": 1e-8,
}

RESULTS_ROOT = REPO_ROOT / "results"

# ---------------------------------------------------------------------------
# Benchmark functions (vectorised, compatible with all algorithms)
# ---------------------------------------------------------------------------


class FuncWrapper:
    """Wraps a vectorised function to provide both .eval() and .evaluate()."""

    def __init__(self, name, fn, bounds, optimum=0.0):
        self._name = name
        self._fn = fn
        self.bounds = bounds
        self.optimum = optimum

    def __call__(self, x):
        """Vectorised call for CLPSO / jDE (batch input)."""
        return self._fn(x)

    def eval(self, x):
        """Single-point evaluation for PSO / H_PSO_SCAC."""
        return float(self._fn(x.reshape(1, -1))[0])

    def evaluate(self, x):
        return self.eval(x)

    def to_str(self):
        return self._name

    @property
    def __name__(self):
        return self._name


def _sphere(x):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return np.sum(x**2, axis=1)


def _rosenbrock(x):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    x_i = x[:, :-1]
    x_next = x[:, 1:]
    return np.sum(100 * (x_next - x_i**2) ** 2 + (x_i - 1) ** 2, axis=1)


def _rastrigin(x):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return 10 * x.shape[1] + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1)


def _ackley(x):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    d = x.shape[1]
    s1 = np.sum(x**2, axis=1)
    s2 = np.sum(np.cos(2 * np.pi * x), axis=1)
    return -20 * np.exp(-0.2 * np.sqrt(s1 / d)) - np.exp(s2 / d) + 20 + np.e


def _griewank(x):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    d = x.shape[1]
    s1 = np.sum(x**2, axis=1) / 4000
    s2 = np.prod(np.cos(x / np.sqrt(np.arange(1, d + 1))), axis=1)
    return s1 - s2 + 1


BENCHMARKS = {
    "sphere": FuncWrapper("sphere", _sphere, (-100.0, 100.0)),
    "rosenbrock": FuncWrapper("rosenbrock", _rosenbrock, (-30.0, 30.0)),
    "rastrigin": FuncWrapper("rastrigin", _rastrigin, (-5.12, 5.12)),
    "ackley": FuncWrapper("ackley", _ackley, (-32.0, 32.0)),
    "griewank": FuncWrapper("griewank", _griewank, (-600.0, 600.0)),
}

# ---------------------------------------------------------------------------
# Helper: velocity bounds from position bounds
# ---------------------------------------------------------------------------


def vel_from_pos(pos_bounds, factor=0.2):
    span = pos_bounds[1] - pos_bounds[0]
    return (-factor * span, factor * span)


# ===================================================================
# Algorithm runners  (each returns full iteration history)
# ===================================================================


def run_standard_pso(func, dim, max_iter, seed):
    """Standard PSO with full history capture."""
    from LLM4PSO.compare.my_pso.Pso import PSO

    np.random.seed(seed)
    bounds = func.bounds
    vb = vel_from_pos(bounds)
    pso = PSO(
        func, FULL_PARAMS["pop_size"], dim, FULL_PARAMS["c1"], FULL_PARAMS["c2"],
        [bounds[0], bounds[1]], [vb[0], vb[1]], flag="else",
        w=FULL_PARAMS["w"], wdamp=FULL_PARAMS["wdamp"],
    )
    pso.pop_init()
    pso.evaluation(pso.flag)
    pso.update_p_best_cost()
    pso.update_g_best_cost()
    history = [pso.get_gBest()]

    for _ in range(max_iter):
        pso.update_pos_vel(pso.w)
        pso.evaluation(pso.flag)
        pso.update_p_best_cost()
        pso.update_g_best_cost()
        history.append(pso.get_gBest())
        pso.w *= pso.wdamp

    return np.array(history)


def run_hpsoscac(func, dim, max_iter, seed):
    """H_PSO_SCAC with full history capture."""
    from LLM4PSO.compare.my_pso.pso_scac import H_PSO_SCAC

    np.random.seed(seed)
    bounds = func.bounds
    vb = vel_from_pos(bounds)
    h = H_PSO_SCAC(
        func, FULL_PARAMS["pop_size"], dim, FULL_PARAMS["c1"], FULL_PARAMS["c2"],
        [bounds[0], bounds[1]], [vb[0], vb[1]], flag="else",
    )
    h.pop_init()
    h.evaluation()
    h.update_p_best_cost()
    h.update_g_best_cost()
    history = [h.get_gBest()]

    for it in range(1, max_iter + 1):
        h.update_c1_c2(it, max_iter)
        h.update_inertia_weight()
        h.update_pos_vel(it)
        h.evaluation()
        h.update_p_best_cost()
        h.update_g_best_cost()
        history.append(h.get_gBest())

    return np.array(history)


def run_clpso(func, dim, max_iter, seed):
    """CLPSO — returns full history."""
    from LLM4PSO.compare.clpso import CLPSO

    np.random.seed(seed)
    bounds = func.bounds
    vb = vel_from_pos(bounds)
    opt = CLPSO(
        func, n_pop=FULL_PARAMS["pop_size"], dim=dim,
        position_bound=bounds, velocity_bound=vb,
        w=FULL_PARAMS["w"], refreshing_gap=7,
    )
    history = opt.optimize(max_iter=max_iter, verbose=False)
    return history


def run_jde(func, dim, max_iter, seed):
    """jDE — returns full history."""
    from LLM4PSO.compare.jde import JDE

    np.random.seed(seed)
    bounds = func.bounds
    opt = JDE(func, n_pop=FULL_PARAMS["pop_size"], dim=dim, bounds=bounds)
    history = opt.optimize(max_iter=max_iter, verbose=False)
    return history


def run_llm4pso(func, dim, max_iter, seed, mode="rule", react_mode="basic"):
    """LLM4PSO — returns (full_history, action_history)."""
    from LLM4PSO.LLMs4PSO import LLM4PSO

    np.random.seed(seed)
    bounds = func.bounds
    vb = vel_from_pos(bounds)

    kwargs = dict(
        dim=dim, flag="cec", func=func,
        w=FULL_PARAMS["w"], pop_size=FULL_PARAMS["pop_size"],
        iterations=max_iter, wdamp=FULL_PARAMS["wdamp"],
        c1=FULL_PARAMS["c1"], c2=FULL_PARAMS["c2"],
        position_bounds=[bounds[0], bounds[1]],
        velocity_bounds=[vb[0], vb[1]],
        stagnation_threshold=FULL_PARAMS["stagnation_threshold"],
        improvement_tolerance=FULL_PARAMS["improvement_tolerance"],
        intervention_mode=mode,
    )

    if mode == "none":
        kwargs["intervention_mode"] = "rule"
        kwargs["stagnation_threshold"] = max_iter * 10  # never trigger
    elif mode in ("llm_react", "llm_react_deep"):
        kwargs["intervention_mode"] = "llm_react"
        kwargs["react_mode"] = react_mode

    opt = LLM4PSO(**kwargs)

    # Re-assign LLM in case constructor failed to create one
    if mode in ("llm_react",) and opt._react_controller is not None:
        try:
            from llms.factory import create_llm
            opt._react_controller.llm = create_llm("deepseek")
        except Exception:
            pass

    history = opt.run()
    return np.array(history), list(opt.action_history)


# ===================================================================
# Data-saving utilities
# ===================================================================


def save_iteration_csv(history, algo_name, func_name, dim, trial, output_dir):
    """Save one trial's iteration history as CSV."""
    out = Path(output_dir) / "iteration_data" / algo_name
    out.mkdir(parents=True, exist_ok=True)
    fname = out / f"{func_name}_d{dim}_trial{trial:03d}.csv"
    np.savetxt(fname, history, delimiter=",", header="best_cost", comments="")
    return fname


def save_summary_json(stats, func_name, dim, output_dir):
    """Save per-(function,dim) aggregated statistics to JSON."""
    out = Path(output_dir) / "data"
    out.mkdir(parents=True, exist_ok=True)
    fname = out / f"{func_name}_d{dim}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return fname


def save_react_thoughts(action_history, func_name, dim, trial, output_dir):
    """Persist LLM ReAct thought chains for each intervention call."""
    out = Path(output_dir) / "llm_react_thoughts" / str(func_name) / f"d{dim}" / f"trial_{trial:03d}"
    out.mkdir(parents=True, exist_ok=True)

    for entry in action_history:
        if entry.get("mode") not in ("llm", "llm_react"):
            continue
        iteration = entry.get("iteration", 0)
        fname = out / f"intervention_{iteration:04d}.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2, default=str)


def save_intervention_log(action_history, algo_name, func_name, dim, trial, output_dir):
    """Save intervention summary (non-LLM modes) for analysis."""
    out = Path(output_dir) / "intervention_logs" / algo_name
    out.mkdir(parents=True, exist_ok=True)
    fname = out / f"{func_name}_d{dim}_trial{trial:03d}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(action_history, f, ensure_ascii=False, indent=2, default=str)


# ===================================================================
# Convergence-curve plotting
# ===================================================================

# Colour maps
ABLATION_COLORS = {
    "none": "#069DFF",
    "rule": "#808080",
    "llm_react": "#A4E048",
    "llm_react_deep": "#010101",
}

COMPARISON_COLORS = {
    "PSO": "#BABABA",
    "HPSOSCAC": "#0001A1",
    "CLPSO": "#037F77",
    "jDE": "#C5272D",
    "LLM4PSO": "#F4A99B",
}


def plot_convergence_curves(
    histories_dict,       # {label: list_of_1d_arrays}
    func_name,
    dim,
    output_path,
    colors=None,
    log_scale=True,
    title_prefix="",
):
    """Plot all algorithms/modes on a single figure with mean ± std band.

    Args:
        histories_dict: {label: [history_array, ...]} — one array per trial.
        func_name: function name for the title.
        dim: dimension.
        output_path: path to save the PNG.
        colors: optional dict {label: colour}.
        log_scale: use log y-axis.
        title_prefix: prefix for the plot title.
    """
    colors = colors or {}
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort labels for consistent legend ordering
    sorted_labels = sorted(histories_dict.keys())
    for label in sorted_labels:
        hists = histories_dict[label]
        if not hists:
            continue

        # Pad to uniform length
        max_len = max(len(h) for h in hists)
        padded = np.full((len(hists), max_len), np.nan)
        for i, h in enumerate(hists):
            padded[i, :len(h)] = h

        mean = np.nanmean(padded, axis=0)
        iters = np.arange(len(mean))
        color = colors.get(label, None)
        ax.plot(iters, mean, label=label, linewidth=1.5, color=color, alpha=0.9)

        if len(hists) >= 5:
            std = np.nanstd(padded, axis=0)
            ax.fill_between(iters, mean - std, mean + std, alpha=0.12, color=color)

    full_title = f"{title_prefix}{func_name} (D={dim})"
    ax.set_title(full_title, pad=12)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Cost" + (" (log scale)" if log_scale else ""))
    if log_scale:
        ax.set_yscale("log")
    ax.legend(fontsize=8, ncol=2 if len(sorted_labels) > 4 else 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {output_path}")


# ===================================================================
# Statistics helpers
# ===================================================================


def compute_stats(histories):
    """Compute aggregate statistics from a list of trial histories (final values)."""
    finals = np.array([h[-1] for h in histories if len(h) > 0 and np.isfinite(h[-1])])
    if len(finals) == 0:
        return {"mean": float("inf"), "std": 0.0, "median": float("inf"),
                "best": float("inf"), "worst": float("inf"), "n_valid": 0}
    return {
        "mean": float(np.mean(finals)),
        "std": float(np.std(finals, ddof=1)) if len(finals) > 1 else 0.0,
        "median": float(np.median(finals)),
        "best": float(np.min(finals)),
        "worst": float(np.max(finals)),
        "n_valid": len(finals),
    }


# ===================================================================
# Experiment: Ablation  (none / rule / llm_react / llm_react_deep)
# ===================================================================

ABLATION_MODES = [
    ("none", "none", None),
    ("rule", "rule", None),
    ("llm_react", "llm_react", "basic"),
    ("llm_react_deep", "llm_react", "deep"),
]


def run_ablation_experiment(functions, dims, n_trials, output_dir, verbose=True):
    """Ablation study comparing intervention modes."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for func_name in functions:
        func = BENCHMARKS[func_name]
        for dim in dims:
            if verbose:
                print(f"\n{'='*55}\n  Ablation: {func_name}  D={dim}\n{'='*55}")

            all_histories = {}          # {mode_label: [array, ...]}
            all_action_histories = {}   # {mode_label: [list_of_actions, ...]}
            summary = {}

            for label, mode, react_mode in ABLATION_MODES:
                if verbose:
                    print(f"  [{label}] ", end="", flush=True)
                t_start = time.time()

                trial_histories = []
                trial_actions = []
                for trial in range(n_trials):
                    seed = 42 + trial * 1009 + dim * 13
                    try:
                        history, actions = run_llm4pso(func, dim, FULL_PARAMS["max_iter"], seed, mode, react_mode)
                        trial_histories.append(history)
                        trial_actions.append(actions)
                        # Save per-trial CSV
                        save_iteration_csv(history, label, func_name, dim, trial, output_dir)
                        # Save LLM thoughts if applicable
                        if mode in ("llm_react", "llm_react_deep"):
                            save_react_thoughts(actions, func_name, dim, trial, output_dir)
                        else:
                            save_intervention_log(actions, label, func_name, dim, trial, output_dir)
                    except Exception as e:
                        print(f"E", end="", flush=True)
                        trial_histories.append(np.array([float("inf")]))
                        trial_actions.append([])

                elapsed = time.time() - t_start
                stats = compute_stats(trial_histories)
                summary[label] = stats
                all_histories[label] = trial_histories
                all_action_histories[label] = trial_actions

                if verbose:
                    n_int = int(np.mean([len(a) for a in trial_actions])) if trial_actions else 0
                    print(f"mean={stats['mean']:.4e}  best={stats['best']:.4e}  intv≈{n_int}  {elapsed:.1f}s")

            # Save summary JSON
            save_summary_json(summary, func_name, dim, output_dir)

            # Plot convergence curves (all modes on one figure)
            curve_path = output_dir / "curves" / f"{func_name}_d{dim}.png"
            plot_convergence_curves(
                all_histories, func_name, dim, str(curve_path),
                colors=ABLATION_COLORS, log_scale=True,
                title_prefix="Ablation: ",
            )

    print(f"\nAblation results saved to {output_dir}")


# ===================================================================
# Experiment: SOTA Comparison  (PSO / HPSOSCAC / CLPSO / jDE / LLM4PSO)
# ===================================================================

COMPARISON_ALGOS = [
    ("PSO", run_standard_pso),
    ("HPSOSCAC", run_hpsoscac),
    ("CLPSO", run_clpso),
    ("jDE", run_jde),
]


def run_comparison_experiment(functions, dims, n_trials, output_dir, llm_mode="rule", verbose=True):
    """Compare LLM4PSO (complete) against SOTA optimisers."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for func_name in functions:
        func = BENCHMARKS[func_name]
        for dim in dims:
            if verbose:
                print(f"\n{'='*55}\n  Comparison: {func_name}  D={dim}\n{'='*55}")

            all_histories = {}
            summary = {}

            # ---- SOTA baselines ----
            for algo_name, runner in COMPARISON_ALGOS:
                if verbose:
                    print(f"  [{algo_name}] ", end="", flush=True)
                t_start = time.time()

                trial_histories = []
                for trial in range(n_trials):
                    seed = 42 + trial * 1009 + dim * 13
                    try:
                        history = runner(func, dim, FULL_PARAMS["max_iter"], seed)
                        trial_histories.append(history)
                        save_iteration_csv(history, algo_name, func_name, dim, trial, output_dir)
                    except Exception:
                        print(f"E", end="", flush=True)
                        trial_histories.append(np.array([float("inf")]))

                elapsed = time.time() - t_start
                stats = compute_stats(trial_histories)
                summary[algo_name] = stats
                all_histories[algo_name] = trial_histories

                if verbose:
                    print(f"mean={stats['mean']:.4e}  best={stats['best']:.4e}  {elapsed:.1f}s")

            # ---- LLM4PSO (complete algorithm) ----
            label = f"LLM4PSO({llm_mode})"
            if verbose:
                print(f"  [{label}] ", end="", flush=True)
            t_start = time.time()

            trial_histories = []
            trial_actions = []
            react_mode = "deep" if "deep" in llm_mode else "basic"
            actual_mode = llm_mode if llm_mode in ("rule", "llm_react") else "rule"

            for trial in range(n_trials):
                seed = 42 + trial * 1009 + dim * 13
                try:
                    history, actions = run_llm4pso(func, dim, FULL_PARAMS["max_iter"], seed, actual_mode, react_mode)
                    trial_histories.append(history)
                    trial_actions.append(actions)
                    save_iteration_csv(history, "LLM4PSO", func_name, dim, trial, output_dir)
                    if actual_mode == "llm_react":
                        save_react_thoughts(actions, func_name, dim, trial, output_dir)
                    else:
                        save_intervention_log(actions, "LLM4PSO", func_name, dim, trial, output_dir)
                except Exception:
                    print(f"E", end="", flush=True)
                    traceback.print_exc()
                    trial_histories.append(np.array([float("inf")]))
                    trial_actions.append([])

            elapsed = time.time() - t_start
            stats = compute_stats(trial_histories)
            summary[label] = stats
            all_histories[label] = trial_histories

            if verbose:
                n_int = int(np.mean([len(a) for a in trial_actions])) if trial_actions else 0
                print(f"mean={stats['mean']:.4e}  best={stats['best']:.4e}  intv≈{n_int}  {elapsed:.1f}s")

            # Save summary JSON
            save_summary_json(summary, func_name, dim, output_dir)

            # Plot convergence curves (all algorithms on one figure)
            curve_path = output_dir / "curves" / f"{func_name}_d{dim}.png"
            plot_convergence_curves(
                all_histories, func_name, dim, str(curve_path),
                colors=COMPARISON_COLORS, log_scale=True,
                title_prefix="",
            )

    print(f"\nComparison results saved to {output_dir}")


# ===================================================================
# CLI
# ===================================================================


def main():
    parser = argparse.ArgumentParser(
        description="LLM4PSO Comprehensive Experiment Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test/run_full_experiments.py --mode quick
  python test/run_full_experiments.py --mode ablation --trials 30
  python test/run_full_experiments.py --mode comparison --trials 30
  python test/run_full_experiments.py --mode all --functions sphere rastrigin --dims 10
        """,
    )
    parser.add_argument("--mode", default="ablation",
                        choices=["quick", "ablation", "comparison", "all"])
    parser.add_argument("--dims", type=int, nargs="+", default=None)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--functions", nargs="+", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--llm-mode", default="rule",
                        choices=["rule", "llm_react"],
                        help="LLM4PSO mode for comparison experiment (default: rule)")
    parser.add_argument("--max-iter", type=int, default=None)
    args = parser.parse_args()
    # ---- Resolve defaults ----
    if args.dims is None:
        args.dims = [30]
    if args.functions is None:
        if args.mode == "quick":
            args.functions = ["sphere", "rastrigin"]
        else:
            args.functions = list(BENCHMARKS.keys())
    if args.trials is None:
        args.trials = 1
    if args.max_iter is not None:
        FULL_PARAMS["max_iter"] = args.max_iter

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---- Run ----
    print(f"{'='*60}")
    print(f"  LLM4PSO Full Experiment Suite")
    print(f"  Mode: {args.mode}  |  Dims: {args.dims}  |  Trials: {args.trials}")
    print(f"  Functions: {args.functions}")
    print(f"  Max iter: {FULL_PARAMS['max_iter']}  |  Pop size: {FULL_PARAMS['pop_size']}")
    print(f"  Stagnation threshold: {FULL_PARAMS['stagnation_threshold']}")
    print(f"  w={FULL_PARAMS['w']}  wdamp={FULL_PARAMS['wdamp']}  c1={FULL_PARAMS['c1']}  c2={FULL_PARAMS['c2']}")
    print(f"{'='*60}")

    if args.mode in ("ablation", "all"):
        out_dir = args.output or (RESULTS_ROOT / f"ablation_{ts}" if args.mode == "ablation" else RESULTS_ROOT / "ablation")
        if args.mode == "all":
            out_dir = RESULTS_ROOT / "ablation"
        run_ablation_experiment(args.functions, args.dims, args.trials, out_dir)

    if args.mode in ("comparison", "all"):
        out_dir = args.output or (RESULTS_ROOT / f"comparison_{ts}" if args.mode == "comparison" else RESULTS_ROOT / "comparison")
        if args.mode == "all":
            out_dir = RESULTS_ROOT / "comparison"
        # FIXME: comparison experiment temporarily disabled — focus on ablation first
        # run_comparison_experiment(args.functions, args.dims, args.trials, out_dir,
        #                           llm_mode=args.llm_mode)
        print(f"  [SKIP] Comparison experiment disabled. Would save to: {out_dir}")

    if args.mode == "quick":
        # Quick smoke test: ablation only, 2 functions, 10D, 1 trial
        out_dir = args.output or (RESULTS_ROOT / "quick_smoke_test")
        print("\n--- Ablation smoke test ---")
        run_ablation_experiment(args.functions, [30], 1, str(Path(out_dir) / "ablation"))
        # FIXME: comparison smoke test temporarily disabled
        # print("\n--- Comparison smoke test ---")
        # run_comparison_experiment(args.functions, [10], 1, str(Path(out_dir) / "comparison"),
        #                           llm_mode=args.llm_mode)

    print(f"\n{'='*60}")
    print(f"  All experiments complete.")
    print(f"  Results saved under: {RESULTS_ROOT}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
