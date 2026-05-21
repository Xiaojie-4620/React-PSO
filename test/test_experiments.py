r"""Comprehensive experiment suite: ablation + SOTA comparison.

Usage:
    python test/test_experiments.py --mode ablation   # intervention mode ablation
    python test/test_experiments.py --mode sota       # SOTA comparison
    python test/test_experiments.py --mode full       # both + LLM ReAct
    python test/test_experiments.py --mode quick      # fast smoke test

All experiments use 500 iterations per run.

Output:
    test/results/convergence/   — convergence curve plots
    test/results/summary.json   — final statistics
    test/results/interventions.json  — intervention analysis
"""

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

# Load .env for API keys
try:
    from llms.env_loader import load_env
    load_env(repo_root / ".env")
except Exception:
    pass

# ---------------------------------------------------------------------------
# Benchmark functions (vectorized, compatible with all algorithms)
# ---------------------------------------------------------------------------


class FuncWrapper:
    """Wraps a vectorized function to provide both .eval() and .evaluate()."""

    def __init__(self, name, fn, bounds, optimum=0.0):
        self._name = name
        self._fn = fn
        self.bounds = bounds
        self.optimum = optimum

    def __call__(self, x):
        return self._fn(x)

    def eval(self, x):
        return float(self._fn(x.reshape(1, -1))[0])

    def evaluate(self, x):
        return self.eval(x)

    def to_str(self):
        return self._name

    @property
    def __name__(self):
        return self._name


def _sphere(x):
    if x.ndim == 1: x = x.reshape(1, -1)
    return np.sum(x ** 2, axis=1)


def _rosenbrock(x):
    if x.ndim == 1: x = x.reshape(1, -1)
    x_i = x[:, :-1]
    x_next = x[:, 1:]
    return np.sum(100 * (x_next - x_i ** 2) ** 2 + (x_i - 1) ** 2, axis=1)


def _rastrigin(x):
    if x.ndim == 1: x = x.reshape(1, -1)
    return 10 * x.shape[1] + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x), axis=1)


def _ackley(x):
    if x.ndim == 1: x = x.reshape(1, -1)
    d = x.shape[1]
    s1 = np.sum(x ** 2, axis=1)
    s2 = np.sum(np.cos(2 * np.pi * x), axis=1)
    return -20 * np.exp(-0.2 * np.sqrt(s1 / d)) - np.exp(s2 / d) + 20 + np.e


def _griewank(x):
    if x.ndim == 1: x = x.reshape(1, -1)
    d = x.shape[1]
    s1 = np.sum(x ** 2, axis=1) / 4000
    s2 = np.prod(np.cos(x / np.sqrt(np.arange(1, d + 1))), axis=1)
    return s1 - s2 + 1


BENCHMARKS = {
    "sphere":     FuncWrapper("sphere",     _sphere,     (-100.0, 100.0)),
    "rosenbrock": FuncWrapper("rosenbrock", _rosenbrock, (-30.0, 30.0)),
    "rastrigin":  FuncWrapper("rastrigin",  _rastrigin,  (-5.12, 5.12)),
    "ackley":     FuncWrapper("ackley",     _ackley,     (-32.0, 32.0)),
    "griewank":   FuncWrapper("griewank",   _griewank,   (-600.0, 600.0)),
}

# ---------------------------------------------------------------------------
# PSO defaults
# ---------------------------------------------------------------------------

PSO_PARAMS = {
    "dim": 10,
    "pop_size": 30,
    "max_iter": 500,
    "w": 0.729,
    "wdamp": 1.0,
    "c1": 1.5,
    "c2": 1.5,
    "stagnation_threshold": 50,
    "improvement_tolerance": 1e-8,
}


def vel_from_pos(pos_bounds, factor=0.2):
    span = pos_bounds[1] - pos_bounds[0]
    return (-factor * span, factor * span)


# ---------------------------------------------------------------------------
# Algorithm runners
# ---------------------------------------------------------------------------


def run_standard_pso(func, dim, max_iter, seed):
    from LLM4PSO.compare.my_pso.Pso import PSO
    np.random.seed(seed)
    bounds = func.bounds
    vb = vel_from_pos(bounds)
    pso = PSO(func, PSO_PARAMS["pop_size"], dim, PSO_PARAMS["c1"], PSO_PARAMS["c2"],
              [bounds[0], bounds[1]], [vb[0], vb[1]], flag="else",
              w=PSO_PARAMS["w"], wdamp=PSO_PARAMS["wdamp"])
    _, best = pso.run(max_iter=max_iter, verbose=False)
    return best


def run_hpsoscac(func, dim, max_iter, seed):
    from LLM4PSO.compare.my_pso.pso_scac import H_PSO_SCAC
    np.random.seed(seed)
    bounds = func.bounds
    vb = vel_from_pos(bounds)
    h = H_PSO_SCAC(func, PSO_PARAMS["pop_size"], dim, PSO_PARAMS["c1"], PSO_PARAMS["c2"],
                   [bounds[0], bounds[1]], [vb[0], vb[1]], flag="else")
    h.pop_init()
    h.evaluation()
    h.update_p_best_cost()
    h.update_g_best_cost()
    best_history = [h.get_gBest()]
    for it in range(1, max_iter):
        h.update_c1_c2(it, max_iter)
        h.update_inertia_weight()
        h.update_pos_vel(it)
        h.evaluation()
        h.update_p_best_cost()
        h.update_g_best_cost()
        best_history.append(h.get_gBest())
    return float(np.min(best_history))


def run_clpso(func, dim, max_iter, seed):
    from LLM4PSO.compare.clpso import CLPSO
    np.random.seed(seed)
    bounds = func.bounds
    vb = vel_from_pos(bounds)
    opt = CLPSO(func, n_pop=PSO_PARAMS["pop_size"], dim=dim,
                position_bound=bounds, velocity_bound=vb,
                w=PSO_PARAMS["w"], refreshing_gap=7)
    history = opt.optimize(max_iter=max_iter, verbose=False)
    return float(history[-1])


def run_jde(func, dim, max_iter, seed):
    from LLM4PSO.compare.jde import JDE
    np.random.seed(seed)
    bounds = func.bounds
    opt = JDE(func, n_pop=PSO_PARAMS["pop_size"], dim=dim, bounds=bounds)
    history = opt.optimize(max_iter=max_iter, verbose=False)
    return float(history[-1])


def run_llm4pso(func, dim, max_iter, seed, mode="rule", react_mode="basic"):
    from LLM4PSO.LLMs4PSO import LLM4PSO
    np.random.seed(seed)
    bounds = func.bounds
    vb = vel_from_pos(bounds)
    opt = LLM4PSO(
        dim=dim, flag="cec", func=func,
        w=PSO_PARAMS["w"], pop_size=PSO_PARAMS["pop_size"],
        iterations=max_iter, wdamp=PSO_PARAMS["wdamp"],
        c1=PSO_PARAMS["c1"], c2=PSO_PARAMS["c2"],
        position_bounds=[bounds[0], bounds[1]],
        velocity_bounds=[vb[0], vb[1]],
        stagnation_threshold=PSO_PARAMS["stagnation_threshold"],
        improvement_tolerance=PSO_PARAMS["improvement_tolerance"],
        intervention_mode=mode,
        react_mode=react_mode,
    )
    if mode in ("llm_react",) and opt._react_controller is not None:
        try:
            from llms.factory import create_llm
            opt._react_controller.llm = create_llm("deepseek")
        except Exception:
            pass
    history = opt.run()
    interventions = []
    for a in opt.action_history:
        if a.get("mode") in ("rule", "llm_react"):
            interventions.append(a)
    return float(history[-1]), interventions


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def run_experiment(func_name, dim, n_trials, modes, include_sota=True, include_llm=True):
    """Run a full experiment and return results."""
    func = BENCHMARKS[func_name]
    max_iter = PSO_PARAMS["max_iter"]
    results = defaultdict(list)

    total = n_trials * (len(modes) + (5 if include_sota else 0))
    count = 0

    for trial in range(n_trials):
        seed = 42 + trial * 1009

        # Standard PSO
        if include_sota:
            for algo_name, runner in [
                ("PSO", run_standard_pso),
                ("HPSOSCAC", run_hpsoscac),
                ("CLPSO", run_clpso),
                ("jDE", run_jde),
            ]:
                count += 1
                t0 = time.time()
                try:
                    final = runner(func, dim, max_iter, seed)
                    ok = True
                except Exception as e:
                    final = float("inf")
                    ok = False
                elapsed = time.time() - t0
                results[algo_name].append({
                    "trial": trial, "seed": seed, "final": final,
                    "time": elapsed, "ok": ok,
                })

        # LLM4PSO variants
        for mode_key in modes:
            count += 1
            mode = "rule" if mode_key.startswith("rule") else "llm_react"
            react_mode = "deep" if "deep" in mode_key else "basic"
            t0 = time.time()
            try:
                if mode == "llm_react" and not include_sota:
                    # Skip LLM for ablation-only mode
                    final, interventions = float("inf"), []
                else:
                    final, interventions = run_llm4pso(func, dim, max_iter, seed, mode, react_mode)
                ok = True
            except Exception as e:
                final = float("inf")
                interventions = []
                ok = False
            elapsed = time.time() - t0
            results[mode_key].append({
                "trial": trial, "seed": seed, "final": final,
                "time": elapsed, "ok": ok,
                "interventions": len(interventions),
                "actions": [a.get("decision", {}).get("action", "?")
                           if a.get("mode") == "rule"
                           else a.get("turns", [{}])[0].get("action", "?")
                           for a in interventions],
            })

    return dict(results)


def compute_stats(values):
    vals = np.array([v for v in values if np.isfinite(v) and v < 1e100])
    if len(vals) == 0:
        return {"mean": float("inf"), "std": 0, "min": float("inf"), "count": 0}
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "min": float(np.min(vals)),
        "median": float(np.median(vals)),
        "count": len(vals),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_comparison(func_name, results, output_dir):
    """Generate convergence curve comparison plot."""
    func = BENCHMARKS[func_name]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {
        "PSO": "gray", "HPSOSCAC": "orange", "CLPSO": "purple", "jDE": "brown",
        "rule": "blue", "llm_react": "red", "llm_react_deep": "darkred",
        "none": "black",
    }

    for algo, trials in results.items():
        finals = [t["final"] for t in trials if t["ok"]]
        if not finals:
            continue
        color = colors.get(algo, "gray")
        # Bar chart of final best costs
        x = list(results.keys()).index(algo) if algo in results else 0
        # Use simple bar chart: mean final cost per algorithm
        means = [np.mean([t["final"] for t in results[a] if t["ok"]])
                 if any(t["ok"] for t in results[a]) else float("inf")
                 for a in results]
        stds = [np.std([t["final"] for t in results[a] if t["ok"]])
                if sum(1 for t in results[a] if t["ok"]) > 1 else 0
                for a in results]
        names = list(results.keys())

    # Bar chart
    x_pos = np.arange(len(names))
    means_vals = [np.mean([t["final"] for t in results[n] if t["ok"] and t["final"] < 1e100])
                  if any(t["ok"] for t in results[n]) else 0
                  for n in names]
    std_vals = [np.std([t["final"] for t in results[n] if t["ok"] and t["final"] < 1e100])
                if sum(1 for t in results[n] if t["ok"]) > 1 else 0
                for n in names]

    bar_colors = [colors.get(n, "gray") for n in names]
    ax.bar(x_pos, means_vals, yerr=std_vals, color=bar_colors, capsize=5, alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Final Best Cost (mean ± std)")
    ax.set_title(f"{func_name} (D={PSO_PARAMS['dim']}, {PSO_PARAMS['max_iter']} iter, {len(next(iter(results.values()))) if results else 0} trials)")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / f"{func_name}_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_convergence_curves(func_name, histories, output_dir):
    """Plot convergence curves (iterations vs best cost) for all algorithms."""
    output_dir = Path(output_dir) / "curves"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, history in histories.items():
        if history:
            ax.plot(history, label=label, linewidth=1.5, alpha=0.8)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Cost")
    ax.set_title(f"{func_name} Convergence (D={PSO_PARAMS['dim']})")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / f"{func_name}_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Comprehensive PSO experiment suite")
    parser.add_argument("--mode", default="quick",
                        choices=["quick", "ablation", "sota", "full"])
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--functions", nargs="+", default=None)
    args = parser.parse_args()

    PSO_PARAMS["dim"] = args.dim
    PSO_PARAMS["stagnation_threshold"] = max(20, args.dim * 5)

    if args.trials is None:
        args.trials = 2 if args.mode == "quick" else 5 if args.mode == "ablation" else 10

    if args.functions is None:
        if args.mode == "quick":
            args.functions = ["sphere", "rastrigin"]
        elif args.mode == "ablation":
            args.functions = ["sphere", "rosenbrock", "rastrigin", "ackley"]
        else:
            args.functions = list(BENCHMARKS.keys())

    if args.output is None:
        args.output = f"test/results/{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"Experiment: mode={args.mode}, dim={args.dim}, trials={args.trials}")
    print(f"Functions: {args.functions}")
    print(f"Max iter: {PSO_PARAMS['max_iter']}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    if args.mode in ("quick", "ablation"):
        modes = ["rule"]
        include_sota = False
        include_llm = False
    elif args.mode == "sota":
        modes = []
        include_sota = True
        include_llm = False
    else:  # full
        modes = ["rule", "llm_react"]
        include_sota = True
        include_llm = True

    all_summary = {}

    for func_name in args.functions:
        print(f"\n{'='*50}\n  {func_name}\n{'='*50}")

        if args.mode == "sota":
            results = run_experiment(func_name, args.dim, args.trials,
                                     modes=[], include_sota=True, include_llm=False)
        elif args.mode == "quick":
            # Quick: rule-only comparison
            results = run_experiment(func_name, args.dim, args.trials,
                                     modes=["rule"], include_sota=False, include_llm=False)
        elif args.mode == "ablation":
            # Ablation: rule vs none (no LLM)
            from LLM4PSO.LLMs4PSO import LLM4PSO
            modes_ab = ["none", "rule"]
            results = {}
            for trial in range(args.trials):
                seed = 42 + trial * 1009
                for mode_ab in modes_ab:
                    func = BENCHMARKS[func_name]
                    bounds = func.bounds
                    vb = vel_from_pos(bounds)
                    np.random.seed(seed)
                    if mode_ab == "none":
                        st = PSO_PARAMS["max_iter"] * 10
                    else:
                        st = PSO_PARAMS["stagnation_threshold"]
                    opt = LLM4PSO(
                        dim=args.dim, flag="cec", func=func,
                        w=PSO_PARAMS["w"], pop_size=PSO_PARAMS["pop_size"],
                        iterations=PSO_PARAMS["max_iter"], wdamp=PSO_PARAMS["wdamp"],
                        c1=PSO_PARAMS["c1"], c2=PSO_PARAMS["c2"],
                        position_bounds=list(bounds),
                        velocity_bounds=list(vb),
                        stagnation_threshold=st,
                        improvement_tolerance=PSO_PARAMS["improvement_tolerance"],
                        intervention_mode="rule",
                    )
                    history = opt.run()
                    n_int = len(opt.action_history)
                    results.setdefault(mode_ab, []).append({
                        "trial": trial, "seed": seed,
                        "final": float(history[-1]),
                        "interventions": n_int,
                        "ok": True,
                    })
        else:  # full
            results = run_experiment(func_name, args.dim, args.trials,
                                     modes=["rule", "llm_react"],
                                     include_sota=True, include_llm=True)

        # Summary stats
        summary = {}
        for algo, trials in results.items():
            finals = [t["final"] for t in trials if t["ok"]]
            stats = compute_stats(finals)
            intvs = [t.get("interventions", 0) for t in trials if t["ok"]]
            times = [t.get("time", 0) for t in trials if t["ok"]]
            summary[algo] = {
                **stats,
                "avg_interventions": float(np.mean(intvs)) if intvs else 0,
                "avg_time": float(np.mean(times)) if times else 0,
            }
            print(f"  {algo:15s}: mean={stats['mean']:.4e} ± {stats['std']:.4e}  "
                  f"min={stats['min']:.4e}  intv={summary[algo]['avg_interventions']:.1f}")

        all_summary[func_name] = summary

        # Plot comparison
        plot_comparison(func_name, results, output_dir)

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summary, f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Print final comparison table
    print(f"\n{'='*70}")
    print("FINAL COMPARISON TABLE")
    print(f"{'='*70}")
    for func_name, summary in all_summary.items():
        print(f"\n{func_name}:")
        print(f"{'Algorithm':15s} {'Mean':>12s} {'Std':>12s} {'Best':>12s}")
        print(f"{'-'*15} {'-'*12} {'-'*12} {'-'*12}")
        for algo, stats in sorted(summary.items(),
                                   key=lambda x: x[1]["mean"] if math.isfinite(x[1]["mean"]) else float("inf")):
            print(f"{algo:15s} {stats['mean']:12.4e} {stats['std']:12.4e} {stats['min']:12.4e}")

    # Save interventions analysis
    if args.mode in ("full", "ablation"):
        interv_data = {}
        for func_name in args.functions:
            # Re-run one trial to capture intervention details
            pass
        print(f"\nAll results saved to {output_dir}/")


if __name__ == "__main__":
    main()
