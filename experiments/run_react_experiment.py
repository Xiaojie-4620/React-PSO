"""Run real LLM-based ReAct PSO experiment using configured API keys.

Usage:
    python experiments/run_react_experiment.py --func sphere --dim 10 --mode llm_react
    python experiments/run_react_experiment.py --func rastrigin --dim 10 --mode deep_react
    python experiments/run_react_experiment.py --func sphere --dim 5 --mode rule
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

# Load .env before anything else
from llms.env_loader import load_env
load_env(repo_root / ".env")


def sphere(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return np.sum(x ** 2, axis=1)


def rastrigin(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return 10 * x.shape[1] + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x), axis=1)


def ackley(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        x = x.reshape(1, -1)
    d = x.shape[1]
    return (
        -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2, axis=1) / d))
        - np.exp(np.sum(np.cos(2 * np.pi * x), axis=1) / d)
        + 20 + np.e
    )


FUNCTIONS = {
    "sphere": sphere,
    "rastrigin": rastrigin,
    "ackley": ackley,
}


def create_llm(provider: str = "deepseek"):
    """Create LLM instance from configured provider."""
    from llms.factory import create_llm as _factory

    try:
        return _factory(provider)
    except Exception as e:
        print(f"[WARN] Failed to create {provider} LLM: {e}")
        # Fallback: try other providers
        for fallback in ["yunwu", "deepseek"]:
            if fallback == provider:
                continue
            try:
                llm = _factory(fallback)
                print(f"[INFO] Fallback to {fallback} LLM")
                return llm
            except Exception:
                pass
        return None


def run_experiment(
    func_name: str,
    dim: int,
    mode: str,
    provider: str = "deepseek",
    pop_size: int = 30,
    max_iter: int = 200,
    stagnation_threshold: int = 30,
    react_mode: str = "basic",
):
    """Run a single experiment and return results."""
    from LLM4PSO.LLMs4PSO import LLM4PSO

    func = FUNCTIONS.get(func_name, sphere)
    pos_bounds = (-100.0, 100.0)
    vel_bounds = (-20.0, 20.0)

    print(f"\n{'='*60}")
    print(f"Experiment: {func_name} D={dim} mode={mode} provider={provider}")
    print(f"pop_size={pop_size}, max_iter={max_iter}, stagnation_threshold={stagnation_threshold}")
    print(f"{'='*60}")

    optimizer = LLM4PSO(
        dim=dim,
        flag="cec",
        func=func,
        w=0.729,
        pop_size=pop_size,
        iterations=max_iter,
        wdamp=1.0,
        c1=1.5,
        c2=1.5,
        position_bounds=pos_bounds,
        velocity_bounds=vel_bounds,
        stagnation_threshold=stagnation_threshold,
        improvement_tolerance=1e-8,
        intervention_mode=mode,
        react_mode=react_mode,
    )

    # Replace LLM with the configured provider
    if mode in ("llm_react",) and optimizer._react_controller is not None:
        llm = create_llm(provider)
        if llm is None:
            print("[ERROR] No LLM provider available. Check .env configuration.")
            return None
        optimizer._react_controller.llm = llm
        print(f"[INFO] Using LLM provider: {type(llm).__name__}")

    t0 = datetime.now()
    history = optimizer.run()
    elapsed = (datetime.now() - t0).total_seconds()

    # Print summary
    n_interventions = len(optimizer.action_history)
    n_react_turns = sum(
        len(a.get("turns", [])) for a in optimizer.action_history
        if a.get("mode") in ("llm_react",)
    )

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Final best cost: {history[-1]:.6e}")
    print(f"  Initial cost:    {history[0]:.6e}")
    print(f"  Improvement:     {(history[0] - history[-1]) / max(abs(history[0]), 1e-12) * 100:.2f}%")
    print(f"  Interventions:   {n_interventions}")
    print(f"  ReAct turns:     {n_react_turns}")
    print(f"  Wall time:       {elapsed:.1f}s")
    print(f"{'='*60}")

    # Print ReAct trace if available
    if optimizer.action_history and mode in ("llm_react",):
        print(f"\n--- ReAct Trace ---")
        for i, action in enumerate(optimizer.action_history):
            if action.get("mode") != "llm_react":
                continue
            print(f"\n[Intervention at iter {action['iteration']}]")
            state = action.get("state", {})
            print(f"  State: {state.get('state_label', '?')} "
                  f"(diversity={state.get('normalized_diversity', '?'):.4f}, "
                  f"stalled={state.get('no_improve_iters', '?')})")
            for turn in action.get("turns", []):
                thought = turn.get("thought", "")[:120]
                print(f"  Turn {turn['turn']}: action={turn['action']}, applied={turn.get('improvement', 0) > 1e-6}")
                if thought:
                    print(f"    Thought: {thought}...")
        print(f"--- End ReAct Trace ---\n")

    return {
        "func_name": func_name,
        "dim": dim,
        "mode": mode,
        "provider": provider,
        "react_mode": react_mode,
        "final_cost": float(history[-1]),
        "initial_cost": float(history[0]),
        "history": history.tolist(),
        "n_interventions": n_interventions,
        "n_react_turns": n_react_turns,
        "wall_time": elapsed,
        "action_history": optimizer.action_history,
    }


def main():
    parser = argparse.ArgumentParser(description="Run LLM-ReAct PSO experiment")
    parser.add_argument("--func", default="sphere", choices=["sphere", "rastrigin", "ackley"])
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--mode", default="llm_react",
                        choices=["rule", "llm", "llm_react"])
    parser.add_argument("--provider", default="deepseek",
                        choices=["deepseek", "yunwu", "kimi"])
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--stagnation-threshold", type=int, default=30)
    parser.add_argument("--react-mode", default="basic",
                        choices=["basic", "deep"])
    parser.add_argument("--output", default=None, help="JSON output path")
    args = parser.parse_args()

    result = run_experiment(
        func_name=args.func,
        dim=args.dim,
        mode=args.mode,
        provider=args.provider,
        pop_size=args.pop_size,
        max_iter=args.max_iter,
        stagnation_threshold=args.stagnation_threshold,
        react_mode=args.react_mode,
    )

    if result and args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Keep only essential data
        save_data = {k: v for k, v in result.items() if k != "action_history"}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
