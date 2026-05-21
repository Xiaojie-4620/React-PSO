"""Statistical metrics for comparing optimization results."""

import numpy as np
from typing import Dict, List, Optional, Tuple


def convergence_speed(history: np.ndarray, target: float, max_iters: int) -> int:
    """Iterations needed to reach a target fitness value.

    Args:
        history: 1D array of best cost per iteration.
        target: Target fitness value to reach.
        max_iters: Maximum iterations (returned if target never reached).

    Returns:
        First iteration where cost <= target, or max_iters if never reached.
    """
    reached = np.where(history <= target)[0]
    return int(reached[0]) if reached.size > 0 else max_iters


def convergence_auc(history: np.ndarray, normalize: bool = True) -> float:
    """Area under the convergence curve (lower is better).

    Args:
        history: 1D array of best cost per iteration.
        normalize: If True, divide by (max_iters * max_cost) to get [0, 1].

    Returns:
        AUC value.
    """
    auc = float(np.trapz(history))
    if normalize and len(history) > 1:
        max_possible = len(history) * np.max(history)
        auc /= max(max_possible, 1e-12)
    return auc


def success_rate(histories: List[np.ndarray], target: float) -> float:
    """Fraction of runs that reached the target fitness."""
    if not histories:
        return 0.0
    successes = sum(1 for h in histories if np.min(h) <= target)
    return successes / len(histories)


def final_statistics(histories: List[np.ndarray]) -> Dict[str, float]:
    """Compute mean, std, median, best, worst of final values across trials."""
    finals = np.array([h[-1] for h in histories])
    return {
        "mean": float(np.mean(finals)),
        "std": float(np.std(finals, ddof=1)),
        "median": float(np.median(finals)),
        "best": float(np.min(finals)),
        "worst": float(np.max(finals)),
    }


def wilcoxon_test(
    a: List[np.ndarray], b: List[np.ndarray]
) -> Dict[str, float]:
    """Wilcoxon signed-rank test between two sets of trials.

    Uses final best costs as paired observations.
    """
    from scipy.stats import wilcoxon

    a_finals = np.array([h[-1] for h in a])
    b_finals = np.array([h[-1] for h in b])
    statistic, p_value = wilcoxon(a_finals, b_finals, zero_method="zsplit")
    return {"statistic": float(statistic), "p_value": float(p_value)}


def friedman_ranking(
    results: Dict[str, List[np.ndarray]]
) -> Dict[str, float]:
    """Friedman test + average rankings across methods.

    Args:
        results: {method_name: [history1, history2, ...]}.

    Returns:
        {method_name: average_rank}
    """
    from scipy.stats import friedmanchisquare

    method_names = list(results.keys())
    n_trials = min(len(v) for v in results.values())
    if n_trials < 2 or len(method_names) < 2:
        return {n: 0.0 for n in method_names}

    # Each row = one trial, each column = one method
    data = np.array([
        [results[m][t][-1] for m in method_names]
        for t in range(n_trials)
    ])

    ranks = np.zeros_like(data, dtype=float)
    for i in range(n_trials):
        ranks[i] = data[i].argsort().argsort() + 1

    avg_ranks = {n: float(np.mean(ranks[:, j])) for j, n in enumerate(method_names)}

    try:
        friedmanchisquare(*[data[:, j] for j in range(len(method_names))])
    except Exception:
        pass

    return avg_ranks


def cohens_d(a: List[np.ndarray], b: List[np.ndarray]) -> float:
    """Cohen's d effect size between two methods (final costs)."""
    a_finals = np.array([h[-1] for h in a])
    b_finals = np.array([h[-1] for h in b])
    diff = np.mean(a_finals) - np.mean(b_finals)
    pooled_std = np.sqrt((np.var(a_finals, ddof=1) + np.var(b_finals, ddof=1)) / 2)
    return float(diff / max(pooled_std, 1e-12))
