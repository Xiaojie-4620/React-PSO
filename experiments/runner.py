"""Multi-trial experiment runner with parallel execution."""

import json
import time
import multiprocessing as mp
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .configs import ExperimentConfig


@dataclass
class TrialResult:
    """Result of a single optimization run."""
    func_name: str
    dim: int
    mode: str
    seed: int
    trial: int
    history: List[float]
    final_cost: float
    n_interventions: int
    n_llm_calls: int
    wall_time: float
    success: bool
    error: Optional[str] = None


@dataclass
class ExperimentReport:
    """Aggregated results for one (function, dim, mode) combination."""
    config_name: str
    func_name: str
    dim: int
    mode: str
    n_trials: int
    trials: List[TrialResult] = field(default_factory=list)
    final_mean: float = 0.0
    final_std: float = 0.0
    final_median: float = 0.0
    final_best: float = 0.0
    final_worst: float = 0.0
    avg_interventions: float = 0.0
    avg_llm_calls: float = 0.0
    avg_wall_time: float = 0.0

    def compute_statistics(self):
        if not self.trials:
            return
        finals = np.array([t.final_cost for t in self.trials])
        self.final_mean = float(np.mean(finals))
        self.final_std = float(np.std(finals, ddof=1))
        self.final_median = float(np.median(finals))
        self.final_best = float(np.min(finals))
        self.final_worst = float(np.max(finals))
        self.avg_interventions = float(np.mean([t.n_interventions for t in self.trials]))
        self.avg_llm_calls = float(np.mean([t.n_llm_calls for t in self.trials]))
        self.avg_wall_time = float(np.mean([t.wall_time for t in self.trials]))


class ExperimentRunner:
    """Runs multi-trial experiments with parallel execution support."""

    def __init__(
        self,
        config: ExperimentConfig,
        build_optimizer: Callable,
        n_workers: int = 0,
        verbose: bool = True,
    ):
        """
        Args:
            config: Experiment configuration.
            build_optimizer: Callable(func, dim, mode, seed) -> optimizer.
                The returned optimizer must have a run() -> np.ndarray method.
            n_workers: Number of parallel workers. 0 = CPU count.
            verbose: Print progress
        """
        self.config = config
        self.build_optimizer = build_optimizer
        self.n_workers = n_workers or mp.cpu_count()
        self.verbose = verbose
        self._reports: List[ExperimentReport] = []

    def run(self) -> List[ExperimentReport]:
        tasks = []
        for func_name, func in self.config.functions:
            for dim in self.config.dimensions:
                for mode in self.config.modes:
                    for trial in range(self.config.n_trials):
                        seed = self._get_seed(trial)
                        tasks.append((func_name, func, dim, mode, trial, seed))

        if self.verbose:
            print(f"Running {len(tasks)} tasks with {self.n_workers} workers...")

        start = time.time()

        if self.n_workers > 1 and len(tasks) > 1:
            with mp.Pool(self.n_workers) as pool:
                results = pool.map(self._run_single, tasks)
        else:
            results = [self._run_single(t) for t in tasks]

        elapsed = time.time() - start
        if self.verbose:
            print(f"Completed in {elapsed:.1f}s")

        self._reports = self._aggregate(results)
        return self._reports

    def _run_single(self, task) -> TrialResult:
        func_name, func, dim, mode, trial, seed = task
        np.random.seed(seed)
        t0 = time.time()

        try:
            optimizer = self.build_optimizer(func, dim, mode, seed)
            history = optimizer.run()
            history_arr = np.asarray(history, dtype=float)

            n_interventions = 0
            n_llm_calls = 0
            if hasattr(optimizer, "action_history"):
                ah = optimizer.action_history
                n_interventions = len(ah)
                n_llm_calls = sum(1 for a in ah if a.get("mode") in ("llm", "llm_react"))

            return TrialResult(
                func_name=func_name,
                dim=dim,
                mode=mode,
                seed=seed,
                trial=trial,
                history=history_arr.tolist(),
                final_cost=float(history_arr[-1]),
                n_interventions=n_interventions,
                n_llm_calls=n_llm_calls,
                wall_time=time.time() - t0,
                success=True,
            )
        except Exception as exc:
            return TrialResult(
                func_name=func_name,
                dim=dim,
                mode=mode,
                seed=seed,
                trial=trial,
                history=[],
                final_cost=float("inf"),
                n_interventions=0,
                n_llm_calls=0,
                wall_time=time.time() - t0,
                success=False,
                error=str(exc),
            )

    def _aggregate(self, results: List[TrialResult]) -> List[ExperimentReport]:
        groups: Dict[tuple, List[TrialResult]] = {}
        for r in results:
            key = (r.func_name, r.dim, r.mode)
            groups.setdefault(key, []).append(r)

        reports = []
        for (func_name, dim, mode), trials in sorted(groups.items()):
            report = ExperimentReport(
                config_name=self.config.name,
                func_name=func_name,
                dim=dim,
                mode=mode,
                n_trials=len(trials),
                trials=sorted(trials, key=lambda t: t.trial),
            )
            report.compute_statistics()
            reports.append(report)
        return reports

    def save_results(self, output_dir: Optional[str] = None):
        output_dir = Path(output_dir or self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for report in self._reports:
            filename = output_dir / f"{report.config_name}_{report.func_name}_d{report.dim}_{report.mode}.json"
            data = {
                "config_name": report.config_name,
                "func_name": report.func_name,
                "dim": report.dim,
                "mode": report.mode,
                "n_trials": report.n_trials,
                "statistics": {
                    "final_mean": report.final_mean,
                    "final_std": report.final_std,
                    "final_median": report.final_median,
                    "final_best": report.final_best,
                    "final_worst": report.final_worst,
                    "avg_interventions": report.avg_interventions,
                    "avg_llm_calls": report.avg_llm_calls,
                    "avg_wall_time": report.avg_wall_time,
                },
                "trials": [
                    {
                        "trial": t.trial,
                        "seed": t.seed,
                        "final_cost": t.final_cost,
                        "n_interventions": t.n_interventions,
                        "n_llm_calls": t.n_llm_calls,
                        "wall_time": t.wall_time,
                        "success": t.success,
                        "error": t.error,
                    }
                    for t in report.trials
                ],
            }
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        if self.verbose:
            print(f"Saved {len(self._reports)} reports to {output_dir}")

    def _get_seed(self, trial: int) -> int:
        if self.config.seeds and trial < len(self.config.seeds):
            return self.config.seeds[trial]
        base = hash(self.config.name) & 0x7FFFFFFF
        return base + trial * 1009
