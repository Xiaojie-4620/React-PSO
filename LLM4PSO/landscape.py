"""Fitness Landscape Analysis for PSO state diagnosis.

Provides quantitative characterization of the local fitness landscape:
ruggedness, information content, gradient structure, basin size, deceptiveness.

These features enable the ReAct agent to reason about *why* the swarm is stuck,
rather than just *that* it is stuck — the core innovation of Landscape-aware PSO.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class LandscapeProfile:
    """Compact description of the local fitness landscape around a swarm."""

    ruggedness: float = 0.0
    information_content: float = 0.0
    gradient_magnitude_mean: float = 0.0
    gradient_magnitude_std: float = 0.0
    estimated_basin_radius: float = 0.0
    deceptiveness: float = 0.0
    modality_estimate: str = "unknown"
    separability_estimate: str = "unknown"
    landscape_label: str = "unclassified"

    def to_dict(self) -> Dict[str, object]:
        return {
            "ruggedness": round(self.ruggedness, 4),
            "information_content": round(self.information_content, 4),
            "gradient_magnitude_mean": round(self.gradient_magnitude_mean, 4),
            "gradient_magnitude_std": round(self.gradient_magnitude_std, 4),
            "estimated_basin_radius": round(self.estimated_basin_radius, 4),
            "deceptiveness": round(self.deceptiveness, 4),
            "modality_estimate": self.modality_estimate,
            "separability_estimate": self.separability_estimate,
            "landscape_label": self.landscape_label,
        }

    def summary(self) -> str:
        """Human-readable one-line description."""
        return (
            f"Landscape: {self.landscape_label} | "
            f"ruggedness={self.ruggedness:.3f}, "
            f"info_content={self.information_content:.3f}, "
            f"gradient={self.gradient_magnitude_mean:.3f}, "
            f"basin={self.estimated_basin_radius:.1f}, "
            f"deceptiveness={self.deceptiveness:.3f}"
        )


class LandscapeAnalyzer:
    """Analyzes the fitness landscape around a swarm using sampling-based methods.

    Usage:
        analyzer = LandscapeAnalyzer(func, bounds=(-100, 100))
        profile = analyzer.analyze(best_position, swarm_bbox)
        # profile can be fed into SwarmStateAnalyzer and ReAct prompts
    """

    def __init__(
        self,
        func: Callable,
        dim: int,
        bounds: Tuple[float, float] = (-100.0, 100.0),
        n_samples: int = 200,
        n_neighbors: int = 5,
        basin_step_ratio: float = 0.01,
        basin_max_steps: int = 50,
        basin_threshold_ratio: float = 0.1,
        random_seed: int = 0,
    ):
        """
        Args:
            func: Objective function, vectorized: func(M, N) -> (M,).
            dim: Problem dimension.
            bounds: (low, high) search space bounds.
            n_samples: Number of LHS samples for ruggedness/gradient estimation.
            n_neighbors: Number of nearest neighbors for local gradient estimation.
            basin_step_ratio: Step size as fraction of search span for basin radius walk.
            basin_max_steps: Maximum steps for basin radius random walks.
            basin_threshold_ratio: Fitness threshold ratio for leaving basin.
        """
        self.func = func
        self.dim = dim
        self.low = float(bounds[0])
        self.high = float(bounds[1])
        self.span = self.high - self.low
        self.n_samples = n_samples
        self.n_neighbors = n_neighbors
        self.basin_step_ratio = basin_step_ratio
        self.basin_max_steps = basin_max_steps
        self.basin_threshold_ratio = basin_threshold_ratio
        self._rng = np.random.RandomState(random_seed)
        self._cache: Dict[Tuple, LandscapeProfile] = {}

    def analyze(
        self,
        best_position: np.ndarray,
        swarm_bbox: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        gbest: Optional[float] = None,
        force_recompute: bool = False,
    ) -> LandscapeProfile:
        """Compute landscape profile for the current search region.

        Args:
            best_position: Current global best position (dim,).
            swarm_bbox: (low, high) bounding box of swarm positions, each shape (dim,).
                        If None, uses the full search bounds.
            gbest: Current global best fitness (used for cache key).
            force_recompute: Skip cache lookup.

        Returns:
            LandscapeProfile with all computed features.
        """
        best_position = np.asarray(best_position, dtype=float).reshape(self.dim)

        if swarm_bbox is None:
            bbox_low = np.full(self.dim, self.low)
            bbox_high = np.full(self.dim, self.high)
        else:
            bbox_low = np.asarray(swarm_bbox[0], dtype=float).reshape(self.dim)
            bbox_high = np.asarray(swarm_bbox[1], dtype=float).reshape(self.dim)
            # Expand slightly to capture surrounding landscape
            margin = 0.2 * (bbox_high - bbox_low)
            bbox_low = np.maximum(self.low, bbox_low - margin)
            bbox_high = np.minimum(self.high, bbox_high + margin)

        if not force_recompute:
            cache_key = self._make_cache_key(best_position, gbest)
            if cache_key in self._cache:
                return self._cache[cache_key]

        # 1. Generate Latin Hypercube Samples within bounding box
        samples = self._latin_hypercube(self.n_samples, bbox_low, bbox_high)
        costs = self._evaluate(samples)

        # 2. Compute ruggedness (autocorrelation-based)
        ruggedness = self._compute_ruggedness(samples, costs)

        # 3. Compute information content (entropy of fitness sequence)
        info_content = self._compute_information_content(costs)

        # 4. Compute gradient magnitudes (local linear fits)
        grad_mean, grad_std = self._compute_gradient_magnitudes(samples, costs)

        # 5. Estimate basin radius around best position
        basin_radius = self._estimate_basin_radius(best_position, gbest)

        # 6. Compute deceptiveness
        deceptiveness = self._compute_deceptiveness(samples, costs, best_position, gbest)

        # 7. Classify landscape
        modality = self._estimate_modality(ruggedness, info_content, grad_std)
        separability = self._estimate_separability(samples, costs)
        label = self._classify_landscape(
            ruggedness, info_content, grad_mean, grad_std, basin_radius, deceptiveness
        )

        profile = LandscapeProfile(
            ruggedness=ruggedness,
            information_content=info_content,
            gradient_magnitude_mean=grad_mean,
            gradient_magnitude_std=grad_std,
            estimated_basin_radius=basin_radius,
            deceptiveness=deceptiveness,
            modality_estimate=modality,
            separability_estimate=separability,
            landscape_label=label,
        )

        cache_key = self._make_cache_key(best_position, gbest)
        if len(self._cache) > 50:
            # Evict oldest entry
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = profile

        return profile

    # ------------------------------------------------------------------
    # Public sampling utility (usable outside the class)
    # ------------------------------------------------------------------

    def sample_landscape(
        self, n: int = 500, bbox: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (samples, costs) for the given region. Useful for visualization."""
        if bbox is None:
            low = np.full(self.dim, self.low)
            high = np.full(self.dim, self.high)
        else:
            low = np.asarray(bbox[0], dtype=float).reshape(self.dim)
            high = np.asarray(bbox[1], dtype=float).reshape(self.dim)
        samples = self._latin_hypercube(n, low, high)
        costs = self._evaluate(samples)
        return samples, costs

    # ------------------------------------------------------------------
    # Internal: Latin Hypercube Sampling
    # ------------------------------------------------------------------

    def _latin_hypercube(
        self, n: int, low: np.ndarray, high: np.ndarray
    ) -> np.ndarray:
        """Generate n Latin Hypercube samples in [low, high]."""
        segments = (np.arange(n)[:, None] + self._rng.uniform(0, 1, size=(n, self.dim))) / n
        for d in range(self.dim):
            self._rng.shuffle(segments[:, d])
        return low + segments * (high - low)

    # ------------------------------------------------------------------
    # Internal: evaluate samples
    # ------------------------------------------------------------------

    def _evaluate(self, samples: np.ndarray) -> np.ndarray:
        """Evaluate objective function on sample array (n, dim) -> (n,)."""
        try:
            result = self.func(samples)
        except Exception:
            result = np.array([self.func.evaluate(s) for s in samples])
        return np.asarray(result, dtype=float).ravel()

    # ------------------------------------------------------------------
    # Ruggedness: autocorrelation decay along distance-ordered samples
    # ------------------------------------------------------------------

    def _compute_ruggedness(
        self, samples: np.ndarray, costs: np.ndarray
    ) -> float:
        """Estimate ruggedness from the autocorrelation of costs along
        a distance-ordered path through the samples.

        Low autocorrelation at small lags → high ruggedness.
        Returns value in [0, 1] where higher = more rugged.
        """
        n = len(costs)
        if n < 10:
            return 0.5

        center = np.mean(samples, axis=0)
        distances = np.linalg.norm(samples - center, axis=1)
        order = np.argsort(distances)
        ordered = costs[order]

        # Standardize
        ordered = (ordered - np.mean(ordered)) / max(np.std(ordered), 1e-12)

        # Fit and remove a linear trend to isolate the stochastic component
        x = np.arange(len(ordered))
        slope = np.polyfit(x, ordered, 1)[0]
        detrended = ordered - slope * (x - x.mean())

        # Compute autocorrelation at lag 1 on the detrended signal
        if len(detrended) < 3:
            return 0.5
        r1 = np.corrcoef(detrended[:-1], detrended[1:])[0, 1]
        r1 = float(max(-1.0, min(1.0, r1)) if np.isfinite(r1) else 0.0)

        # Ruggedness = 1 - |autocorr| — less correlated = more rugged
        return float(np.clip(1.0 - abs(r1), 0.0, 1.0))

    # ------------------------------------------------------------------
    # Information Content: entropy of fitness-change patterns
    # ------------------------------------------------------------------

    def _compute_information_content(self, costs: np.ndarray) -> float:
        """Compute entropy-based information content of the fitness sequence.

        Encodes fitness changes along distance-ordered samples and measures
        the entropy of the resulting symbol sequence.
        High entropy → more structure / information in the landscape.
        Returns [0, 1].
        """
        n = len(costs)
        if n < 6:
            return 0.5

        center = np.mean(costs)
        scale = max(np.std(costs), 1e-12)
        normalized = (costs - center) / scale

        # Quantize into 5 levels
        bins = np.digitize(normalized, bins=np.array([-1.5, -0.5, 0.5, 1.5]))
        # Encode 3-point patterns
        patterns = []
        for i in range(len(bins) - 2):
            pattern = (bins[i], bins[i + 1], bins[i + 2])
            patterns.append(pattern)

        if not patterns:
            return 0.5

        unique, counts = np.unique(
            [hash(p) for p in patterns], return_counts=True
        )
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log2(np.maximum(probs, 1e-12)))
        max_entropy = np.log2(min(len(patterns), 125))  # 5^3 = 125 max patterns
        return float(np.clip(entropy / max(max_entropy, 1e-12), 0.0, 1.0))

    # ------------------------------------------------------------------
    # Gradient Magnitudes: local linear regression
    # ------------------------------------------------------------------

    def _compute_gradient_magnitudes(
        self, samples: np.ndarray, costs: np.ndarray
    ) -> Tuple[float, float]:
        """Estimate gradient magnitudes via local k-NN linear fits.

        Returns (mean_gradient_magnitude, std_gradient_magnitude).
        Magnitudes are normalized by the fitness range.
        """
        n = len(samples)
        if n < self.n_neighbors + 2:
            return 0.0, 0.0

        k = min(self.n_neighbors, n - 1)
        fitness_range = max(np.ptp(costs), 1e-12)
        gradients = []

        # Sample a subset for efficiency
        n_eval = min(50, n)
        indices = self._rng.choice(n, size=n_eval, replace=False)

        for idx in indices:
            point = samples[idx]
            distances = np.linalg.norm(samples - point, axis=1)
            nn_idx = np.argpartition(distances, k + 1)[: k + 1]
            nn_idx = nn_idx[nn_idx != idx][:k]

            if len(nn_idx) < 2:
                continue

            X = samples[nn_idx] - point
            y = costs[nn_idx] - costs[idx]

            try:
                grad, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                grad_mag = np.linalg.norm(grad)
                gradients.append(grad_mag / fitness_range)
            except np.linalg.LinAlgError:
                continue

        if not gradients:
            return 0.0, 0.0

        return float(np.mean(gradients)), float(np.std(gradients))

    # ------------------------------------------------------------------
    # Basin Radius: random walk until fitness increases significantly
    # ------------------------------------------------------------------

    def _estimate_basin_radius(
        self, best_position: np.ndarray, gbest: Optional[float]
    ) -> float:
        """Estimate the radius of the current basin of attraction.

        From the best position, walks in random directions until fitness
        increases significantly, indicating departure from the basin.
        """
        best_position = np.asarray(best_position, dtype=float).ravel()
        if gbest is None:
            gbest = float(self._evaluate(best_position.reshape(1, -1))[0])

        step_size = self.basin_step_ratio * self.span
        threshold = self.basin_threshold_ratio * abs(gbest) + 1e-8

        radii = []
        n_walks = min(10, self.dim)

        for _ in range(n_walks):
            direction = self._rng.normal(0, 1, size=self.dim)
            direction /= max(np.linalg.norm(direction), 1e-12)

            current = best_position.copy()
            for step in range(self.basin_max_steps):
                current = current + direction * step_size
                current = np.clip(current, self.low, self.high)

                try:
                    f_current = float(self._evaluate(current.reshape(1, -1))[0])
                except Exception:
                    break

                if f_current - gbest > threshold:
                    radii.append(step * step_size)
                    break
            else:
                radii.append(self.basin_max_steps * step_size)

        return float(np.mean(radii)) if radii else self.span

    # ------------------------------------------------------------------
    # Deceptiveness: gradient vs. global optimum direction
    # ------------------------------------------------------------------

    def _compute_deceptiveness(
        self,
        samples: np.ndarray,
        costs: np.ndarray,
        best_position: np.ndarray,
        gbest: Optional[float],
    ) -> float:
        """Measure how deceptive the landscape is around the current best.

        Compares the local gradient direction at the best position with
        the direction toward the best sample point.
        High deceptiveness → gradient points away from promising regions.
        """
        best_position = np.asarray(best_position, dtype=float).ravel()
        n = len(samples)
        if n < self.n_neighbors + 2:
            return 0.0

        # Find gradient at best position
        k = min(self.n_neighbors, n - 1)
        distances = np.linalg.norm(samples - best_position, axis=1)
        nn_idx = np.argpartition(distances, k + 1)[: k + 1]
        nn_idx = nn_idx[distances[nn_idx] > 1e-12][:k]

        if len(nn_idx) < 2:
            return 0.0

        X = samples[nn_idx] - best_position
        y = costs[nn_idx] - (gbest if gbest is not None else costs[nn_idx].min())

        try:
            grad, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-12:
                return 0.0
            grad_dir = grad / grad_norm
        except np.linalg.LinAlgError:
            return 0.0

        # Direction toward best sample in the neighborhood
        best_sample_idx = nn_idx[np.argmin(costs[nn_idx])]
        toward_best = samples[best_sample_idx] - best_position
        toward_norm = np.linalg.norm(toward_best)
        if toward_norm < 1e-12:
            return 0.0
        toward_dir = toward_best / toward_norm

        # Deceptiveness = 1 - cosine_similarity (negative cosine → deceptive)
        cosine = float(np.dot(grad_dir, toward_dir))
        return float(np.clip((1.0 - cosine) / 2.0, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Landscape classification
    # ------------------------------------------------------------------

    def _estimate_modality(
        self, ruggedness: float, info_content: float, grad_std: float
    ) -> str:
        if ruggedness < 0.3 and info_content < 0.4:
            return "unimodal"
        if ruggedness > 0.7 or info_content > 0.7:
            return "multimodal_many"
        return "multimodal_few"

    def _estimate_separability(
        self, samples: np.ndarray, costs: np.ndarray
    ) -> str:
        """Heuristic: if pairwise interactions are weak, the function is more separable."""
        n = len(samples)
        if n < 20 or self.dim < 2:
            return "unknown"

        # Compare variance explained by individual dimensions vs. interaction
        subset = samples[:min(n, 100)]
        c = costs[:min(n, 100)]
        c_centered = c - np.mean(c)

        var_explained = 0.0
        for d in range(min(self.dim, 10)):
            corr = np.corrcoef(subset[:, d], c_centered)[0, 1]
            var_explained += abs(corr) if np.isfinite(corr) else 0.0

        avg_var = var_explained / min(self.dim, 10)
        if avg_var > 0.5:
            return "separable"
        if avg_var > 0.2:
            return "partially_separable"
        return "non_separable"

    def _classify_landscape(
        self,
        ruggedness: float,
        info_content: float,
        grad_mean: float,
        grad_std: float,
        basin_radius: float,
        deceptiveness: float,
    ) -> str:
        """Assign a human-readable landscape label."""
        basin_frac = basin_radius / max(self.span, 1e-12)

        if deceptiveness > 0.6:
            return "deceptive_valley"
        if ruggedness > 0.7 and info_content > 0.6:
            return "rugged_multi_basin"
        if ruggedness > 0.7 and grad_mean < 0.1:
            return "rugged_plateau"
        if grad_mean < 0.05 and info_content < 0.3:
            return "flat_plain"
        if basin_frac < 0.05 and ruggedness > 0.5:
            return "needle_haystack"
        if grad_mean > 0.3 and basin_frac > 0.2:
            return "broad_valley"
        if info_content > 0.5 and ruggedness > 0.5:
            return "complex_terrain"
        return "moderate_hills"

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _make_cache_key(
        self, position: np.ndarray, gbest: Optional[float]
    ) -> Tuple:
        """Create a cache key from position (rounded) and fitness."""
        rounded = tuple(np.round(position, decimals=4))
        fitness_key = round(gbest, 4) if gbest is not None else 0.0
        return rounded + (fitness_key,)
