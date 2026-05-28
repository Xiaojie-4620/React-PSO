from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class SwarmState:
    iteration: int
    gbest: float
    previous_gbest: Optional[float]
    best_delta: float
    relative_improvement: float
    no_improve_iters: int
    diversity: float
    normalized_diversity: float
    position_std_mean: float
    swarm_radius: float
    velocity_norm_mean: float
    velocity_norm_std: float
    velocity_zero_ratio: float
    velocity_direction_consistency: float
    fitness_best: float
    fitness_mean: float
    fitness_worst: float
    fitness_std: float
    fitness_cv: float
    boundary_hit_ratio: float
    velocity_clip_ratio: float
    state_label: str
    reasons: Tuple[str, ...]

    def to_prompt_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["reasons"] = list(self.reasons)
        return data


class SwarmStateAnalyzer:
    """Convert raw PSO arrays into compact, explainable search-state metrics."""

    def __init__(
        self,
        position_bounds: Sequence[Any],
        velocity_bounds: Sequence[Any],
        dim: int,
        improvement_tolerance: float = 1e-3,
        stagnation_window: int = 10,
        diversity_low_ratio: float = 0.01,
        diversity_high_ratio: float = 0.10,
        boundary_epsilon_ratio: float = 1e-6,
    ):
        self.dim = dim
        self.improvement_tolerance = improvement_tolerance
        self.stagnation_window = max(1, int(stagnation_window))
        self.diversity_low_ratio = diversity_low_ratio
        self.diversity_high_ratio = diversity_high_ratio
        self.position_low, self.position_high = self._normalize_bounds(position_bounds)
        self.velocity_low, self.velocity_high = self._normalize_bounds(velocity_bounds)
        self.search_span = np.maximum(self.position_high - self.position_low, 1e-12)
        self.velocity_span = np.maximum(self.velocity_high - self.velocity_low, 1e-12)
        self.search_diameter = float(np.linalg.norm(self.search_span))
        self.velocity_diameter = float(np.linalg.norm(self.velocity_span))
        self.boundary_epsilon = np.maximum(self.search_span * boundary_epsilon_ratio, 1e-12)
        self.velocity_epsilon = np.maximum(self.velocity_span * boundary_epsilon_ratio, 1e-12)
        # Emergency thresholds
        self.diversity_collapse_threshold = 0.005
        self.velocity_death_threshold = 0.85
        self.velocity_frozen_threshold = 0.7
        self.fitness_cv_multimodal_threshold = 0.05

    def analyze(self, pso, iteration: int, history: Sequence[float], no_improve_iters: int, landscape=None) -> SwarmState:
        position = np.asarray(pso.position, dtype=float)
        velocity = np.asarray(pso.velocity, dtype=float)
        cost = np.asarray(pso.cost, dtype=float)

        gbest = float(pso.get_gBest())
        previous_gbest = float(history[-2]) if len(history) >= 2 else None
        best_delta = 0.0 if previous_gbest is None else previous_gbest - gbest
        relative_improvement = best_delta / max(abs(previous_gbest or 0.0), 1e-12)

        center = np.mean(position, axis=0)
        distances = np.linalg.norm(position - center, axis=1)
        diversity = float(np.mean(distances))
        normalized_diversity = diversity / max(self.search_diameter, 1e-12)
        position_std_mean = float(np.mean(np.std(position, axis=0)))
        swarm_radius = float(np.max(distances)) if distances.size else 0.0

        velocity_norm = np.linalg.norm(velocity, axis=1)
        velocity_norm_mean = float(np.mean(velocity_norm))
        velocity_norm_std = float(np.std(velocity_norm))
        velocity_zero_threshold = max(self.velocity_diameter * 1e-6, 1e-12)
        velocity_zero_ratio = float(np.mean(velocity_norm <= velocity_zero_threshold))
        velocity_direction_consistency = self._direction_consistency(velocity, velocity_norm, velocity_zero_threshold)

        fitness_best = float(np.min(cost))
        fitness_mean = float(np.mean(cost))
        fitness_worst = float(np.max(cost))
        fitness_std = float(np.std(cost))
        fitness_cv = fitness_std / max(abs(fitness_mean), 1e-12)

        lower_hits = position <= (self.position_low + self.boundary_epsilon)
        upper_hits = position >= (self.position_high - self.boundary_epsilon)
        boundary_hit_ratio = float(np.mean(lower_hits | upper_hits))

        low_clip = velocity <= (self.velocity_low + self.velocity_epsilon)
        high_clip = velocity >= (self.velocity_high - self.velocity_epsilon)
        velocity_clip_ratio = float(np.mean(low_clip | high_clip))

        state_label, reasons = self._classify(
            no_improve_iters=no_improve_iters,
            normalized_diversity=normalized_diversity,
            velocity_zero_ratio=velocity_zero_ratio,
            velocity_direction_consistency=velocity_direction_consistency,
            boundary_hit_ratio=boundary_hit_ratio,
            velocity_clip_ratio=velocity_clip_ratio,
            relative_improvement=relative_improvement,
            fitness_cv=fitness_cv,
            landscape=landscape,
        )

        return SwarmState(
            iteration=iteration,
            gbest=gbest,
            previous_gbest=previous_gbest,
            best_delta=float(best_delta),
            relative_improvement=float(relative_improvement),
            no_improve_iters=int(no_improve_iters),
            diversity=diversity,
            normalized_diversity=float(normalized_diversity),
            position_std_mean=position_std_mean,
            swarm_radius=swarm_radius,
            velocity_norm_mean=velocity_norm_mean,
            velocity_norm_std=velocity_norm_std,
            velocity_zero_ratio=velocity_zero_ratio,
            velocity_direction_consistency=velocity_direction_consistency,
            fitness_best=fitness_best,
            fitness_mean=fitness_mean,
            fitness_worst=fitness_worst,
            fitness_std=fitness_std,
            fitness_cv=float(fitness_cv),
            boundary_hit_ratio=boundary_hit_ratio,
            velocity_clip_ratio=velocity_clip_ratio,
            state_label=state_label,
            reasons=tuple(reasons),
        )

    def _normalize_bounds(self, bounds: Sequence[Any]) -> Tuple[np.ndarray, np.ndarray]:
        low = np.asarray(bounds[0], dtype=float)
        high = np.asarray(bounds[1], dtype=float)
        if low.ndim == 0:
            low = np.full(self.dim, float(low))
        if high.ndim == 0:
            high = np.full(self.dim, float(high))
        return low.reshape(self.dim), high.reshape(self.dim)

    @staticmethod
    def _direction_consistency(velocity: np.ndarray, velocity_norm: np.ndarray, threshold: float) -> float:
        active = velocity_norm > threshold
        if np.count_nonzero(active) < 2:
            return 0.0
        unit_velocity = velocity[active] / velocity_norm[active, None]
        cosine = unit_velocity @ unit_velocity.T
        upper = cosine[np.triu_indices_from(cosine, k=1)]
        if upper.size == 0:
            return 0.0
        return float(np.mean(np.abs(upper)))

    def _classify(
        self,
        no_improve_iters: int,
        normalized_diversity: float,
        velocity_zero_ratio: float,
        velocity_direction_consistency: float,
        boundary_hit_ratio: float,
        velocity_clip_ratio: float,
        relative_improvement: float,
        fitness_cv: float,
        landscape=None,
    ) -> Tuple[str, List[str]]:
        reasons: List[str] = []

        # Build informational reasons
        stalled = no_improve_iters >= self.stagnation_window
        if stalled:
            reasons.append(f"no improvement for {no_improve_iters} iterations")
        if normalized_diversity <= self.diversity_low_ratio:
            reasons.append(f"low normalized diversity ({normalized_diversity:.5f})")
        if velocity_zero_ratio >= 0.5:
            reasons.append(f"many particles have near-zero velocity ({velocity_zero_ratio:.2f})")
        if boundary_hit_ratio >= 0.15:
            reasons.append("many coordinates are on search boundaries")
        if velocity_clip_ratio >= 0.15:
            reasons.append("many velocity components are clipped")
        if velocity_direction_consistency >= 0.85:
            reasons.append("particle velocities are highly aligned")

        improving = relative_improvement > self.improvement_tolerance

        # === Layer 1: Emergency signals (NO stall requirement) ===
        if normalized_diversity <= self.diversity_collapse_threshold and velocity_zero_ratio >= 0.7:
            reasons.append("EMERGENCY: diversity collapsed with frozen particles")
            return "diversity_collapse", reasons

        if velocity_zero_ratio >= self.velocity_death_threshold and normalized_diversity > self.diversity_collapse_threshold:
            reasons.append("EMERGENCY: particles frozen but still spread out")
            return "velocity_death", reasons

        # === Layer 2: Landscape-aware classification (takes priority when data available) ===
        if landscape is not None:
            label = self._classify_with_landscape(
                stalled, normalized_diversity, velocity_zero_ratio,
                boundary_hit_ratio, fitness_cv, landscape, reasons,
            )
            if label is not None:
                return label, reasons

        # === Layer 3: Stalled states with discriminating conditions ===
        if stalled:
            # 3a: Boundary stagnation
            if boundary_hit_ratio >= 0.15:
                return "boundary_stagnation", reasons

            # 3b: Diverse but stalled — multimodal trap detected BEFORE full collapse
            if normalized_diversity >= 0.03 and fitness_cv >= self.fitness_cv_multimodal_threshold:
                reasons.append("swarm is diverse but cannot improve — likely multimodal trap")
                return "multimodal_diverse_stall", reasons

            # 3c: Full premature convergence (low diversity AND frozen)
            if normalized_diversity <= self.diversity_low_ratio and velocity_zero_ratio >= 0.5:
                return "premature_convergence", reasons

            # 3d: Velocity collapse WITHOUT diversity collapse
            #     NOW reachable: requires frozen velocity BUT diversity still above low threshold
            if velocity_zero_ratio >= self.velocity_frozen_threshold:
                reasons.append("particles frozen but diversity still present")
                return "velocity_collapse", reasons

            # 3e: Generic slow stagnation (diversity and velocity both moderate)
            return "slow_stagnation", reasons

        # === Layer 4: Normal (non-stalled) states ===
        if improving:
            return "normal_search", reasons
        if normalized_diversity <= self.diversity_low_ratio:
            return "normal_convergence", reasons
        return "normal_search", reasons

    def _classify_with_landscape(
        self, stalled, normalized_diversity, velocity_zero_ratio,
        boundary_hit_ratio, fitness_cv, landscape, reasons,
    ) -> Optional[str]:
        """Landscape-aware fine-grained classification.

        Uses LandscapeProfile features to distinguish cases that look similar
        statistically but require different interventions.
        Does NOT require 'stalled' — landscape features can identify problems early.
        """
        lp = landscape

        # Rugged plateau: high ruggedness, weak gradient
        if lp.ruggedness > 0.5 and lp.gradient_magnitude_mean < 0.15:
            reasons.append(f"rugged plateau (ruggedness={lp.ruggedness:.3f}, gradient={lp.gradient_magnitude_mean:.3f})")
            return "rugged_plateau_trap"

        # Deceptive basin: high deceptiveness
        if lp.deceptiveness > 0.4 and normalized_diversity <= 0.05:
            reasons.append(f"deceptive basin (deceptiveness={lp.deceptiveness:.3f})")
            return "deceptive_basin"

        # Deep valley: smooth with informative gradients, not stalled
        if lp.ruggedness < 0.35 and lp.gradient_magnitude_mean > 0.15 and not stalled:
            reasons.append("deep valley with informative gradients")
            return "deep_valley_chase"

        # Needle in haystack: high info content, hard to find optimum
        if lp.information_content > 0.5 and normalized_diversity <= 0.05:
            reasons.append(f"needle-in-haystack landscape (IC={lp.information_content:.3f})")
            return "needle_in_haystack"

        return None
