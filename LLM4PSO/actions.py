from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ActionResult:
    name: str
    applied: bool
    changed_particles: int
    inertia_weight: Optional[float]
    summary: str
    params: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StrategyToolbox:
    """Whitelisted interventions that can safely modify a PSO swarm."""

    def __init__(self, position_bounds: Sequence[Any], velocity_bounds: Sequence[Any], dim: int):
        self.dim = dim
        self.position_low, self.position_high = self._normalize_bounds(position_bounds)
        self.velocity_low, self.velocity_high = self._normalize_bounds(velocity_bounds)
        self.position_span = np.maximum(self.position_high - self.position_low, 1e-12)
        self.velocity_span = np.maximum(self.velocity_high - self.velocity_low, 1e-12)

    @property
    def actions(self):
        return {
            "none": self.none,
            "adjust_parameters": self.adjust_parameters,
            "reset_worst_particles": self.reset_worst_particles,
            "gaussian_mutation": self.gaussian_mutation,
            "levy_flight": self.levy_flight,
            "opposition_reinit": self.opposition_reinit,
            "basin_hopping": self.basin_hopping,
            "gradient_descent_step": self.gradient_descent_step,
            "landscape_adaptive_mutation": self.landscape_adaptive_mutation,
            "landscape_adaptive_restart": self.landscape_adaptive_restart,
        }

    def apply(self, pso, action_name: str, params: Optional[Dict[str, Any]] = None) -> ActionResult:
        params = dict(params or {})
        action = self.actions.get(action_name)
        if action is None:
            return ActionResult(
                name=action_name,
                applied=False,
                changed_particles=0,
                inertia_weight=None,
                summary=f"Unknown action '{action_name}' was ignored.",
                params=params,
            )
        return action(pso, **params)

    def none(self, pso, **params) -> ActionResult:
        return ActionResult(
            name="none",
            applied=False,
            changed_particles=0,
            inertia_weight=params.get("inertia_weight"),
            summary="No intervention was applied.",
            params=params,
        )

    def adjust_parameters(
        self,
        pso,
        inertia_weight: Optional[float] = None,
        c1: Optional[float] = None,
        c2: Optional[float] = None,
        **params,
    ) -> ActionResult:
        if c1 is not None:
            pso.c1 = self._clip_float(c1, 0.0, 4.0)
        if c2 is not None:
            pso.c2 = self._clip_float(c2, 0.0, 4.0)
        inertia = None if inertia_weight is None else self._clip_float(inertia_weight, 0.0, 2.0)
        action_params = {"inertia_weight": inertia, "c1": c1, "c2": c2, **params}
        return ActionResult(
            name="adjust_parameters",
            applied=True,
            changed_particles=0,
            inertia_weight=inertia,
            summary=f"Adjusted parameters to inertia={inertia}, c1={pso.c1}, c2={pso.c2}.",
            params=action_params,
        )

    def reset_worst_particles(
        self,
        pso,
        ratio: float = 0.25,
        reset_velocity: bool = True,
        inertia_weight: Optional[float] = 1.1,
        **params,
    ) -> ActionResult:
        indices = self._select_indices(pso, ratio=ratio, mode="worst", keep_best=True)
        if indices.size == 0:
            return self.none(pso, reason="no eligible particles")

        pso.position[indices] = np.random.uniform(self.position_low, self.position_high, size=(indices.size, self.dim))
        if reset_velocity:
            pso.velocity[indices] = np.random.uniform(self.velocity_low, self.velocity_high, size=(indices.size, self.dim))

        self._clip_swarm(pso)
        inertia = self._optional_inertia(inertia_weight)
        action_params = {"ratio": ratio, "reset_velocity": reset_velocity, "inertia_weight": inertia, **params}
        return ActionResult(
            name="reset_worst_particles",
            applied=True,
            changed_particles=int(indices.size),
            inertia_weight=inertia,
            summary=f"Reinitialized {indices.size} weak particles to restore diversity.",
            params=action_params,
        )

    def gaussian_mutation(
        self,
        pso,
        ratio: float = 0.20,
        scale: float = 0.05,
        target: str = "worst",
        inertia_weight: Optional[float] = None,
        **params,
    ) -> ActionResult:
        indices = self._select_indices(pso, ratio=ratio, mode=target, keep_best=True)
        if indices.size == 0:
            return self.none(pso, reason="no eligible particles")

        scale = self._clip_float(scale, 0.0, 1.0)
        noise = np.random.normal(0.0, scale, size=(indices.size, self.dim)) * self.position_span
        pso.position[indices] = pso.position[indices] + noise
        pso.velocity[indices] = 0.5 * pso.velocity[indices] + np.random.normal(
            0.0, scale, size=(indices.size, self.dim)
        ) * self.velocity_span

        self._clip_swarm(pso)
        inertia = self._optional_inertia(inertia_weight)
        action_params = {
            "ratio": ratio,
            "scale": scale,
            "target": target,
            "inertia_weight": inertia,
            **params,
        }
        return ActionResult(
            name="gaussian_mutation",
            applied=True,
            changed_particles=int(indices.size),
            inertia_weight=inertia,
            summary=f"Applied Gaussian mutation to {indices.size} particles.",
            params=action_params,
        )

    def levy_flight(
        self,
        pso,
        ratio: float = 0.20,
        scale: float = 0.02,
        beta: float = 1.5,
        target: str = "worst",
        inertia_weight: Optional[float] = 1.0,
        **params,
    ) -> ActionResult:
        indices = self._select_indices(pso, ratio=ratio, mode=target, keep_best=True)
        if indices.size == 0:
            return self.none(pso, reason="no eligible particles")

        scale = self._clip_float(scale, 0.0, 1.0)
        beta = self._clip_float(beta, 1.01, 2.0)
        steps = self._levy_steps(indices.size, beta)
        direction = np.random.normal(0.0, 1.0, size=(indices.size, self.dim))
        direction_norm = np.linalg.norm(direction, axis=1, keepdims=True)
        direction_norm[direction_norm == 0.0] = 1.0
        direction = direction / direction_norm

        center = np.asarray(pso.g_Best_Position, dtype=float)
        jump = direction * steps[:, None] * self.position_span * scale
        pso.position[indices] = center + jump
        pso.velocity[indices] = np.clip(jump, self.velocity_low, self.velocity_high)

        self._clip_swarm(pso)
        inertia = self._optional_inertia(inertia_weight)
        action_params = {
            "ratio": ratio,
            "scale": scale,
            "beta": beta,
            "target": target,
            "inertia_weight": inertia,
            **params,
        }
        return ActionResult(
            name="levy_flight",
            applied=True,
            changed_particles=int(indices.size),
            inertia_weight=inertia,
            summary=f"Applied Levy-flight jumps to {indices.size} particles.",
            params=action_params,
        )

    def opposition_reinit(
        self,
        pso,
        ratio: float = 0.25,
        velocity_scale: float = 0.2,
        target: str = "worst",
        inertia_weight: Optional[float] = 0.9,
        **params,
    ) -> ActionResult:
        indices = self._select_indices(pso, ratio=ratio, mode=target, keep_best=True)
        if indices.size == 0:
            return self.none(pso, reason="no eligible particles")

        opposite = self.position_low + self.position_high - pso.position[indices]
        jitter = np.random.normal(0.0, 0.01, size=(indices.size, self.dim)) * self.position_span
        pso.position[indices] = opposite + jitter
        pso.velocity[indices] = np.random.uniform(
            -velocity_scale * self.velocity_span,
            velocity_scale * self.velocity_span,
            size=(indices.size, self.dim),
        )

        self._clip_swarm(pso)
        inertia = self._optional_inertia(inertia_weight)
        action_params = {
            "ratio": ratio,
            "velocity_scale": velocity_scale,
            "target": target,
            "inertia_weight": inertia,
            **params,
        }
        return ActionResult(
            name="opposition_reinit",
            applied=True,
            changed_particles=int(indices.size),
            inertia_weight=inertia,
            summary=f"Moved {indices.size} particles to opposition-based positions.",
            params=action_params,
        )

    # ------------------------------------------------------------------
    # Landscape-adaptive actions (Phase 3)
    # ------------------------------------------------------------------

    def basin_hopping(
        self,
        pso,
        ratio: float = 0.25,
        basin_radius: float = 1.0,
        target: str = "worst",
        inertia_weight: Optional[float] = 1.0,
        **params,
    ) -> ActionResult:
        """Hop selected particles to random positions within estimated basin radius.

        More targeted than levy_flight — uses the estimated basin size to jump
        to neighboring basins rather than completely random long jumps.
        """
        indices = self._select_indices(pso, ratio=ratio, mode=target, keep_best=True)
        if indices.size == 0:
            return self.none(pso, reason="no eligible particles")

        basin_radius = self._clip_float(basin_radius, 0.0, self.position_span.max())
        radius_vec = (basin_radius / max(self.position_span.max(), 1e-12)) * self.position_span

        gbest = np.asarray(pso.g_Best_Position, dtype=float)
        direction = np.random.normal(0.0, 1.0, size=(indices.size, self.dim))
        direction_norm = np.linalg.norm(direction, axis=1, keepdims=True)
        direction_norm[direction_norm == 0.0] = 1.0
        direction = direction / direction_norm

        hop_scale = np.random.uniform(0.5, 1.5, size=(indices.size, 1)) * radius_vec
        pso.position[indices] = gbest + direction * hop_scale
        pso.velocity[indices] = np.random.uniform(
            -0.3 * self.velocity_span, 0.3 * self.velocity_span, size=(indices.size, self.dim),
        )

        self._clip_swarm(pso)
        inertia = self._optional_inertia(inertia_weight)
        return ActionResult(
            name="basin_hopping",
            applied=True,
            changed_particles=int(indices.size),
            inertia_weight=inertia,
            summary=f"Basin-hopped {indices.size} particles within radius {basin_radius:.1f}.",
            params={"ratio": ratio, "basin_radius": basin_radius, "target": target, "inertia_weight": inertia, **params},
        )

    def gradient_descent_step(
        self,
        pso,
        ratio: float = 0.20,
        step_size: float = 0.1,
        target: str = "worst",
        inertia_weight: Optional[float] = 0.8,
        **params,
    ) -> ActionResult:
        """Move selected particles along estimated local gradient direction.

        Only useful when gradient information is reliable (low ruggedness).
        """
        indices = self._select_indices(pso, ratio=ratio, mode=target, keep_best=True)
        if indices.size == 0:
            return self.none(pso, reason="no eligible particles")

        step_size = self._clip_float(step_size, 0.001, 1.0)

        # Estimate gradient for each selected particle via nearest neighbors
        all_positions = np.asarray(pso.position, dtype=float)
        all_costs = np.asarray(pso.cost, dtype=float)
        n_all = all_positions.shape[0]

        for local_i, global_i in enumerate(indices):
            distances = np.linalg.norm(all_positions - all_positions[global_i], axis=1)
            nn_idx = np.argpartition(distances, min(5, n_all))[: min(5, n_all)]
            nn_idx = nn_idx[nn_idx != global_i][:4]
            if len(nn_idx) < 2:
                continue

            X = all_positions[nn_idx] - all_positions[global_i]
            y = all_costs[nn_idx] - all_costs[global_i]
            try:
                grad, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                grad_norm = np.linalg.norm(grad)
                if grad_norm > 1e-12:
                    grad = grad / grad_norm
                pso.position[global_i] = pso.position[global_i] - step_size * self.position_span * grad
            except np.linalg.LinAlgError:
                continue

        self._clip_swarm(pso)
        inertia = self._optional_inertia(inertia_weight)
        return ActionResult(
            name="gradient_descent_step",
            applied=True,
            changed_particles=int(indices.size),
            inertia_weight=inertia,
            summary=f"Applied gradient descent step to {indices.size} particles.",
            params={"ratio": ratio, "step_size": step_size, "target": target, "inertia_weight": inertia, **params},
        )

    def landscape_adaptive_mutation(
        self,
        pso,
        ratio: float = 0.20,
        ruggedness: float = 0.5,
        target: str = "worst",
        inertia_weight: Optional[float] = None,
        **params,
    ) -> ActionResult:
        """Gaussian mutation whose scale adapts to landscape ruggedness.

        Higher ruggedness → larger mutations to cross rough terrain.
        Lower ruggedness → smaller, more precise mutations.
        """
        indices = self._select_indices(pso, ratio=ratio, mode=target, keep_best=True)
        if indices.size == 0:
            return self.none(pso, reason="no eligible particles")

        ruggedness = self._clip_float(ruggedness, 0.0, 1.0)
        # Map ruggedness to noise scale: [0.01, 0.15]
        adaptive_scale = 0.01 + ruggedness * 0.14

        noise = np.random.normal(0.0, adaptive_scale, size=(indices.size, self.dim)) * self.position_span
        pso.position[indices] = pso.position[indices] + noise
        pso.velocity[indices] = 0.5 * pso.velocity[indices] + np.random.normal(
            0.0, adaptive_scale * 0.5, size=(indices.size, self.dim)
        ) * self.velocity_span

        self._clip_swarm(pso)
        inertia = self._optional_inertia(inertia_weight)
        return ActionResult(
            name="landscape_adaptive_mutation",
            applied=True,
            changed_particles=int(indices.size),
            inertia_weight=inertia,
            summary=f"Landscape-adaptive mutation (scale={adaptive_scale:.3f}) on {indices.size} particles.",
            params={"ratio": ratio, "ruggedness": ruggedness, "target": target, "inertia_weight": inertia, **params},
        )

    def landscape_adaptive_restart(
        self,
        pso,
        ratio: float = 0.25,
        basin_radius: float = 1.0,
        reset_velocity: bool = True,
        inertia_weight: Optional[float] = 1.05,
        **params,
    ) -> ActionResult:
        """Restart particles within a region whose size adapts to the basin radius.

        Small basin → wider restart to escape.
        Large basin → tighter restart to stay in promising region.
        """
        indices = self._select_indices(pso, ratio=ratio, mode="worst", keep_best=True)
        if indices.size == 0:
            return self.none(pso, reason="no eligible particles")

        basin_radius = self._clip_float(basin_radius, 0.0, self.position_span.max())
        # Invert: small basin → large restart radius, large basin → small restart
        basin_frac = basin_radius / max(self.position_span.max(), 1e-12)
        # restart_radius ranges from 0.02 * span (large basin) to 0.5 * span (tiny basin)
        restart_frac = float(np.clip(0.5 - 0.48 * basin_frac, 0.02, 0.5))
        restart_span = restart_frac * self.position_span

        gbest = np.asarray(pso.g_Best_Position, dtype=float)
        pso.position[indices] = gbest + np.random.uniform(
            -restart_span, restart_span, size=(indices.size, self.dim),
        )
        if reset_velocity:
            pso.velocity[indices] = np.random.uniform(
                self.velocity_low * 0.3, self.velocity_high * 0.3,
                size=(indices.size, self.dim),
            )

        self._clip_swarm(pso)
        inertia = self._optional_inertia(inertia_weight)
        return ActionResult(
            name="landscape_adaptive_restart",
            applied=True,
            changed_particles=int(indices.size),
            inertia_weight=inertia,
            summary=(
                f"Landscape-adaptive restart of {indices.size} particles "
                f"(restart_frac={restart_frac:.3f}, basin_radius={basin_radius:.1f})."
            ),
            params={"ratio": ratio, "basin_radius": basin_radius, "reset_velocity": reset_velocity, "inertia_weight": inertia, **params},
        )

    # ------------------------------------------------------------------

    def _select_indices(self, pso, ratio: float, mode: str, keep_best: bool) -> np.ndarray:
        n_particles = int(np.asarray(pso.position).shape[0])
        if n_particles <= 1:
            return np.array([], dtype=int)
        count = max(1, int(np.ceil(n_particles * self._clip_float(ratio, 0.0, 1.0))))
        count = min(count, n_particles - 1 if keep_best else n_particles)

        best_idx = int(np.argmin(pso.cost)) if keep_best else None
        candidates = np.arange(n_particles)
        if keep_best:
            candidates = candidates[candidates != best_idx]

        if mode == "random":
            return np.random.choice(candidates, size=count, replace=False)

        costs = np.asarray(pso.cost, dtype=float)
        order = np.argsort(costs[candidates])
        if mode == "best":
            selected = candidates[order[:count]]
        else:
            selected = candidates[order[-count:]]
        return np.asarray(selected, dtype=int)

    def _clip_swarm(self, pso):
        pso.position = np.clip(pso.position, self.position_low, self.position_high)
        pso.velocity = np.clip(pso.velocity, self.velocity_low, self.velocity_high)

    def _normalize_bounds(self, bounds: Sequence[Any]) -> Tuple[np.ndarray, np.ndarray]:
        low = np.asarray(bounds[0], dtype=float)
        high = np.asarray(bounds[1], dtype=float)
        if low.ndim == 0:
            low = np.full(self.dim, float(low))
        if high.ndim == 0:
            high = np.full(self.dim, float(high))
        return low.reshape(self.dim), high.reshape(self.dim)

    @staticmethod
    def _clip_float(value: float, low: float, high: float) -> float:
        return float(np.clip(float(value), low, high))

    def _optional_inertia(self, inertia_weight: Optional[float]) -> Optional[float]:
        if inertia_weight is None:
            return None
        return self._clip_float(inertia_weight, 0.0, 2.0)

    @staticmethod
    def _levy_steps(n_steps: int, beta: float) -> np.ndarray:
        sigma = (
            math.gamma(1 + beta)
            * np.sin(np.pi * beta / 2)
            / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        u = np.random.normal(0.0, sigma, size=n_steps)
        v = np.random.normal(0.0, 1.0, size=n_steps)
        return u / np.maximum(np.abs(v), 1e-12) ** (1 / beta)
