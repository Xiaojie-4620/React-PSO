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
