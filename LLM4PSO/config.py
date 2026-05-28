"""Standardized PSO configuration dataclass."""

from dataclasses import dataclass, field
from typing import Tuple, Union

import numpy as np


@dataclass
class PSOConfig:
    """All tunable parameters for a PSO run in one place."""

    # Problem
    dim: int
    flag: str = "cec"  # 'cec', 'opfunu_cec', 'else'
    func: object = None

    # Population
    pop_size: int = 50
    max_iter: int = 500

    # PSO coefficients
    w: float = 0.729
    wdamp: float = 0.99
    c1: float = 1.5
    c2: float = 1.5

    # Bounds (scalar or per-dimension arrays)
    position_bounds: Tuple = (-100.0, 100.0)
    velocity_bounds: Tuple = (-20.0, 20.0)

    # Stagnation detection
    stagnation_threshold: int = 100
    improvement_tolerance: float = 1e-3

    # Intervention
    intervention_mode: str = "rule"  # 'rule', 'llm', 'llm_react'

    @property
    def velocity_factor(self) -> float:
        """Velocity bound as fraction of position bound span."""
        span = np.asarray(self.position_bounds[1]) - np.asarray(self.position_bounds[0])
        vel_span = np.asarray(self.velocity_bounds[1]) - np.asarray(self.velocity_bounds[0])
        return float(np.mean(vel_span / np.maximum(np.abs(span), 1e-12)))

    @classmethod
    def for_cec2017(cls, dim: int = 30, **overrides) -> "PSOConfig":
        """Preset for CEC 2017 benchmark suite.

        Args:
            dim: Problem dimension (10, 30, 50, or 100).
            **overrides: Key-value pairs to override defaults.
        """
        bounds = (-100.0, 100.0)
        return cls(
            dim=dim,
            flag="cec",
            pop_size=min(50, max(20, dim * 2)),
            max_iter=10_000 * dim,
            w=0.729,
            wdamp=1.0,
            c1=1.5,
            c2=1.5,
            position_bounds=bounds,
            velocity_bounds=(-20.0, 20.0),
            stagnation_threshold=max(50, dim * 10),
            improvement_tolerance=1e-8,
            **overrides,
        )

    @classmethod
    def for_cec2014(cls, dim: int = 30, **overrides) -> "PSOConfig":
        """Preset for CEC 2014 benchmark suite."""
        bounds = (-100.0, 100.0)
        return cls(
            dim=dim,
            flag="opfunu_cec",
            pop_size=min(50, max(20, dim * 2)),
            max_iter=10_000 * dim,
            w=0.729,
            wdamp=1.0,
            c1=1.5,
            c2=1.5,
            position_bounds=bounds,
            velocity_bounds=(-20.0, 20.0),
            stagnation_threshold=max(50, dim * 10),
            improvement_tolerance=1e-8,
            **overrides,
        )

    @classmethod
    def for_quick_test(cls, dim: int = 10, **overrides) -> "PSOConfig":
        """Small, fast preset for development and testing."""
        bounds = (-100.0, 100.0)
        vel = 0.2 * (np.asarray(bounds[1]) - np.asarray(bounds[0]))
        return cls(
            dim=dim,
            flag="else",
            pop_size=20,
            max_iter=50,
            w=0.9,
            wdamp=0.99,
            c1=1.5,
            c2=1.5,
            position_bounds=bounds,
            velocity_bounds=(-vel, vel),
            stagnation_threshold=100,
            improvement_tolerance=1e-3,
            **overrides,
        )

    def to_dict(self) -> dict:
        return {
            "dim": self.dim,
            "pop_size": self.pop_size,
            "max_iter": self.max_iter,
            "w": self.w,
            "wdamp": self.wdamp,
            "c1": self.c1,
            "c2": self.c2,
            "position_bounds": self.position_bounds,
            "velocity_bounds": self.velocity_bounds,
            "stagnation_threshold": self.stagnation_threshold,
            "improvement_tolerance": self.improvement_tolerance,
            "intervention_mode": self.intervention_mode,
        }
