"""Experiment configuration definitions.

Declare experiment matrices: which functions, dimensions, intervention modes,
and PSO parameters to compare.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass
class ExperimentConfig:
    """Defines one experiment grid."""

    name: str
    description: str = ""

    # Functions to evaluate: list of (name, callable) tuples
    functions: List[tuple] = field(default_factory=list)

    # Dimensions to test
    dimensions: List[int] = field(default_factory=lambda: [10, 30])

    # Intervention modes to compare
    modes: List[str] = field(default_factory=lambda: ["rule"])

    # PSO parameters (common across runs)
    pop_size: int = 50
    max_iter: Optional[int] = None  # auto-computed from dim if None
    w: float = 0.729
    wdamp: float = 1.0
    c1: float = 1.5
    c2: float = 1.5
    position_bounds: tuple = (-100.0, 100.0)
    velocity_factor: float = 0.2

    # Stagnation / intervention config
    stagnation_threshold: Optional[int] = None  # auto if None
    improvement_tolerance: float = 1e-8

    # Experiment execution
    n_trials: int = 30
    seeds: Optional[Sequence[int]] = None  # generated if None

    # Results
    output_dir: str = "./experiments/results"


# Pre-defined experiment configs

def ablation_intervention_modes() -> ExperimentConfig:
    """Compare different intervention modes on a subset of CEC 2017."""
    return ExperimentConfig(
        name="ablation_modes",
        description="Ablation study comparing intervention modes",
        modes=["none", "rule", "llm_react"],
        dimensions=[10, 30],
        n_trials=30,
    )


def ablation_react_depth() -> ExperimentConfig:
    """Compare ReAct depth (1, 2, 3 turns)."""
    return ExperimentConfig(
        name="ablation_react_depth",
        description="Compare ReAct turn depths",
        modes=["llm_react"],
        dimensions=[30],
        n_trials=30,
    )


def full_cec2017_benchmark() -> ExperimentConfig:
    """Full CEC 2017 benchmark evaluation."""
    return ExperimentConfig(
        name="cec2017_full",
        description="Full CEC 2017 benchmark (30 functions, 4 dims, 30 trials)",
        modes=["rule", "llm_react"],
        dimensions=[10, 30, 50, 100],
        n_trials=30,
    )
