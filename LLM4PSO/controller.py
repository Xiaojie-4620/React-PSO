from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class ControlDecision:
    state_label: str
    thought: str
    action: str
    params: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RuleBasedReActController:
    """Template ReAct controller for deterministic, explainable interventions."""

    def decide(self, state) -> ControlDecision:
        label = state.state_label

        if label == "boundary_stagnation":
            return ControlDecision(
                state_label=label,
                thought=(
                    "The swarm is stagnant while many coordinates sit on the search boundary; "
                    "an opposition move should pull weak particles back into the search region."
                ),
                action="opposition_reinit",
                params={"ratio": 0.25, "velocity_scale": 0.15, "inertia_weight": 0.9},
            )

        if label == "premature_convergence":
            return ControlDecision(
                state_label=label,
                thought=(
                    "Diversity and velocity both collapsed, so the swarm likely converged to a local basin; "
                    "resetting weak particles should restore exploration."
                ),
                action="reset_worst_particles",
                params={"ratio": 0.30, "reset_velocity": True, "inertia_weight": 1.1},
            )

        if label == "velocity_collapse":
            return ControlDecision(
                state_label=label,
                thought=(
                    "Particle motion is almost exhausted before meaningful progress resumed; "
                    "Gaussian perturbation can restart local movement without discarding the whole swarm."
                ),
                action="gaussian_mutation",
                params={"ratio": 0.25, "scale": 0.04, "target": "worst", "inertia_weight": 1.0},
            )

        if label == "multimodal_trap":
            return ControlDecision(
                state_label=label,
                thought=(
                    "The swarm remains diverse but the best value is not improving, which suggests a rugged "
                    "multi-basin landscape; Levy flight gives selected particles a longer exploratory jump."
                ),
                action="levy_flight",
                params={"ratio": 0.25, "scale": 0.03, "beta": 1.5, "target": "worst", "inertia_weight": 1.0},
            )

        if label == "slow_stagnation":
            if state.normalized_diversity < 0.08:
                return ControlDecision(
                    state_label=label,
                    thought=(
                        "Progress has stalled and diversity is already low; weak-particle reset is the most direct "
                        "way to recover exploration."
                    ),
                    action="reset_worst_particles",
                    params={"ratio": 0.20, "reset_velocity": True, "inertia_weight": 1.05},
                )
            return ControlDecision(
                state_label=label,
                thought=(
                    "Progress has stalled but the swarm still has some spread; a moderate mutation should test "
                    "nearby basins before stronger restarts."
                ),
                action="gaussian_mutation",
                params={"ratio": 0.20, "scale": 0.03, "target": "worst", "inertia_weight": 0.95},
            )

        if label == "normal_convergence":
            return ControlDecision(
                state_label=label,
                thought="The swarm appears to be converging normally; reduce exploration pressure.",
                action="adjust_parameters",
                params={"inertia_weight": 0.7},
            )

        return ControlDecision(
            state_label=label,
            thought="The swarm is still making acceptable progress; no intervention is needed.",
            action="none",
            params={},
        )
