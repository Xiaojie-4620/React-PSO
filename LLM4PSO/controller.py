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

    def decide(self, state, landscape=None) -> ControlDecision:
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

        # --- New emergency labels (triggered early, before full collapse) ---

        if label == "diversity_collapse":
            return ControlDecision(
                state_label=label,
                thought=(
                    "Diversity has catastrophically collapsed with frozen particles. "
                    "Immediate aggressive reset to restore any exploration capability."
                ),
                action="reset_worst_particles",
                params={"ratio": 0.40, "reset_velocity": True, "inertia_weight": 1.2},
            )

        if label == "velocity_death":
            return ControlDecision(
                state_label=label,
                thought=(
                    "Particles are frozen in place but diversity is not yet fully collapsed. "
                    "Gaussian perturbation can restart local movement before diversity dies."
                ),
                action="gaussian_mutation",
                params={"ratio": 0.30, "scale": 0.06, "target": "worst", "inertia_weight": 1.1},
            )

        if label == "multimodal_diverse_stall":
            lp = landscape
            if lp is not None and lp.estimated_basin_radius > 0:
                return ControlDecision(
                    state_label=label,
                    thought=(
                        f"Swarm is diverse but stalled — likely a multimodal trap. "
                        f"Landscape shows basin_radius={lp.estimated_basin_radius:.1f}. "
                        f"Basin-hopping to explore neighboring basins."
                    ),
                    action="basin_hopping",
                    params={
                        "ratio": 0.25,
                        "basin_radius": lp.estimated_basin_radius,
                        "target": "worst",
                        "inertia_weight": 1.05,
                    },
                )
            return ControlDecision(
                state_label=label,
                thought=(
                    "Swarm remains diverse but cannot improve — multimodal trap. "
                    "Levy flight provides heavy-tailed jumps to escape local basins."
                ),
                action="levy_flight",
                params={"ratio": 0.25, "scale": 0.04, "beta": 1.7, "target": "worst", "inertia_weight": 1.0},
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

        if label == "rugged_plateau_trap":
            lp = landscape
            rug = lp.ruggedness if lp is not None else 0.7
            return ControlDecision(
                state_label=label,
                thought=(
                    f"The landscape is a rugged plateau (ruggedness={rug:.3f}) with weak gradients. "
                    "Landscape-adaptive mutation scales perturbations to the terrain roughness."
                ),
                action="landscape_adaptive_mutation",
                params={"ratio": 0.30, "ruggedness": rug, "target": "worst", "inertia_weight": 1.1},
            )

        if label == "deceptive_basin":
            lp = landscape
            basin_r = lp.estimated_basin_radius if lp is not None and lp.estimated_basin_radius > 0 else 1.0
            return ControlDecision(
                state_label=label,
                thought=(
                    "The local gradient points away from promising regions — the basin is deceptive. "
                    "Basin-hopping from gBest can jump particles out of the deceptive attractor."
                ),
                action="basin_hopping",
                params={"ratio": 0.30, "basin_radius": basin_r * 2.0, "target": "worst", "inertia_weight": 1.05},
            )

        if label == "deep_valley_chase":
            lp = landscape
            step = 0.1
            if lp is not None and lp.gradient_magnitude_mean > 0:
                step = min(0.2, max(0.02, lp.gradient_magnitude_mean * 0.5))
            return ControlDecision(
                state_label=label,
                thought=(
                    "The swarm is in a smooth valley with reliable gradients. "
                    "Applying gradient descent steps to accelerate convergence toward the basin floor."
                ),
                action="gradient_descent_step",
                params={"ratio": 0.15, "step_size": step, "target": "worst", "inertia_weight": 0.6},
            )

        if label == "needle_in_haystack":
            lp = landscape
            basin_r = lp.estimated_basin_radius if lp is not None and lp.estimated_basin_radius > 0 else 1.0
            return ControlDecision(
                state_label=label,
                thought=(
                    "The landscape has very high information content — the optimum is a needle in a haystack. "
                    "Landscape-adaptive restart with inverted basin scaling to search aggressively."
                ),
                action="landscape_adaptive_restart",
                params={"ratio": 0.30, "basin_radius": basin_r, "reset_velocity": True, "inertia_weight": 1.1},
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
