"""Smart LLM invocation scheduling to control API cost.

Provides policies that decide WHEN to call the LLM during stagnation,
rather than calling it on every stagnation event.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PolicyConfig:
    """Configuration for intervention scheduling policies."""

    # Cooldown: minimum iterations between two LLM calls
    cooldown_iters: int = 50

    # Escalation: use rule-based controller for first N stagnation events
    escalation_threshold: int = 3

    # Budget: maximum total LLM calls per run
    budget: int = 50

    # Confidence gate: only call LLM when rule controller is uncertain
    # (when state is multimodal_trap or slow_stagnation, the rule controller
    # has lower confidence compared to clear-cut cases)
    confidence_gate: bool = True

    # Which policies are active
    use_cooldown: bool = True
    use_escalation: bool = True
    use_budget: bool = True
    use_confidence_gate: bool = True


class InterventionScheduler:
    """Decides whether to invoke the LLM based on configurable policies.

    Usage:
        scheduler = InterventionScheduler(PolicyConfig(cooldown_iters=50, budget=20))
        ...
        if scheduler.should_call_llm(state, iteration):
            # invoke LLM ReAct controller
    """

    def __init__(self, config: Optional[PolicyConfig] = None):
        self.config = config or PolicyConfig()
        self._call_count: int = 0
        self._last_call_iteration: int = -999
        self._stagnation_event_count: int = 0

    def should_call_llm(self, state, iteration: int) -> bool:
        """Check all active policies. Returns True if LLM should be called."""

        self._stagnation_event_count += 1

        if self.config.use_budget and not self._check_budget():
            return False

        if self.config.use_cooldown and not self._check_cooldown(iteration):
            return False

        if self.config.use_escalation and not self._check_escalation():
            return False

        if self.config.use_confidence_gate and not self._check_confidence(state):
            return False

        return True

    def record_call(self, iteration: int):
        """Call after a successful LLM invocation."""
        self._call_count += 1
        self._last_call_iteration = iteration

    def _check_budget(self) -> bool:
        return self._call_count < self.config.budget

    def _check_cooldown(self, iteration: int) -> bool:
        return (iteration - self._last_call_iteration) >= self.config.cooldown_iters

    def _check_escalation(self) -> bool:
        """Only escalate to LLM after rule-based controller has been tried N times."""
        return self._stagnation_event_count > self.config.escalation_threshold

    def _check_confidence(self, state) -> bool:
        """Gate: skip LLM for clear-cut cases the rule controller handles well.

        Rule controller is most reliable for:
          - boundary_stagnation -> opposition_reinit
          - premature_convergence -> reset_worst_particles
          - velocity_collapse -> gaussian_mutation
          - normal_search -> none

        LLM adds most value for ambiguous cases:
          - multimodal_trap (complex landscape reasoning needed)
          - slow_stagnation (context-dependent strategy selection needed)
        """
        ambiguous_labels = {"multimodal_trap", "slow_stagnation"}
        label = getattr(state, "state_label", "")
        return label in ambiguous_labels

    @property
    def stats(self) -> Dict:
        return {
            "call_count": self._call_count,
            "stagnation_events": self._stagnation_event_count,
            "call_rate": (
                self._call_count / max(1, self._stagnation_event_count)
            ),
            "budget_remaining": self.config.budget - self._call_count,
        }
