"""Self-evolving rule controller that learns from intervention outcomes.

Starts identical to RuleBasedReActController but adapts its action mappings
and parameters based on observed intervention results over time.
"""

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .controller import ControlDecision, RuleBasedReActController


@dataclass
class ActionStats:
    """Running statistics for an action's performance on a state label."""
    count: int = 0
    success_count: int = 0
    total_improvement: float = 0.0
    avg_params: Dict[str, float] = None

    def __post_init__(self):
        if self.avg_params is None:
            self.avg_params = {}

    @property
    def success_rate(self) -> float:
        return self.success_count / max(1, self.count)

    @property
    def avg_improvement(self) -> float:
        return self.total_improvement / max(1, self.count)

    def update(self, improvement: float, success: bool, params: Dict[str, Any]):
        self.count += 1
        if success:
            self.success_count += 1
        self.total_improvement += improvement
        # Exponential moving average of numeric params
        for key, val in params.items():
            if isinstance(val, (int, float, bool)):
                old = self.avg_params.get(key, float(val))
                self.avg_params[key] = old * 0.8 + float(val) * 0.2


class AdaptiveRuleController(RuleBasedReActController):
    """A RuleBasedReActController that evolves its action mappings from experience.

    After each run, the controller can learn from the intervention history:
    - Which actions work best per state label?
    - What parameters produce the best results?
    - Are there new state labels (e.g., landscape-aware) that need mappings?

    Usage:
        ctrl = AdaptiveRuleController()
        ...
        # After a run:
        ctrl.learn_from_history(action_history)
        # Save evolved policy:
        ctrl.save_policy("learned_policy.json")
    """

    def __init__(self):
        super().__init__()
        self._action_stats: Dict[str, Dict[str, ActionStats]] = {}
        self._custom_rules: Dict[str, Tuple[str, str, Dict[str, Any]]] = {}
        self._param_deltas: Dict[str, Dict[str, float]] = {}

    def decide(self, state) -> ControlDecision:
        label = state.state_label

        # Check for learned override
        if label in self._custom_rules:
            action, thought, base_params = self._custom_rules[label]
            params = dict(base_params)
            # Apply parameter deltas
            if label in self._param_deltas:
                for key, delta in self._param_deltas[label].items():
                    if key in params and isinstance(params[key], (int, float)):
                        params[key] = float(
                            np.clip(float(params[key]) + delta, 0.0, 4.0)
                        )
            return ControlDecision(
                state_label=label, thought=thought, action=action, params=params,
            )

        return super().decide(state)

    def learn_from_history(
        self,
        action_history: List[Dict[str, Any]],
        min_samples: int = 3,
    ):
        """Update action mappings from an intervention history.

        Args:
            action_history: List of intervention records from LLM4PSO.run().
            min_samples: Minimum observations required to override a rule.
        """
        # Reset per learning cycle
        self._action_stats.clear()

        for record in action_history:
            self._process_record(record)

        # Determine best action per state label
        for label, action_map in self._action_stats.items():
            best_action = None
            best_score = -float("inf")
            for action, stats in action_map.items():
                if stats.count < min_samples:
                    continue
                score = stats.avg_improvement
                if score > best_score:
                    best_score = score
                    best_action = action

            if best_action is not None:
                stats = action_map[best_action]
                # Use default thought from parent if this is a new label
                default_decision = self.decide_via_label(label)
                self._custom_rules[label] = (
                    best_action,
                    f"[Learned] Best action for '{label}' "
                    f"(success_rate={stats.success_rate:.2f}, "
                    f"avg_improvement={stats.avg_improvement:.4e}, "
                    f"n={stats.count})",
                    stats.avg_params,
                )

                # Compute parameter deltas relative to defaults
                if default_decision and default_decision.params:
                    deltas = {}
                    for key, val in stats.avg_params.items():
                        if key in default_decision.params and isinstance(
                            default_decision.params[key], (int, float)
                        ):
                            deltas[key] = float(val) - float(default_decision.params[key])
                    if deltas:
                        self._param_deltas[label] = deltas

    def save_policy(self, path: str):
        """Save learned policy to JSON for reuse across runs."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "custom_rules": {
                label: {
                    "action": action,
                    "thought": thought,
                    "params": params,
                }
                for label, (action, thought, params) in self._custom_rules.items()
            },
            "param_deltas": self._param_deltas,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_policy(self, path: str):
        """Load a previously saved policy."""
        path = Path(path)
        if not path.exists():
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for label, rule in data.get("custom_rules", {}).items():
            self._custom_rules[label] = (
                rule["action"],
                rule["thought"],
                rule["params"],
            )

        self._param_deltas = data.get("param_deltas", {})

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _process_record(self, record: Dict[str, Any]):
        """Extract learning signal from one intervention record."""
        state_dict = record.get("state", {})
        label = state_dict.get("state_label", "unknown")

        if record.get("mode") == "rule":
            action = record.get("decision", {}).get("action", "")
            params = record.get("decision", {}).get("params", {})
        elif record.get("mode") in ("llm_react",):
            turns = record.get("turns", [])
            if not turns:
                return
            action = turns[0].get("action", "none")
            params = turns[0].get("params", {})
        else:
            return

        if not action or action == "none":
            return

        # Estimate improvement from this intervention
        improvement = 0.0
        if record.get("mode") == "rule":
            result = record.get("result", {})
            if result.get("applied"):
                improvement = 1e-4  # Default positive signal for applied action
        elif record.get("mode") in ("llm_react",):
            turns = record.get("turns", [])
            for t in turns:
                improvement += t.get("improvement", 0.0)

        success = improvement > 1e-6

        if label not in self._action_stats:
            self._action_stats[label] = {}
        if action not in self._action_stats[label]:
            self._action_stats[label][action] = ActionStats()

        self._action_stats[label][action].update(improvement, success, params)

    def decide_via_label(self, label: str) -> Optional[ControlDecision]:
        """Get the default decision from parent for a given label."""
        # Create a minimal fake state with just the label
        class _FakeState:
            state_label = label
        return super().decide(_FakeState())
