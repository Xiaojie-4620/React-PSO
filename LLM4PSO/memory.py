"""Intervention Memory System for experience-based learning.

Records every intervention outcome and retrieves similar past experiences
to guide the ReAct agent. Implements Layer 5 (Feedback Learning) of the
Agent-PSO architecture.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class MemoryEntry:
    """A single intervention record."""

    state_label: str
    state_features: Dict[str, float]
    action: str
    params: Dict[str, Any]
    improvement_delta: float
    success: bool
    function_name: str
    dim: int
    iteration: int
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_label": self.state_label,
            "state_features": self.state_features,
            "action": self.action,
            "params": self.params,
            "improvement_delta": self.improvement_delta,
            "success": self.success,
            "function_name": self.function_name,
            "dim": self.dim,
            "iteration": self.iteration,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        return cls(
            state_label=data["state_label"],
            state_features=data["state_features"],
            action=data["action"],
            params=data.get("params", {}),
            improvement_delta=data["improvement_delta"],
            success=data["success"],
            function_name=data.get("function_name", ""),
            dim=data.get("dim", 0),
            iteration=data.get("iteration", 0),
            timestamp=data.get("timestamp", ""),
        )


class InterventionMemory:
    """Stores and retrieves intervention experiences.

    Supports:
    - Recording intervention outcomes
    - Querying similar past interventions by state features
    - Computing aggregate success statistics
    - Cross-run persistence via JSON files
    """

    # Feature keys used for similarity computation (must be present in state_features)
    SIMILARITY_FEATURES = [
        "normalized_diversity",
        "velocity_zero_ratio",
        "velocity_norm_mean",
        "swarm_radius",
        "fitness_cv",
        "boundary_hit_ratio",
        "velocity_direction_consistency",
        "relative_improvement",
    ]

    def __init__(
        self,
        max_entries: int = 5000,
        success_threshold: float = 1e-4,
        similarity_threshold: float = 0.0,
    ):
        self.max_entries = max_entries
        self.success_threshold = success_threshold
        self.similarity_threshold = similarity_threshold
        self._entries: List[MemoryEntry] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        state,
        action: str,
        params: Dict[str, Any],
        improvement_delta: float,
        function_name: str = "",
        dim: int = 0,
        iteration: int = 0,
    ) -> MemoryEntry:
        """Record an intervention and its outcome.

        Args:
            state: SwarmState or dict from to_prompt_dict().
            action: Name of the applied action.
            params: Parameters used for the action.
            improvement_delta: Change in gbest after intervention (positive = improvement).
            function_name: Name of the function being optimized.
            dim: Problem dimension.
            iteration: Iteration at which intervention occurred.

        Returns:
            The created MemoryEntry.
        """
        features = self._extract_features(state)
        success = improvement_delta > self.success_threshold

        entry = MemoryEntry(
            state_label=self._get_state_label(state),
            state_features=features,
            action=action,
            params=params,
            improvement_delta=improvement_delta,
            success=success,
            function_name=function_name,
            dim=dim,
            iteration=iteration,
            timestamp=datetime.now().isoformat(),
        )

        self._entries.append(entry)
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries:]

        return entry

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def query(
        self,
        state,
        k: int = 5,
        require_success: bool = False,
        same_function: bool = False,
    ) -> List[MemoryEntry]:
        """Retrieve the top-K most similar past interventions.

        Similarity is computed via cosine similarity on normalized feature vectors.

        Args:
            state: Current SwarmState or dict.
            k: Number of results to return.
            require_success: If True, only return entries where success=True.
            same_function: If True, only return entries for the same function.

        Returns:
            List of MemoryEntry sorted by similarity (descending).
        """
        if not self._entries:
            return []

        query_vec = self._to_feature_vector(self._extract_features(state))
        candidates = self._entries

        if same_function:
            func_name = self._get_function_name(state)
            if func_name:
                candidates = [e for e in candidates if e.function_name == func_name]

        if require_success:
            candidates = [e for e in candidates if e.success]

        if not candidates:
            return []

        sims = []
        for entry in candidates:
            entry_vec = self._to_feature_vector(entry.state_features)
            sim = self._cosine_similarity(query_vec, entry_vec)
            if sim >= self.similarity_threshold:
                sims.append((sim, entry))

        sims.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in sims[:k]]

    def get_success_rate(self, action: str, state_label: Optional[str] = None) -> float:
        """Compute the success rate of a specific action, optionally filtered by state label."""
        filtered = [
            e for e in self._entries
            if e.action == action and (state_label is None or e.state_label == state_label)
        ]
        if not filtered:
            return 0.5  # neutral prior
        return sum(1 for e in filtered if e.success) / len(filtered)

    def best_action_for_label(self, state_label: str, min_samples: int = 3) -> Optional[str]:
        """Return the action with the highest success rate for a given state label."""
        entries = [e for e in self._entries if e.state_label == state_label]
        if len(entries) < min_samples:
            return None

        rates: Dict[str, List[float]] = {}
        for e in entries:
            rates.setdefault(e.action, []).append(1.0 if e.success else 0.0)

        best_action = None
        best_rate = -1.0
        for action, scores in rates.items():
            if len(scores) < min_samples:
                continue
            avg = np.mean(scores)
            if avg > best_rate:
                best_rate = avg
                best_action = action

        return best_action

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None):
        """Save all entries to a JSON file."""
        path = Path(path or "./logs/memory/interventions.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in self._entries], f, ensure_ascii=False, indent=2)

    def load(self, path: Optional[str] = None):
        """Load entries from a JSON file. Existing entries are preserved."""
        path = Path(path or "./logs/memory/interventions.json")
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            self._entries.append(MemoryEntry.from_dict(item))
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries:]

    # ------------------------------------------------------------------
    # Summarization for prompts
    # ------------------------------------------------------------------

    def build_memory_context(self, state, k: int = 3) -> str:
        """Generate a text summary of relevant past experiences for LLM prompt."""
        similar = self.query(state, k=k)
        if not similar:
            return "No similar past experiences found."

        lines = ["Past similar intervention experiences:"]
        for i, entry in enumerate(similar, 1):
            outcome = "improved" if entry.success else "did not improve"
            lines.append(
                f"  {i}. State: {entry.state_label} → Action: {entry.action} "
                f"(delta={entry.improvement_delta:+.4e}, {outcome}, "
                f"on {entry.function_name} D={entry.dim})"
            )
        return "\n".join(lines)

    @property
    def stats(self) -> Dict:
        total = len(self._entries)
        if total == 0:
            return {"total_entries": 0, "success_rate": 0.0}
        return {
            "total_entries": total,
            "success_rate": sum(1 for e in self._entries if e.success) / total,
            "unique_actions": len(set(e.action for e in self._entries)),
            "unique_labels": len(set(e.state_label for e in self._entries)),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_features(self, state) -> Dict[str, float]:
        """Extract numeric features from a SwarmState or dict."""
        if hasattr(state, "to_prompt_dict"):
            d = state.to_prompt_dict()
        elif isinstance(state, dict):
            d = state
        else:
            return {}

        features = {}
        for key in self.SIMILARITY_FEATURES:
            if key in d and d[key] is not None:
                features[key] = float(d[key])
        return features

    @staticmethod
    def _get_state_label(state) -> str:
        if hasattr(state, "state_label"):
            return state.state_label
        if isinstance(state, dict):
            return state.get("state_label", "unknown")
        return "unknown"

    @staticmethod
    def _get_function_name(state) -> str:
        return getattr(state, "function_name", "")

    def _to_feature_vector(self, features: Dict[str, float]) -> np.ndarray:
        vec = np.zeros(len(self.SIMILARITY_FEATURES))
        for i, key in enumerate(self.SIMILARITY_FEATURES):
            if key in features:
                vec[i] = features[key]
        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            vec = vec / norm
        return vec

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        dot = float(np.dot(a, b))
        return max(0.0, min(1.0, dot))
