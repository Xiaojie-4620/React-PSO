"""Deep Chain-of-Thought ReAct Controller with structured reasoning.

Extends LLMReActController with a five-stage reasoning template:
  Observe → Diagnose → Strategize → Act → Reflect

Tracks diagnosis accuracy over time and uses self-critique after failed
interventions to improve subsequent decisions.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .llm_react import LLMReActController, ReActResult, ReActTurn

# Five-stage reasoning system prompt template
DEEP_REACT_SYSTEM_PROMPT = """
You are an expert in particle swarm optimization (PSO) and fitness landscape analysis.
Your role is to analyze swarm states using structured scientific reasoning and decide interventions.

# Problem Context
- Objective function: {func_description}
- Problem dimension: {dim}
- Known landscape properties: {landscape_prior}

# Historical Experience
{memory_context}

# Your Reasoning Process
You MUST follow this five-stage reasoning for each decision:

## Stage 1: OBSERVE
Describe what the swarm metrics reveal:
- Is the swarm stalled? For how long?
- What is the diversity, velocity distribution, boundary status?
- What does the landscape profile tell us? (ruggedness, gradients, basin structure)

## Stage 2: DIAGNOSE
Form a hypothesis about WHY the swarm is in this state:
- Is this premature convergence, a multimodal trap, a deceptive basin, or normal progress?
- What is the root cause, not just the symptom?
- Link your diagnosis to specific metric values

## Stage 3: STRATEGIZE
Propose a strategy based on your diagnosis:
- What is the goal of this intervention? (restore diversity, escape basin, accelerate descent...)
- Which tool(s) would achieve this goal?
- Why is this strategy better than alternatives?

## Stage 4: ACT
Execute ONE tool with precise parameters chosen for the current situation.

## Stage 5: REFLECT (after seeing the result)
- Did the intervention produce the expected outcome?
- If not, what was wrong with your diagnosis?
- How should the strategy change for the next attempt?

# Available Tools
```json
{tools_json}
```

# Response Format
Respond with a JSON object containing your reasoning chain and action:
```json
{{
    "observe": "What the metrics show (be specific with values)...",
    "diagnose": "My hypothesis about the root cause is...",
    "strategize": "The best approach is... because...",
    "action": "tool_name",
    "params": {{"param1": value1, ...}},
    "done": false
}}
```

After receiving an observation, reflect on it:
```json
{{
    "reflect": "The intervention did/did not work because...",
    "revised_diagnose": "Updated understanding...",
    "action": "next_tool_or_none",
    "params": {{}},
    "done": true_or_false
}}
```

Set done=true when you believe no further intervention is needed.
"""


@dataclass
class DiagnosisRecord:
    """Tracks a diagnosis and whether it was correct."""
    iteration: int
    state_label: str
    diagnosis: str
    action: str
    improvement_delta: float
    was_correct: bool
    reflection: str = ""


@dataclass
class DeepReActTurn(ReActTurn):
    """Extended turn with full reasoning chain."""
    observe: str = ""
    diagnose: str = ""
    strategize: str = ""
    reflect: str = ""


class DeepReActController(LLMReActController):
    """ReAct controller with structured five-stage reasoning.

    Overrides prompt construction to use the DEEP_REACT_SYSTEM_PROMPT
    template and tracks diagnosis accuracy over time.
    """

    def __init__(
        self,
        llm,
        toolbox,
        max_turns: int = 3,
        function_description: str = "",
        dim: int = 30,
        memory=None,
        landscape_prior: str = "unknown",
        verbose: bool = True,
    ):
        super().__init__(
            llm=llm,
            toolbox=toolbox,
            max_turns=max_turns,
            function_description=function_description,
            dim=dim,
            memory=memory,
            verbose=verbose,
        )
        self.landscape_prior = landscape_prior
        self.diagnosis_history: List[DiagnosisRecord] = []

    def _build_initial_messages(self, state, iteration: int) -> List[Dict[str, str]]:
        state_json = json.dumps(state.to_prompt_dict(), ensure_ascii=True, indent=2)
        tools_json = json.dumps(self.tool_definitions, ensure_ascii=True, indent=2)

        memory_context = ""
        if self.memory is not None:
            memory_context = self.memory.build_memory_context(state, k=3)
        if not memory_context:
            memory_context = "No prior experience available."

        # Include diagnosis accuracy stats if available
        diagnosis_stats = self._diagnosis_stats_context()

        system_prompt = DEEP_REACT_SYSTEM_PROMPT.format(
            func_description=self.function_description or "unknown function",
            dim=self.dim,
            landscape_prior=self.landscape_prior,
            memory_context=memory_context + diagnosis_stats,
            tools_json=tools_json,
        )

        user_message = (
            f"Iteration {iteration}: the swarm state is shown below.\n\n"
            f"```json\n{state_json}\n```\n\n"
            f"Follow the five-stage reasoning process (Observe → Diagnose → Strategize → Act) "
            f"and return your decision as JSON."
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

    def _format_observation(self, action_result, gbest_before, gbest_after):
        base = super()._format_observation(action_result, gbest_before, gbest_after)
        return (
            base + "\n\nNow REFLECT: Was your diagnosis correct? "
            "Did the intervention help? Update your understanding and decide the next step. "
            "Respond with: reflect, revised_diagnose, action, params, done."
        )

    def _parse_response(self, text: str) -> Optional[Dict[str, Any]]:
        parsed = super()._parse_response(text)
        if parsed is None:
            return None
        # Normalize deep reasoning fields into the standard format
        if "observe" in parsed or "diagnose" in parsed or "strategize" in parsed:
            thought_parts = []
            for field in ["observe", "diagnose", "strategize", "reflect"]:
                if field in parsed and parsed[field]:
                    thought_parts.append(f"[{field.upper()}] {parsed[field]}")
            if thought_parts:
                parsed.setdefault("thought", " | ".join(thought_parts))
        return parsed

    def _create_turn(self, turn_idx, parsed, action_result, gbest_before, gbest_after):
        """Create a DeepReActTurn with full reasoning chain."""
        turn = DeepReActTurn(
            turn=turn_idx,
            thought=parsed.get("thought", ""),
            action=parsed.get("action", "none"),
            params=parsed.get("params", {}),
            observation=self._format_observation(action_result, gbest_before, gbest_after),
            improvement=gbest_before - gbest_after,
            done=parsed.get("done", False),
            observe=parsed.get("observe", ""),
            diagnose=parsed.get("diagnose", ""),
            strategize=parsed.get("strategize", ""),
            reflect=parsed.get("reflect", ""),
        )
        return turn

    def decide_and_act(self, pso, state, iteration, gbest_history, flag: str = "normal"):
        """Override to use DeepReActTurn for detailed tracking."""
        messages = self._build_initial_messages(state, iteration)
        result = ReActResult(applied=False)
        gbest_before = float(pso.get_gBest())

        for turn_idx in range(1, self.max_turns + 1):
            response_text = self._call_llm(messages)
            parsed = self._parse_response(response_text)

            if parsed is None:
                if self.verbose:
                    print(f"[DeepReAct] Turn {turn_idx}: failed to parse LLM response, stopping")
                break

            action_name = parsed.get("action", "none")
            params = parsed.get("params", {})
            done = parsed.get("done", False)

            action_result = self.toolbox.apply(pso, action_name, params)
            # CRITICAL: re-evaluate immediately so improvement feedback is real
            pso.evaluation(flag)
            pso.update_p_best_cost()
            pso.update_g_best_cost()
            gbest_after = float(pso.get_gBest())

            turn = self._create_turn(turn_idx, parsed, action_result, gbest_before, gbest_after)
            result.turns.append(turn)

            if action_result.applied:
                result.applied = True
            if action_result.inertia_weight is not None:
                result.final_inertia_weight = action_result.inertia_weight

            if self.verbose:
                diag = parsed.get("diagnose", "")[:60]
                print(
                    f"[DeepReAct] Turn {turn_idx}: diagnose='{diag}...', "
                    f"action={action_name}, applied={action_result.applied}, done={done}"
                )

            # Record diagnosis
            if parsed.get("diagnose"):
                self.diagnosis_history.append(DiagnosisRecord(
                    iteration=iteration,
                    state_label=getattr(state, "state_label", "unknown"),
                    diagnosis=parsed["diagnose"],
                    action=action_name,
                    improvement_delta=turn.improvement,
                    was_correct=turn.improvement > 1e-6,
                    reflection=parsed.get("reflect", ""),
                ))

            if done or action_name == "none":
                break

            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": turn.observation})
            gbest_before = gbest_after

        return result

    def _diagnosis_stats_context(self) -> str:
        if not self.diagnosis_history:
            return ""
        correct = sum(1 for d in self.diagnosis_history if d.was_correct)
        total = len(self.diagnosis_history)
        return (
            f"\n\n# Your Diagnosis Track Record\n"
            f"Correct diagnoses: {correct}/{total} ({correct/max(1,total)*100:.0f}% accuracy).\n"
            f"Learn from past mistakes to improve your diagnostic accuracy.\n"
        )
