import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from llms.prompt import PROMPT_REACT_TOOLS, PROMPT_REACT_WITH_MEMORY


@dataclass
class ReActTurn:
    """Single Thought → Action → Observation cycle."""
    turn: int
    thought: str
    action: str
    params: Dict[str, Any]
    observation: str
    improvement: float
    done: bool


@dataclass
class ReActResult:
    applied: bool
    turns: List[ReActTurn] = field(default_factory=list)
    final_inertia_weight: Optional[float] = None


class LLMReActController:
    """Multi-turn ReAct controller that uses an LLM to reason about swarm state
    and select interventions via Tool Calling.

    The controller iterates:
      1. Send current SwarmState + history to LLM
      2. LLM responds with {"thought": ..., "action": ..., "params": {...}, "done": bool}
      3. Execute the chosen action via StrategyToolbox
      4. Feed ActionResult back as Observation for the next turn
      5. Stop when LLM signals "done", max_turns is reached, or "none" is chosen
    """

    def __init__(
        self,
        llm,
        toolbox,
        max_turns: int = 3,
        function_description: str = "",
        dim: int = 30,
        memory=None,
        verbose: bool = True,
    ):
        self.llm = llm
        self.toolbox = toolbox
        self.max_turns = max_turns
        self.function_description = function_description
        self.dim = dim
        self.memory = memory  # Optional InterventionMemory instance
        self.verbose = verbose

    @property
    def tool_definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "reset_worst_particles",
                "description": (
                    "Randomly reinitialize the worst-performing particles within the search bounds. "
                    "Best for premature convergence when diversity has collapsed."
                ),
                "parameters": {
                    "ratio": {"type": "float", "description": "Fraction of particles to reset (0.0-1.0)", "default": 0.25},
                    "reset_velocity": {"type": "bool", "description": "Whether to also randomize velocities", "default": True},
                    "inertia_weight": {"type": "float", "description": "New inertia weight after intervention (0.0-2.0)", "default": 1.1},
                },
            },
            {
                "name": "gaussian_mutation",
                "description": (
                    "Add Gaussian noise to selected particles' positions and velocities. "
                    "Good for local exploration when the swarm is nearly converged but still improving."
                ),
                "parameters": {
                    "ratio": {"type": "float", "description": "Fraction of particles to mutate (0.0-1.0)", "default": 0.20},
                    "scale": {"type": "float", "description": "Noise scale relative to search span (0.0-1.0)", "default": 0.05},
                    "target": {"type": "str", "description": "Which particles to target: 'worst', 'best', or 'random'", "default": "worst"},
                    "inertia_weight": {"type": "float", "description": "New inertia weight after intervention (0.0-2.0)", "default": None},
                },
            },
            {
                "name": "levy_flight",
                "description": (
                    "Relocate selected particles around the global best using Levy-distributed jumps. "
                    "Best for multimodal traps where long jumps can escape local basins."
                ),
                "parameters": {
                    "ratio": {"type": "float", "description": "Fraction of particles to jump (0.0-1.0)", "default": 0.20},
                    "scale": {"type": "float", "description": "Jump distance scale relative to search span (0.0-1.0)", "default": 0.02},
                    "beta": {"type": "float", "description": "Levy distribution tail parameter (1.01-2.0). Higher = heavier tail", "default": 1.5},
                    "target": {"type": "str", "description": "Which particles to target: 'worst', 'best', or 'random'", "default": "worst"},
                    "inertia_weight": {"type": "float", "description": "New inertia weight after intervention (0.0-2.0)", "default": 1.0},
                },
            },
            {
                "name": "opposition_reinit",
                "description": (
                    "Move selected particles to their opposition positions (reflected through search space center) "
                    "with small jitter. Best for boundary stagnation when particles are trapped at edges."
                ),
                "parameters": {
                    "ratio": {"type": "float", "description": "Fraction of particles to move (0.0-1.0)", "default": 0.25},
                    "velocity_scale": {"type": "float", "description": "Scale of new random velocities relative to velocity bounds", "default": 0.2},
                    "target": {"type": "str", "description": "Which particles to target: 'worst', 'best', or 'random'", "default": "worst"},
                    "inertia_weight": {"type": "float", "description": "New inertia weight after intervention (0.0-2.0)", "default": 0.9},
                },
            },
            {
                "name": "adjust_parameters",
                "description": (
                    "Adjust PSO control parameters (inertia weight, cognitive/social coefficients) "
                    "without modifying any particle positions."
                ),
                "parameters": {
                    "inertia_weight": {"type": "float", "description": "New inertia weight (0.0-2.0)", "default": 0.7},
                    "c1": {"type": "float", "description": "New cognitive coefficient (0.0-4.0)", "default": None},
                    "c2": {"type": "float", "description": "New social coefficient (0.0-4.0)", "default": None},
                },
            },
            {
                "name": "none",
                "description": (
                    "Take no action. Use this when the swarm is making acceptable progress "
                    "and no intervention is needed. Must set 'done': true when choosing this."
                ),
                "parameters": {},
            },
            {
                "name": "basin_hopping",
                "description": (
                    "Hop selected particles to random positions within an estimated basin radius from gBest. "
                    "More targeted than levy_flight — uses estimated basin size to jump to neighboring basins. "
                    "Best for multimodal traps where the basin structure is known from landscape analysis."
                ),
                "parameters": {
                    "ratio": {"type": "float", "description": "Fraction of particles to hop (0.0-1.0)", "default": 0.25},
                    "basin_radius": {"type": "float", "description": "Estimated basin radius for hopping distance", "default": 1.0},
                    "target": {"type": "str", "description": "Which particles to target: 'worst', 'best', or 'random'", "default": "worst"},
                    "inertia_weight": {"type": "float", "description": "New inertia weight after intervention (0.0-2.0)", "default": 1.0},
                },
            },
            {
                "name": "gradient_descent_step",
                "description": (
                    "Move selected particles along estimated local gradient direction (downhill). "
                    "Only useful when gradient information is reliable (low ruggedness). "
                    "Best for smooth valleys where direct descent can accelerate convergence."
                ),
                "parameters": {
                    "ratio": {"type": "float", "description": "Fraction of particles to move (0.0-1.0)", "default": 0.20},
                    "step_size": {"type": "float", "description": "Step size relative to search span (0.001-1.0)", "default": 0.1},
                    "target": {"type": "str", "description": "Which particles to target: 'worst', 'best', or 'random'", "default": "worst"},
                    "inertia_weight": {"type": "float", "description": "New inertia weight after intervention (0.0-2.0)", "default": 0.8},
                },
            },
            {
                "name": "landscape_adaptive_mutation",
                "description": (
                    "Gaussian mutation whose noise scale adapts to landscape ruggedness. "
                    "Higher ruggedness -> larger mutations to cross rough terrain. "
                    "Lower ruggedness -> smaller, more precise local search. "
                    "Best for rugged plateaus where standard gaussian_mutation is poorly calibrated."
                ),
                "parameters": {
                    "ratio": {"type": "float", "description": "Fraction of particles to mutate (0.0-1.0)", "default": 0.20},
                    "ruggedness": {"type": "float", "description": "Landscape ruggedness estimate (0.0-1.0). Higher = rougher.", "default": 0.5},
                    "target": {"type": "str", "description": "Which particles to target: 'worst', 'best', or 'random'", "default": "worst"},
                    "inertia_weight": {"type": "float", "description": "New inertia weight after intervention (0.0-2.0)", "default": None},
                },
            },
            {
                "name": "landscape_adaptive_restart",
                "description": (
                    "Restart particles within a region whose size adapts inversely to the estimated basin radius. "
                    "Small basin -> wider restart to escape. Large basin -> tighter restart to stay in promising region. "
                    "Best for needle-in-haystack landscapes where targeted restarts beat random reinit."
                ),
                "parameters": {
                    "ratio": {"type": "float", "description": "Fraction of particles to restart (0.0-1.0)", "default": 0.25},
                    "basin_radius": {"type": "float", "description": "Estimated basin radius for inverse scaling of restart region", "default": 1.0},
                    "reset_velocity": {"type": "bool", "description": "Whether to also randomize velocities", "default": True},
                    "inertia_weight": {"type": "float", "description": "New inertia weight after intervention (0.0-2.0)", "default": 1.05},
                },
            },
        ]

    def decide_and_act(self, pso, state, iteration: int, gbest_history: List[float], flag: str = "normal", landscape=None) -> ReActResult:
        """Run the ReAct loop for a single intervention episode.

        Returns a ReActResult with all turns and whether any action was applied.
        """
        messages = self._build_initial_messages(state, iteration, landscape=landscape)
        result = ReActResult(applied=False)
        gbest_before = float(pso.get_gBest())

        for turn_idx in range(1, self.max_turns + 1):
            response_text = self._call_llm(messages)
            parsed = self._parse_response(response_text)

            if parsed is None:
                if self.verbose:
                    print(f"[ReAct] Turn {turn_idx}: failed to parse LLM response, stopping")
                break

            thought = parsed.get("thought", "")
            action_name = parsed.get("action", "none")
            params = parsed.get("params", {})
            done = parsed.get("done", False)

            action_result = self.toolbox.apply(pso, action_name, params)
            # CRITICAL: re-evaluate immediately so improvement feedback is real
            pso.evaluation(flag)
            pso.update_p_best_cost()
            pso.update_g_best_cost()
            gbest_after = float(pso.get_gBest())
            improvement = gbest_before - gbest_after
            observation = self._format_observation(action_result, gbest_before, gbest_after)

            turn = ReActTurn(
                turn=turn_idx,
                thought=thought,
                action=action_name,
                params=params,
                observation=observation,
                improvement=improvement,
                done=done,
            )
            result.turns.append(turn)

            if action_result.applied:
                result.applied = True
            if action_result.inertia_weight is not None:
                result.final_inertia_weight = action_result.inertia_weight

            if self.verbose:
                print(
                    f"[ReAct] Turn {turn_idx}: thought='{thought[:80]}...', "
                    f"action={action_name}, applied={action_result.applied}, done={done}"
                )

            if done or action_name == "none":
                break

            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": (
                f"Observation: {observation}\n\n"
                f"Analyze what this outcome tells you. Was your previous action effective? "
                f"What is the current best strategy? Respond with JSON: "
                f"{{\"thought\": \"<your analysis and reasoning>\", \"action\": \"<tool_name>\", \"params\": {{...}}, \"done\": false}}"
            )})
            gbest_before = gbest_after

        return result

    def _build_initial_messages(self, state, iteration: int, landscape=None) -> List[Dict[str, str]]:
        state_json = json.dumps(state.to_prompt_dict(), ensure_ascii=True, indent=2)
        tools_json = json.dumps(self.tool_definitions, ensure_ascii=True, indent=2)

        memory_context = ""
        if self.memory is not None:
            memory_context = self.memory.build_memory_context(state, k=3)

        if memory_context:
            template = PROMPT_REACT_WITH_MEMORY
        else:
            template = PROMPT_REACT_TOOLS

        system_prompt = template.format(
            tools_json=tools_json,
            func_description=self.function_description or "unknown function",
            dim=self.dim,
            memory_context=memory_context or "No prior experience available.",
        )

        # Include landscape profile if available
        landscape_block = ""
        if landscape is not None:
            try:
                lp_dict = landscape.to_dict() if hasattr(landscape, 'to_dict') else vars(landscape)
                lp_json = json.dumps(lp_dict, ensure_ascii=True, indent=2)
                landscape_block = (
                    f"\n\n## Online Landscape Analysis (current search region)\n"
                    f"```json\n{lp_json}\n```\n"
                    f"Landscape label: {landscape.landscape_label}\n"
                )
            except Exception:
                pass

        user_message = (
            f"Iteration {iteration}: the swarm state is shown below.\n\n"
            f"```json\n{state_json}\n```"
            f"{landscape_block}\n"
            f"Analyze the state, think about what is happening to the swarm, "
            f"and decide which tool to call (or 'none' if no intervention is needed).\n"
            f"Return your decision as a JSON object with keys: thought, action, params, done."
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        # Preferred path: BaseLLM with proper chat interface
        if hasattr(self.llm, "chat"):
            try:
                result = self.llm.chat(messages)
                content = result.get("content", "")
                if content:
                    return content
            except Exception as exc:
                if self.verbose:
                    print(f"[ReAct] LLM chat() failed: {exc}, falling back to getResponse()")

        # Fallback: old getResponse interface
        combined = "\n".join(
            f"[{m['role']}]: {m['content']}" for m in messages
        )
        return self.llm.getResponse(combined)

    def _parse_response(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None

        text = text.strip()

        # Remove markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object with regex
        import re
        match = re.search(r'\{[^{}]*"thought"[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        if self.verbose:
            print(f"[ReAct] Could not parse response: {text[:300]}")
        return None

    def _format_observation(self, action_result, gbest_before: float, gbest_after: float) -> str:
        delta = gbest_before - gbest_after
        if action_result.applied:
            return (
                f"Action '{action_result.name}' was applied. "
                f"{action_result.summary} "
                f"Best cost changed from {gbest_before:.6e} to {gbest_after:.6e} "
                f"(delta={delta:+.6e})."
            )
        return (
            f"Action '{action_result.name}' was NOT applied (no particles changed). "
            f"Best cost remains {gbest_after:.6e}."
        )
