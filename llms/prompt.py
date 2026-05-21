import json

"""
construct corresponding prompts
"""
PROMPT_CEC_V1 = """
The particle swarm optimization algorithm has become stuck at a local optimum; the current information of the particle swarm are shown below.
# state summary
{state_summary}
# position and velocity(I reduced the size of the particle swarm using cosine similarity)
position:{position}
velocity:{velocity}.
# algorithm parameters' setting
func_name: {func_name}
func_dimension: {dimension}
population_size: {pop_size}
position_bounds: {pos_b}
velocity_bounds: {vel_b}
Finally, the fitness value of the optimal position found by the current particle swarm is {best_cost}.
Based on the above information, I require a response in JSON format.Your response text should consist of the following two parts.
1. An executable Python code to help the current particle swarm escape its local optimum, the heuristic receive two input as [position, velocity] then return particle's new position and velocity.
2. A suggested inertia weight coefficient to assist the PSO in exploring uncharted territory.
The generated function must be deterministic except for numpy random calls, must keep the input array shapes, and must not read/write files or call external services.
Return the result in JSON format like:
{{
    "code": "import numpy as np\n ...",
    "inertia_weight": 0.9 or else
}}
"""

PROMPT_REACT_TOOLS = """
You are an expert in particle swarm optimization (PSO). Your role is to analyze the current swarm state and decide which intervention tool to apply to help the swarm escape local optima or improve convergence.

# Problem Context
- Objective function: {func_description}
- Problem dimension: {dim}

# Available Tools
You have the following tools at your disposal. For each decision, choose exactly ONE tool.

```json
{tools_json}
```

# Decision Guidelines
- **premature_convergence / velocity_collapse**: use reset_worst_particles or gaussian_mutation to restore diversity
- **boundary_stagnation**: use opposition_reinit to pull particles back into the search space
- **multimodal_trap**: use levy_flight for long jumps to escape local basins
- **slow_stagnation with low diversity**: use reset_worst_particles
- **slow_stagnation with moderate diversity**: use gaussian_mutation
- **normal_convergence**: use adjust_parameters to reduce exploration (lower inertia)
- **normal_search**: use none (no intervention needed)

# Response Format
You MUST respond with a single JSON object:
```json
{{
    "thought": "Your reasoning about WHY the swarm is in this state and WHY you chose this action",
    "action": "tool_name",
    "params": {{"param1": value1, ...}},
    "done": false
}}
```

If you believe no intervention is needed, set action to "none" and done to true.
After applying an action, set done to true if you think the intervention is sufficient.
Set done to false if you want to observe the result and potentially apply another action.

Important: params values must be valid JSON types (numbers, strings, booleans), not expressions.
"""

PROMPT_REACT_WITH_MEMORY = """
You are an expert in particle swarm optimization (PSO). Your role is to analyze the current swarm state and decide which intervention tool to apply to help the swarm escape local optima or improve convergence.

# Problem Context
- Objective function: {func_description}
- Problem dimension: {dim}

# Historical Experience (similar past situations and their outcomes)
{memory_context}

Use this experience to inform your decision — prefer actions that worked well in similar situations, avoid actions that failed.

# Available Tools
You have the following tools at your disposal. For each decision, choose exactly ONE tool.

```json
{tools_json}
```

# Decision Guidelines
- **premature_convergence / velocity_collapse**: use reset_worst_particles or gaussian_mutation to restore diversity
- **boundary_stagnation**: use opposition_reinit to pull particles back into the search space
- **multimodal_trap**: use levy_flight for long jumps to escape local basins
- **rugged_plateau_trap**: use levy_flight or landscape_adaptive_mutation (larger scale for rugged terrain)
- **deceptive_basin**: use opposition_reinit or basin_hopping to escape the deceptive region
- **needle_in_haystack**: use levy_flight with heavy tails + large ratio
- **deep_valley_chase**: use adjust_parameters to accelerate descent (reduce inertia)
- **slow_stagnation with low diversity**: use reset_worst_particles
- **slow_stagnation with moderate diversity**: use gaussian_mutation or landscape_adaptive_mutation
- **normal_convergence**: use adjust_parameters to reduce exploration (lower inertia)
- **normal_search**: use none (no intervention needed)

# Response Format
You MUST respond with a single JSON object:
```json
{{
    "thought": "Your reasoning about WHY the swarm is in this state, informed by past experience, and WHY you chose this action",
    "action": "tool_name",
    "params": {{"param1": value1, ...}},
    "done": false
}}
```

If no intervention is needed, set action to "none" and done to true.
After applying an action, set done to true if you think the intervention is sufficient.
Set done to false if you want to observe the result and potentially apply another action.
"""

PROMPT_NORMAL_V1 = """"""

class LoadPrompt:
    def __init__(self, ):
        # self.types = types # the problem's type
        self._format = {"code": "import numpy as np \n ...", "inertia_weight": 2.0}
        pass

    def generate_prompt4Cec(self, position, velocity, func_name, dimension, best_cost):
        prompt = (
            f"First, the current position and velocities of the particle swarm are shown below.\n "
            f"position:{position},\n velocity:{velocity}.\n"
            f"Second, the objective function and dimensions of the problem are as follows.\n"
            f" functions:{func_name}, dim:{dimension}.\n"
            f"Finally, the fitness value of the optimal position found by the current particle swarm is {best_cost}.\n"
            f"Based on the information above and The function's best solution, could you generate executable Python "
            f"code to help the current particle swarm escape its local optimum,"
            f"the heuristic receive two input as [position, velocity] then return particle's new position and velocity, "
            f"and also return a suggested inertia weight coefficient to assist the PSO in updating its current velocity.\n"
            f"Return the result in JSON format, such as {json.dumps(self._format)}. \n"
            f"Attention!!! The executable code returned to me only has the return value of [position, velocity]."
        )
        return prompt

    # 产生实际问题的提示词
    def generate_prompt4Tsp(self):
        pass

    def generate_prompt4Cec_v(self, position, velocity, func_name, dimension, best_cost):
        prompt = (
            f"First, the current position and velocities of the particle swarm are shown below.\n "
            f"position:{position},\n velocity:{velocity}.\n"
            f"Second, the objective function and dimensions of the problem are as follows.\n"
            f" functions:{func_name}, dim:{dimension}.\n"
            f"Finally, the fitness value of the optimal position found by the current particle swarm is {best_cost}.\n"
            f"Based on the information above and The function's best solution, could you generate executable Python "
            f"code to help the current particle swarm escape its local optimum,"
            f"the heuristic receive two input as [position, velocity] then return particle's new position and velocity, "
            f"and also return a suggested inertia weight coefficient to assist the PSO in updating its current velocity.\n"
            f"Return the result in JSON format, such as {json.dumps(self._format)}. \n"
            f"Attention!!! The executable code returned to me only has the return value of [position, velocity]."
        )
        return prompt
    
# if __name__ == "__main__":
#     from opfunu.cec_based.cec2017 import F12017
#     lp = LoadPrompt('cec')
#     print(lp.generate_prompt4Cec(3, 4, F12017(10), 10, 100))
