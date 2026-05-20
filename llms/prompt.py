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
