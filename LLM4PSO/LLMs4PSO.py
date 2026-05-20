import importlib.util as util
import inspect
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

try:
    from .actions import StrategyToolbox
    from .controller import RuleBasedReActController
    from .PSO import PSO
    from .feature import Feature
    from .state import SwarmStateAnalyzer
except ImportError:
    from actions import StrategyToolbox
    from controller import RuleBasedReActController
    from PSO import PSO
    from feature import Feature
    from state import SwarmStateAnalyzer

from llms.prompt import PROMPT_CEC_V1
from tools.extract_code import Extract


class LLM4PSO:
    def __init__(
        self,
        dim,
        flag,
        func,
        w,
        pop_size: int,
        iterations: int,
        wdamp: float,
        c1,
        c2,
        position_bounds,
        velocity_bounds,
        stagnation_threshold=200,
        improvement_tolerance=1e-3,
        intervention_mode="rule",
    ):
        self.dim = dim
        self.flag = flag
        self.func = func
        self.n_pop = pop_size
        self.max_iter = iterations
        self.wdamp = wdamp
        self.c1 = c1
        self.c2 = c2
        self.position_bounds = position_bounds
        self.velocity_bounds = velocity_bounds
        self.w = w
        self.Global_best = []
        self.state_history = []
        self.action_history = []
        self.stagnation_counter = 0
        self.stagnation_threshold = int(stagnation_threshold)
        self.improvement_tolerance = float(improvement_tolerance)
        self.intervention_mode = intervention_mode
        self.stagnation = False
        self.feat = Feature()
        self.feat.getAllFuncName()
        self.func_name = self._function_description()
        self.state_analyzer = SwarmStateAnalyzer(
            position_bounds=position_bounds,
            velocity_bounds=velocity_bounds,
            dim=dim,
            improvement_tolerance=self.improvement_tolerance,
            stagnation_window=min(max(5, self.stagnation_threshold // 2), max(5, self.stagnation_threshold)),
        )
        self.toolbox = StrategyToolbox(position_bounds, velocity_bounds, dim)
        self.controller = RuleBasedReActController()

    def run(self):
        pso = PSO(
            self.func,
            self.n_pop,
            self.dim,
            self.c1,
            self.c2,
            self.position_bounds,
            self.velocity_bounds,
        )
        pso.pop_init()
        pso.evaluation(self.flag)
        pso.update_p_best_cost()
        pso.update_g_best_cost()
        self.Global_best.append(pso.get_gBest())
        self.state_history.append(self.state_analyzer.analyze(pso, 0, self.Global_best, self.stagnation_counter))

        for i in range(1, self.max_iter):
            self._update_stagnation_counter()
            state = self.state_analyzer.analyze(pso, i, self.Global_best, self.stagnation_counter)
            self.state_history.append(state)

            if self._should_intervene(state):
                self.stagnation = self._apply_intervention(pso, state, i)
                if self.stagnation:
                    self.stagnation_counter = 0

            if self.stagnation:
                pso.evaluation(self.flag)
                pso.update_p_best_cost()
                pso.update_g_best_cost()
                self.Global_best.append(pso.get_gBest())
            else:
                pso.update_pos_vel(self.w)
                pso.evaluation(self.flag)
                pso.update_p_best_cost()
                pso.update_g_best_cost()
                self.Global_best.append(pso.get_gBest())

            if i % 100 == 0:
                print(
                    f"Iteration[{i}], Best_cost: {self.Global_best[-1]}, "
                    f"weight: {self.w}, state: {state.state_label}"
                )

            if not self.stagnation:
                self.w *= self.wdamp

            self.stagnation = False

        return np.array(self.Global_best)

    def _apply_intervention(self, pso, state, iteration):
        if self.intervention_mode == "llm":
            print(
                f"--- Iteration[{iteration}]: {state.state_label}; "
                f"calling LLM heuristic controller ---"
            )
            pos, vel = self.reduce_particles_by_cosine_similarity(
                pso,
                max_particles=max(1, self.n_pop // 2),
            )
            suggested_w = self.generate_heuristic(pos, vel, pso.get_gBest(), state)
            if suggested_w is not None:
                self.w = suggested_w
            applied = self.execute_heuristic(pso)
            self.action_history.append(
                {
                    "iteration": iteration,
                    "mode": "llm",
                    "state": state.to_prompt_dict(),
                    "applied": applied,
                    "inertia_weight": suggested_w,
                }
            )
            return applied

        decision = self.controller.decide(state)
        result = self.toolbox.apply(pso, decision.action, decision.params)
        if result.inertia_weight is not None:
            self.w = result.inertia_weight
        self.action_history.append(
            {
                "iteration": iteration,
                "mode": "rule",
                "state": state.to_prompt_dict(),
                "decision": decision.to_dict(),
                "result": result.to_dict(),
            }
        )
        if result.applied:
            print(
                f"--- Iteration[{iteration}]: {state.state_label}; "
                f"{decision.action}; {result.summary} ---"
            )
        return result.applied

    def generate_heuristic(self, pos, vel, gbest, state=None):
        state_summary = {}
        if state is not None:
            state_summary = state.to_prompt_dict()

        prompt = PROMPT_CEC_V1.format(
            state_summary=json.dumps(state_summary, ensure_ascii=True, indent=2),
            position=pos,
            velocity=vel,
            func_name=self.func_name,
            dimension=self.dim,
            pop_size=self.n_pop,
            pos_b=self.position_bounds,
            vel_b=self.velocity_bounds,
            best_cost=gbest,
        )
        heuristic_path = Path.cwd() / "heuristic.py"
        print(prompt)

        try:
            from llms.LLM import DeepSeek

            llm = DeepSeek()
            response = llm.getResponse(prompt)
        except Exception as exc:
            print(f"---failed to call LLM: {exc}---")
            return None

        separator = "-" * 20
        print(f"{separator}\n{response}\n{separator}")

        extracted = Extract(response).extract_code()
        if extracted is None:
            return None

        code, inertia = extracted
        Extract(response).save_code(heuristic_path, code)
        self.save_to_json(code, inertia, prompt, self.func_name, self.dim, state_summary)
        return inertia

    def execute_heuristic(self, pso):
        heuristic_path = Path.cwd() / "heuristic.py"
        if not heuristic_path.exists():
            print("------Not found heuristic.py------")
            return False

        spec = util.spec_from_file_location("heuristic_model", heuristic_path)
        heuristic_model = util.module_from_spec(spec)

        try:
            spec.loader.exec_module(heuristic_model)
        except Exception as exc:
            print(f"---failed to import heuristic: {exc}---")
            return False

        functions = inspect.getmembers(heuristic_model, inspect.isfunction)
        if not functions:
            print("failed to find function")
            return False

        func_name, target_func = functions[0]
        print(f"success find function named: {func_name}")

        try:
            result = target_func(pso.position, pso.velocity)
            if not isinstance(result, (tuple, list)) or len(result) not in (2, 3):
                print("---failed to execute heuristic: invalid return value---")
                return False

            new_position, new_velocity = result[0], result[1]
            if len(result) == 3:
                try:
                    self.w = float(result[2])
                except (TypeError, ValueError):
                    pass

            pso.position = np.clip(new_position, self.position_bounds[0], self.position_bounds[1])
            pso.velocity = np.clip(new_velocity, self.velocity_bounds[0], self.velocity_bounds[1])
            print("-----success executing heuristic-----")
            return True
        except Exception as exc:
            print(f"---failed to execute heuristic: {exc}---")
            return False

    def visualization(self, history_data):
        iteration = np.arange(len(history_data))
        plt.figure(1)
        plt.xlabel("Iteration")
        plt.ylabel("Best_Cost")
        plt.plot(iteration, history_data, "r--")
        plt.show()

    def to_stirng(self):
        print(
            "parameters are as follows:"
            f"dimension: {self.dim}, wdamp: {self.wdamp}, w: {self.w}, "
            f"pop_size: {self.n_pop}, iterations: {self.max_iter}, "
            f"position_bounds: {self.position_bounds}, velocity_bounds: {self.velocity_bounds}, "
            f"c1: {self.c1}, c2: {self.c2}, problem_type: {self.flag}"
        )

    def save_data(self):
        df = pd.DataFrame({"Best_cost": np.array(self.Global_best)})
        if self.flag == "opfunu_cec":
            name = self.func.name.split(":")[0]
            df.to_csv(f"./data/{name}2014's_Best_Cost_dim={self.dim}.csv")
        elif self.flag == "cec":
            name = self.func.__name__
            df.to_csv(f"./data/{name}2017's_Best_Cost_dim={self.dim}.csv")
        else:
            df.to_csv(f"./data/dataFor{self.func_name}_dim={self.dim}.csv")

    def save_to_json(self, code, inertia, prompt, func_name, dim, state_summary=None):
        save_dir = Path("./logs") / "llm_records"
        save_dir.mkdir(parents=True, exist_ok=True)

        existing = list(save_dir.glob("response*.json"))
        record_id = len(existing) + 1
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_func_name = "".join(c if c.isalnum() else "_" for c in str(func_name))
        filename = save_dir / f"response_{record_id:03d}_{safe_func_name}_{timestamp}.json"

        records = {
            "record_id": record_id,
            "timestamp": timestamp,
            "inertia": inertia,
            "prompt": prompt,
            "func_name": func_name,
            "dim": dim,
            "state_summary": state_summary or {},
            "code": code,
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"LLM intervention record saved to {filename.name}")

    def reduce_particles_by_cosine_similarity(
        self,
        pso: PSO,
        similarity: float = 0.98,
        max_particles: int = 15,
        keep_best: bool = True,
        mode: str = "pos",
    ):
        position = np.asarray(pso.position, dtype=float).copy()
        velocity = np.asarray(pso.velocity, dtype=float).copy()
        costs = np.asarray(pso.cost, dtype=float).copy()
        max_particles = max(1, int(max_particles))

        n = position.shape[0]
        if n == 0:
            return position, np.array([], dtype=float)
        if n <= max_particles:
            return position, velocity

        if mode == "pos_vel":
            feat = np.hstack([position, velocity])
        else:
            feat = position

        norms = np.linalg.norm(feat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        feat_norm = feat / norms

        order = np.argsort(costs)
        selected = []

        if keep_best:
            selected.append(int(np.argmin(costs)))

        for idx in order:
            idx = int(idx)
            if len(selected) >= max_particles:
                break
            if idx in selected:
                continue
            if not selected:
                selected.append(idx)
                continue
            sims = feat_norm[idx] @ feat_norm[selected].T
            if np.max(sims) < similarity:
                selected.append(idx)

        if len(selected) < min(max_particles, n):
            selected_set = set(selected)
            for idx in order:
                idx = int(idx)
                if len(selected) >= max_particles:
                    break
                if idx not in selected_set:
                    selected.append(idx)

        selected = np.array(selected, dtype=int)
        return position[selected], velocity[selected]

    def _update_stagnation_counter(self):
        if len(self.Global_best) < 2:
            self.stagnation_counter = 0
            return

        improvement = self.Global_best[-2] - self.Global_best[-1]
        if improvement <= self.improvement_tolerance:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0

    def _should_intervene(self, state):
        return state.no_improve_iters >= self.stagnation_threshold

    def _function_description(self):
        if hasattr(self.func, "to_str"):
            return self.func.to_str()
        if hasattr(self.func, "__name__"):
            func_name = self.func.__name__
            return self.feat.descriptions.get(func_name, func_name)
        if hasattr(self.func, "name"):
            name_attr = self.func.name
            return name_attr() if callable(name_attr) else str(name_attr)
        return str(self.func)
