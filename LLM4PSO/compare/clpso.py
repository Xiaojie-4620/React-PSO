"""
Comprehensive Learning PSO (CLPSO) for baseline comparison.
Reference: Liang et al., "Comprehensive Learning Particle Swarm Optimizer
for Global Optimization of Multimodal Functions", IEEE TEVC, 2006.
Key idea: Each particle learns from different particles' personal bests
for different dimensions, preventing premature convergence.
"""

import numpy as np


class CLPSO:
    """Comprehensive Learning Particle Swarm Optimizer."""

    def __init__(
        self,
        func,
        n_pop: int = 50,
        dim: int = 30,
        position_bound=(-100, 100),
        velocity_bound=(-20, 20),
        w: float = 0.729,
        c: float = 1.5,
        refreshing_gap: int = 7,
    ):
        self.func = func
        self.n_pop = n_pop
        self.dim = dim
        self.pos_low, self.pos_high = float(position_bound[0]), float(position_bound[1])
        self.vel_low, self.vel_high = float(velocity_bound[0]), float(velocity_bound[1])
        self.w = w
        self.c = c
        self.refreshing_gap = refreshing_gap

        self.position = None
        self.velocity = None
        self.pbest_pos = None
        self.pbest_cost = None
        self.gbest_pos = None
        self.gbest_cost = np.inf
        self.cost = None

        # CLPSO-specific: exemplar indices for each particle and dimension
        self.exemplar_idx = None
        self.flag_pbest = None  # whether using self pbest
        self.stagnation = None  # iteration counter since last improvement

    def optimize(self, max_iter: int, verbose: bool = False):
        self._init_population()
        self._evaluate()
        self._update_pbest()
        self._update_gbest()
        self.exemplar_idx = np.zeros((self.n_pop, self.dim), dtype=int)
        self.flag_pbest = np.zeros(self.n_pop, dtype=bool)
        self.stagnation = np.zeros(self.n_pop, dtype=int)

        best_history = [self.gbest_cost]

        for it in range(max_iter):
            # Refresh exemplars for stagnated particles
            for i in range(self.n_pop):
                if self.stagnation[i] >= self.refreshing_gap:
                    self._assign_exemplar(i)

            # Update velocities and positions
            r = np.random.rand(self.n_pop, self.dim)
            exemplar_pos = self._get_exemplar_positions()
            self.velocity = (
                self.w * self.velocity
                + self.c * r * (exemplar_pos - self.position)
            )
            self.velocity = np.clip(self.velocity, self.vel_low, self.vel_high)
            self.position = self.position + self.velocity
            self.position = np.clip(self.position, self.pos_low, self.pos_high)

            self._evaluate()
            improved = self._update_pbest()
            self.stagnation[~improved] += 1
            self.stagnation[improved] = 0
            self._update_gbest()
            best_history.append(self.gbest_cost)

            if verbose and (it + 1) % 100 == 0:
                print(f"CLPSO Iter {it + 1:4d} | Best = {self.gbest_cost:.6e}")

        return np.array(best_history)

    def _init_population(self):
        self.position = np.random.uniform(self.pos_low, self.pos_high, (self.n_pop, self.dim))
        self.velocity = np.zeros((self.n_pop, self.dim))
        self.pbest_cost = np.full(self.n_pop, np.inf)
        self.pbest_pos = self.position.copy()

    def _evaluate(self):
        self.cost = self.func(self.position)

    def _update_pbest(self):
        improved = self.cost < self.pbest_cost
        self.pbest_cost[improved] = self.cost[improved]
        self.pbest_pos[improved] = self.position[improved]
        return improved

    def _update_gbest(self):
        idx = np.argmin(self.pbest_cost)
        if self.pbest_cost[idx] < self.gbest_cost:
            self.gbest_cost = self.pbest_cost[idx]
            self.gbest_pos = self.pbest_pos[idx].copy()

    def _assign_exemplar(self, i: int):
        """Assign exemplar for particle i using tournament selection."""
        for d in range(self.dim):
            f1 = np.random.randint(self.n_pop)
            f2 = np.random.randint(self.n_pop)
            if self.pbest_cost[f1] < self.pbest_cost[f2]:
                self.exemplar_idx[i, d] = f1
            else:
                self.exemplar_idx[i, d] = f2

        # With probability Pc_i, use self pbest for all dimensions
        Pc = 0.05 + 0.45 * (np.exp(10 * (i) / (self.n_pop - 1)) - 1) / (np.exp(10) - 1)
        if np.random.rand() < Pc:
            self.flag_pbest[i] = True

    def _get_exemplar_positions(self):
        """Build exemplar position matrix."""
        exemplars = np.zeros_like(self.position)
        for i in range(self.n_pop):
            if self.flag_pbest[i]:
                exemplars[i] = self.pbest_pos[i]
            else:
                for d in range(self.dim):
                    exemplars[i, d] = self.pbest_pos[self.exemplar_idx[i, d], d]
        return exemplars
