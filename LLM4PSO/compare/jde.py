"""
Self-adaptive Differential Evolution (jDE) for baseline comparison.
Reference: Brest et al., "Self-Adapting Control Parameters in Differential
Evolution: A Comparative Study", IEEE TEVC, 2006.
Key idea: F and CR parameters are self-adapted per individual with
probabilistic mutation before each generation.
"""

import numpy as np


class JDE:
    """Self-adaptive Differential Evolution (jDE)."""

    def __init__(
        self,
        func,
        n_pop: int = 50,
        dim: int = 30,
        bounds=(-100, 100),
        tau1: float = 0.1,
        tau2: float = 0.1,
        Fl: float = 0.1,
        Fu: float = 0.9,
    ):
        self.func = func
        self.n_pop = n_pop
        self.dim = dim
        self.low, self.high = float(bounds[0]), float(bounds[1])
        self.tau1 = tau1
        self.tau2 = tau2
        self.Fl = Fl
        self.Fu = Fu

        self.population = None
        self.cost = None
        self.F = None  # mutation factor per individual
        self.CR = None  # crossover rate per individual
        self.best_cost = np.inf
        self.best_solution = None

    def optimize(self, max_iter: int, verbose: bool = False):
        self._init_population()
        self._evaluate()

        best_history = [self.best_cost]

        for gen in range(max_iter):
            self._adapt_parameters()

            for i in range(self.n_pop):
                # Mutation: DE/rand/1
                candidates = [j for j in range(self.n_pop) if j != i]
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
                mutant = self.population[r1] + self.F[i] * (self.population[r2] - self.population[r3])

                # Crossover: binomial
                j_rand = np.random.randint(self.dim)
                trial = np.where(
                    np.random.rand(self.dim) < self.CR[i],
                    mutant,
                    self.population[i],
                )
                trial[j_rand] = mutant[j_rand]  # ensure at least one dimension
                trial = np.clip(trial, self.low, self.high)

                # Selection
                trial_cost = self.func(trial.reshape(1, -1))
                if trial_cost[0] < self.cost[i]:
                    self.population[i] = trial
                    self.cost[i] = trial_cost[0]
                    if trial_cost[0] < self.best_cost:
                        self.best_cost = trial_cost[0]
                        self.best_solution = trial.copy()

            best_history.append(self.best_cost)

            if verbose and (gen + 1) % 100 == 0:
                print(f"jDE Gen {gen + 1:4d} | Best = {self.best_cost:.6e}")

        return np.array(best_history)

    def _init_population(self):
        self.population = np.random.uniform(self.low, self.high, (self.n_pop, self.dim))
        self.F = np.full(self.n_pop, 0.5)
        self.CR = np.full(self.n_pop, 0.9)

    def _evaluate(self):
        self.cost = self.func(self.population)
        self.best_idx = np.argmin(self.cost)
        self.best_cost = self.cost[self.best_idx]
        self.best_solution = self.population[self.best_idx].copy()

    def _adapt_parameters(self):
        """Self-adapt F and CR for each individual."""
        r1 = np.random.rand(self.n_pop)
        r2 = np.random.rand(self.n_pop)

        new_F = np.where(r1 < self.tau1, self.Fl + np.random.rand(self.n_pop) * self.Fu, self.F)
        new_CR = np.where(r2 < self.tau2, np.random.rand(self.n_pop), self.CR)

        self.F = np.clip(new_F, 0.1, 1.0)
        self.CR = np.clip(new_CR, 0.0, 1.0)
