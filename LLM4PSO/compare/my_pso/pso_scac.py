import numpy as np

class H_PSO_SCAC:
    def __init__(self, func, n_pop, dim, c1, c2, position_bound: list, velocity_bound: list, flag):
        self.func = func  # 目标函数类
        self.n_pop = n_pop # 粒子数量
        self.dim = dim # 问题维度
        self.w = 1 # inertia weight
        # self.wdamp = wdamp # 惯性权重的阻尼系数
        self.c1 = c1 # 个体加速系数
        self.c2 = c2 # 社会加速系数
        self.position_bound = position_bound
        self.velocity_bound = velocity_bound
        # Initialization
        self.position = np.array([])
        self.velocity = np.array([])
        self.cost = None
        self.p_Best_Position = np.zeros((n_pop, dim))
        self.p_Best_Cost = np.array([np.inf for i in range(n_pop)])
        self.g_Best_Position = np.zeros(dim)
        self.g_Best_Cost = np.inf
        self.flag = flag # the problem type
        self.u0 = None # initial swarm particle average fitness values


    def pop_init(self):
        '''we apply OBL to initial particle's position
        '''
        position_raw = np.random.uniform(self.position_bound[0], self.position_bound[1], size=(self.n_pop, self.dim))
        lower_bound = np.full(self.dim, self.position_bound[0]) if np.isscalar(self.position_bound[0]) else np.array(self.position_bound[0])
        upper_bound = np.full(self.dim, self.position_bound[1]) if np.isscalar(self.position_bound[1]) else np.array(self.position_bound[1])
        position_opp = lower_bound + upper_bound - position_raw
        # merge
        position = np.vstack((position_raw, position_opp))
        # calculate fitness values
        if self.flag == 'cec':
            fit_cat = np.array([self.func.evaluate(pos) for pos in position])
        else:
            fit_cat = np.array([self.func.eval(pos) for pos in position])
        idx = np.argsort(fit_cat)[:self.n_pop] # Select the best n_pop particle
        self.position = position[idx]
        self.u0 = fit_cat[idx].mean() # update initial swarm particle average fitness values

        # initialization with normal distribution
        # for _ in range(self.n_pop):
        #     vel = np.random.uniform(self.velocity_bound[0], self.velocity_bound[1], size=self.dim)
        #     self.velocity.append(vel)
        # use zero array to initialization
        self.velocity = np.random.uniform(-0.1, 0.1, size=(self.n_pop, self.dim))
        # Inverse to np.ndarray
        # self.position = self.position if isinstance(self.position, np.ndarray) else np.array(self.position)
        # self.velocity = self.velocity if isinstance(self.velocity, np.ndarray) else np.array(self.velocity)
        # Initialize individual optimal position
        self.p_Best_Position = self.position

    def evaluation(self):
        ''' Calculate the fitness value of each particle and store it in cost.
        '''
        cost = []
        if self.flag == 'cec':
            for pos in self.position:
                cost.append(self.func.evaluate(pos))
            self.cost = cost if isinstance(cost, np.ndarray) else np.array(cost)
        else:
            for pos in self.position:
                cost.append(self.func.eval(pos))
            self.cost = cost if isinstance(cost, np.ndarray) else np.array(cost)

    def update_p_best_cost(self):
        """ Update p_best based on the fitness values obtained from the evaluation.
        """
        for i in range(self.n_pop):
            if self.cost[i] < self.p_Best_Cost[i]:
                self.p_Best_Cost[i] = self.cost[i]
                self.p_Best_Position[i] = self.position[i]

    def update_g_best_cost(self):
        """ Update g_best based on the fitness values obtained from the evaluation.
        """
        global_min_idx = np.argmin(self.p_Best_Cost)
        current = self.p_Best_Cost[global_min_idx]
        if current < self.g_Best_Cost:
            self.g_Best_Cost = current
            self.g_Best_Position = self.p_Best_Position[global_min_idx]

    def update_pos_vel(self, curr_it):
        """ Update the position and velocity of the particle swarm
        """
        # update velocity
        r1 = np.random.rand(self.dim)
        r2 = np.random.rand(self.dim)
        velocity = (self.w * self.velocity
                         + self.c1 * r1 * (self.p_Best_Position - self.position)
                         + self.c2 * r2 * (self.g_Best_Position - self.position))
        # Apply Velocity In Lower and Upper Bound Limits
        velocity = np.clip(velocity, self.velocity_bound[0], self.velocity_bound[1])

        # introducing dynamic weight, acceleration coefficient and best-so-far position to update the new position
        if self.flag == 'cec':
            fit = np.array([self.func.evaluate(pos) for pos in self.position])
        else:
            fit = np.array([self.func.eval(pos) for pos in self.position])
        u = fit.mean() if hasattr(self, 'u0') else fit.mean()
        if not hasattr(self, 'u0'):
            self.u0 = u
        wij = np.exp(fit / self.u0) / (1 + np.exp(-fit / self.u0) ** curr_it)
        wij = wij.reshape(-1, 1)
        wij_p = 1 - wij
        psi = wij
        rho = np.random.rand(self.n_pop, 1)
        wij = np.maximum(wij, 1e-3)
        # update position
        self.position = (wij * self.position + velocity * wij_p + rho * psi * self.g_Best_Position)
        # Apply Position In Lower and Upper Bound Limits
        self.position = np.clip(self.position, self.position_bound[0], self.position_bound[1])
        # record velocity
        self.velocity = velocity

    def get_gBest(self):
        return self.g_Best_Cost

    def update_c1_c2(self, it_curr, it_max):
        sigma = 2.0
        delta = 0.5
        ratio = it_curr / it_max
        c1 = sigma * np.sin((np.pi / 2) * (1 - ratio)) + delta
        c2 = sigma * np.cos((np.pi / 2) * (1 - ratio)) + delta
        self.c1, self.c2 = c1, c2

    def update_inertia_weight(self):
        """sine map is used to adjust the inertia weights ω of the PSO method during the search process.
        """
        c = np.random.uniform(3, 4, size=1)# control parameter default set 4.0; [0 < c <= 4.0]
        self.w = (c / 4.0) * np.sin(np.pi * self.w)


