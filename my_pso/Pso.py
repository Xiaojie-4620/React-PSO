import numpy as np


class PSO:
    def __init__(self, func, n_pop, dim, c1, c2, position_bound: list, velocity_bound: list, flag, w: float, wdamp: float):
        self.func = func  # 目标函数类
        self.n_pop = n_pop # 粒子数量
        self.dim = dim # 问题维度
        self.w = w # 惯性权重
        self.wdamp = wdamp # 惯性权重的阻尼系数
        self.c1 = c1 # 个体加速系数
        self.c2 = c2 # 社会加速系数
        self.position_bound = position_bound
        self.velocity_bound = velocity_bound
        # Initialization
        self.position = []
        self.velocity = []
        self.cost = None
        self.p_Best_Position = np.zeros((n_pop, dim))
        self.p_Best_Cost = np.array([np.inf for i in range(n_pop)])
        self.g_Best_Position = np.zeros(dim)
        self.g_Best_Cost = np.inf
        self.flag = flag

    def pop_init(self):
        '''
        :input: None
        :return: a numpy array for initial particle position and velocity
        '''
        # Randomly initialize the position of the particle swarm
        self.position = np.random.uniform(self.position_bound[0], self.position_bound[1], size = (self.n_pop, self.dim))

        # Randomly initialize the velocity of the particle swarm
        # self.velocity = np.random.uniform(self.velocity_bound[0], self.velocity_bound[1], size = (self.n_pop, self.dim))

        # Initialize the particle swarm velocity with all zeros
        self.velocity = np.zeros((self.n_pop, self.dim))

        # Inverse to np.ndarray
        self.position = self.position if isinstance(self.position, np.ndarray) else np.array(self.position)
        self.velocity = self.velocity if isinstance(self.velocity, np.ndarray) else np.array(self.velocity)

        # Initialize individual optimal position
        self.p_Best_Position = self.position.copy()
        self.p_Best_Cost = np.full(self.n_pop, np.inf)

    def evaluation(self, flag):
        ''' Calculate the fitness value of each particle and store it in cost.
        '''
        if flag == 'opfunu_cec':
            self.cost = np.array([self.func.evaluate(pos) for pos in self.position])
        else:
            self.cost = np.array([self.func.eval(pos) for pos in self.position])

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
            self.g_Best_Position = self.p_Best_Position[global_min_idx].copy()

    def update_pos_vel(self, omega):
        """ Update the position and velocity of the particle swarm
        """
        r1 = np.random.rand(self.n_pop, self.dim)
        r2 = np.random.rand(self.n_pop, self.dim)
        # update velocity
        self.velocity = (omega * self.velocity
                         + self.c1 * r1 * (self.p_Best_Position - self.position)
                         + self.c2 * r2 * (self.g_Best_Position - self.position))
        # Apply Velocity In Lower and Upper Bound Limits
        self.velocity = np.clip(self.velocity, self.velocity_bound[0], self.velocity_bound[1])

        # update position
        self.position = self.position + self.velocity
        # Apply Position In Lower and Upper Bound Limits
        self.position = np.clip(self.position, self.position_bound[0], self.position_bound[1])
    def get_gBest(self):
        return self.g_Best_Cost

    def run(self, max_iter=1000, verbose=True):
        """ 主循环 """
        self.pop_init()
        self.evaluation(self.flag)
        self.update_p_best_cost()
        self.update_g_best_cost()
    
        for it in range(max_iter):
            self.update_pos_vel(self.w)
            self.evaluation(self.flag)
            self.update_p_best_cost()
            self.update_g_best_cost()
    
            if verbose and (it + 1) % 100 == 0:
                print(f"Iter {it + 1:4d} | Best Cost = {self.g_Best_Cost:.6e}")
            self.w *= self.wdamp
    
        return self.g_Best_Position.copy(), self.g_Best_Cost
