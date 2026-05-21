import matplotlib.pyplot as plt
import numpy as np
from opfunu.cec_based.cec2014 import *
from LLM4PSO.compare.my_pso.function import Func
# from my_pso.Pso import PSO
from LLM4PSO.PSO import PSO
from LLM4PSO.cec2017_py.cec2017.functions import all_functions
from opfunu.cec_based.cec2017 import *
from LLM4PSO.compare.my_pso.pso_scac import H_PSO_SCAC

class ImplPso:
    """ 用于实现具体的PSO算法流程
    """
    def __init__(self, flag):
        self.flag = flag
        self.Global_best = []
    def plots(self,):
        plt.figure()
        plt.plot(np.arange(len(self.Global_best)), self.Global_best)
        plt.ylabel('Best Cost')
        plt.xlabel('Iterations')
        plt.show()
    Global_best = []

    def run_pso(self, func, n_pop, dim, w, max_iter, wdamp, c1, c2, position_bound, velocity_bound):
        pso = PSO(func, n_pop, dim, c1, c2, position_bound, velocity_bound)
        # Initialization
        pso.pop_init()
        pso.evaluation(self.flag)
        pso.update_g_best_cost()
        self.Global_best.append(pso.get_gBest())
        for i in range(1, max_iter):
            # 更新速度和位置
            pso.update_pos_vel(w)
            # 计算适应度值
            pso.evaluation(self.flag)
            pso.update_p_best_cost()
            pso.update_g_best_cost()
            # record the global best
            self.Global_best.append(pso.get_gBest())
            # display iteration info per 100 times
            if i%100 == 0:
                print(f"Iteration: {i} Best_Cost: {pso.get_gBest()} weight: {w}, C1 & C2:{pso.c1, pso.c2}")
            # update parameter's w
            w = w * wdamp

if __name__ == '__main__':
    func = Func()
    kappa = 1
    phi1 = 2.05
    phi2 = 2.05
    phi = phi1 + phi2
    chi = 2 * kappa / abs(2 - phi - np.sqrt(phi ** 2 - 4 * phi))
    pop_size = 50
    dim = 10
    max_iter = 1000
    w = 2
    wdamp = 0.99
    c1 = 2.05
    c2 = 2.05
    position_bound = np.array([-100, 100])
    velocity_bound = 0.2 * position_bound

    for func in all_functions:
        runner = ImplPso('cec')
        runner.run_pso(func, pop_size, dim, w, max_iter, wdamp, c1, c2, position_bound, velocity_bound)
        runner.plots()