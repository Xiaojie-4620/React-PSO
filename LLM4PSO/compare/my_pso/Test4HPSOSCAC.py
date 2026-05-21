import matplotlib.pyplot as plt
from LLM4PSO.compare.my_pso.pso_scac import H_PSO_SCAC
from opfunu.cec_based.cec2017 import *
import numpy as np

# Group A: Unimodal Functions
class Sphere:
    def eval(self, x):
        return sum(xi ** 2 for xi in x)

    def name(self):
        return "Sphere"

class Schwefel_2_22:
    def eval(self, x):
        return sum(abs(xi) for xi in x) + np.prod([abs(xi) for xi in x])

    def name(self):
        return "Schwefel_2_22"

class Schwefel_1_2:
    def eval(self, x):
        return sum(sum(x[:j+1]) ** 2 for j in range(len(x)))

    def name(self):
        return "Schwefel_1_2"

class Schwefel_2_21:
    def eval(self, x):
        return max(abs(xi) for xi in x)

    def name(self):
        return "Schwefel_2_21"

class Rosenbrock:
    def eval(self, x):
        return sum(100 * (x[i+1] - x[i]**2)**2 + (x[i] - 1)**2 for i in range(len(x) - 1))

    def name(self):
        return "Rosenbrock"

class Step:
    def eval(self, x):
        return sum((xi + 0.5)**2 for xi in x)

    def name(self):
        return "Step"

class Noise:
    def eval(self, x):
        return sum(i * x[i-1]**4 for i in range(1, len(x)+1)) + np.random.random()

    def name(self):
        return "Noise"

# Group B: Multimodal Functions
class Rastrigin:
    def eval(self, x):
        return sum(xi**2 - 10 * np.cos(2 * np.pi * xi) + 10 for xi in x)

    def name(self):
        return "Rastrigin"

class Ackley:
    def eval(self, x):
        n = len(x)
        sum1 = sum(xi**2 for xi in x)
        sum2 = sum(np.cos(2 * np.pi * xi) for xi in x)
        return -20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e

    def name(self):
        return "Ackley"

class Griewank:
    def eval(self, x):
        sum_part = sum(xi**2 for xi in x) / 4000
        prod_part = np.prod(np.cos(x[i] / np.sqrt(i+1)) for i in range(len(x)))
        return sum_part - prod_part + 1

    def name(self):
        return "Griewank"

class Penalized1:
    def eval(self, x):
        def u(xi, a, k, m):
            return k * (xi - a)**m if xi > a else (0 if abs(xi) <= a else k * (-xi - a)**m)
        n = len(x)
        y = 1 + (x + 1) / 4
        term1 = np.sin(np.pi * y[0])**2
        term2 = sum((y[i] - 1)**2 * (1 + 10 * np.sin(np.pi * y[i+1])**2) for i in range(n-1))
        term3 = (y[-1] - 1)**2
        penalty = sum(u(x[i], 10, 100, 4) for i in range(n))
        return (np.pi / n) * (term1 + term2 + term3) + penalty

    def name(self):
        return "Penalized1"

class Penalized2:
    def eval(self, x):
        def u(xi, a, k, m):
            return k * (xi - a)**m if xi > a else (0 if abs(xi) <= a else k * (-xi - a)**m)
        n = len(x)
        y = 1 + (x + 1) / 4
        term1 = np.sin(3 * np.pi * x[0])**2
        term2 = sum((x[i] - 1)**2 * (1 + np.sin(3 * np.pi * x[i+1])**2) for i in range(n-1))
        term3 = (x[-1] - 1)**2 * (1 + np.sin(2 * np.pi * x[-1])**2)
        penalty = sum(u(x[i], 5, 100, 4) for i in range(n))
        return 0.1 * (term1 + term2 + term3) + penalty

    def name(self):
        return "Penalized2"

# Test_func = [
#     Sphere(),
#     Schwefel_2_22(),
#     Schwefel_1_2(),
#     Schwefel_2_21(),
#     Rosenbrock(),
#     Step(),
#     Noise(),
#     Rastrigin(),
#     Ackley(),
#     Griewank(),
#     Penalized1(),
#     Penalized2()
# ]

# 参数设置
n_pop = 40
dim = 30
max_iter = 1000
position_bound = [-30, 30]
velocity_bound = [-6, 6]
func = Rosenbrock()
# 初始化算法
# for func in Test_func:
pso = H_PSO_SCAC(
    func=func,
    n_pop=n_pop,
    dim=dim,
    c1=2.0,
    c2=2.0,
    position_bound=position_bound,
    velocity_bound=velocity_bound,
    flag='standard'
)

# 初始化种群
pso.pop_init()

# 记录最优适应度
gbest_history = []

# 主循环
for it in range(max_iter):
    pso.evaluation()
    pso.update_p_best_cost()
    pso.update_g_best_cost()
    pso.update_c1_c2(it, max_iter)
    pso.update_inertia_weight()
    pso.update_pos_vel(it)
    gbest_history.append(pso.get_gBest())

# 输出结果
print("最优解:", pso.g_Best_Position)
print("最优值:", pso.g_Best_Cost)

# 可视化收敛曲线
plt.plot(gbest_history)
plt.yscale('log')
plt.xlabel("Iteration")
plt.ylabel("Best Fitness (log scale)")
plt.title(f"H-PSO-SCAC on {func.name()} Function")
plt.grid(True)
plt.show()