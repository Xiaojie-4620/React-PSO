import numpy as np

class Func:
    def eval(self, x):
        return np.sum(x**2)

    def to_str(self):
        return "sum(x**2)"

    def name(self):
        return "Sum of squares"
