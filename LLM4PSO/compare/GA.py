"""
Example: use Geatpy (GA) to optimize a CEC2014 benchmark function from OPFUNU.
Requirements:
    pip install geatpy opfunu numpy

Notes:
- The user wrote "greatpy" but for genetic algorithms the commonly used package is "geatpy" (PyPI: geatpy).
- OPFUNU provides CEC2014 benchmark functions; we use opfunu.cec_based.cec2014.F12014 as an example.
- Depending on geatpy / opfunu versions, small API differences may exist (encoding/Population constructor). Adjust if needed.
"""

import numpy as np
import greatpy as ea
from opfunu.cec_based.cec2014 import F12014

class CEC2014Problem(ea.Problem):
    """Wrap an OPFUNU CEC2014 function in a geatpy Problem.
    This implementation assumes a single-objective minimization problem.
    """
    def __init__(self, func):
        name = func.__class__.__name__
        M = 1  # number of objectives
        maxormins = [1] * M  # 1 for minimization
        Dim = func.ndim
        # varTypes: 0 means continuous
        varTypes = [0] * Dim
        # opfunu provides ranges/bounds via lb/ub and also has 'ranges' & 'borders'
        lb = func.lb.tolist()
        ub = func.ub.tolist()
        # geatpy expects 'ranges' and 'borders' attributes on the problem
        ranges = [[lb[i], ub[i]] for i in range(Dim)]
        # lbin/ubin indicate whether boundaries are included (1 included)
        lbin = [1] * Dim
        ubin = [1] * Dim

        # Save the opfunu function instance for evaluation
        self.func = func

        # Call parent constructor
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

        # Also set ranges/borders attributes expected by crtfld
        # geatpy uses attributes .ranges and .borders on the Problem instance
        self.ranges = np.array(ranges).T.tolist()  # shape (2, Dim) typically transposed
        # borders: list of [lbin, ubin] per variable, but geatpy examples use problem.borders
        self.borders = [lbin, ubin]

    def aimFunc(self, pop):
        """Calculate objective values for a population.
        pop.Phen is the phenotype matrix: shape (NIND, Dim)
        Must set pop.ObjV to a numpy array of shape (NIND, M)
        """
        Vars = pop.Phen  # phenotype matrix
        NIND = Vars.shape[0]
        ObjV = np.zeros((NIND, 1))
        for i in range(NIND):
            x = Vars[i, :]
            # opfunu expects a 1D numpy array
            val = float(self.func.evaluate(np.asarray(x)))
            ObjV[i, 0] = val
        pop.ObjV = ObjV


def run_ga_on_cec1(ndim=30, pop_size=50, max_gen=200, seed=1):
    np.random.seed(seed)

    # instantiate the CEC function (F1 in this example)
    func = F12014(ndim=ndim)

    # wrap in geatpy Problem
    problem = CEC2014Problem(func)

    # Encoding: 'R' or 'RI' for real/integer mix. Here pure real -> 'R'
    Encoding = 'R'
    NIND = pop_size

    # Create the field descriptor (Encoding, varTypes, ranges, borders)
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)

    # Instantiate a Population object
    population = ea.Population(Encoding, Field, NIND)

    # Choose a GA template (SEGA: strengthened elitist GA template)
    myAlgorithm = ea.soea_SEGA_templet(problem, population)
    myAlgorithm.MAXGEN = max_gen
    myAlgorithm.logTras = 1
    myAlgorithm.verbose = True
    myAlgorithm.drawing = 0  # set 1 to see plots (matplotlib required)

    # Run the algorithm
    NDSet = myAlgorithm.run()

    # Finalize and get best individual
    BestIndi = myAlgorithm.finishing(population)

    # BestIndi is the best individual (may be a Population object); try to extract phenotype and objective
    try:
        best_x = BestIndi.Phen[0, :]
        best_f = BestIndi.ObjV[0, 0]
    except Exception:
        # fallback: evaluate the best from population
        # population might have been updated in-place; search for minimum ObjV
        pop = population
        idx = np.argmin(pop.ObjV[:, 0])
        best_x = pop.Phen[idx, :]
        best_f = pop.ObjV[idx, 0]

    print(f"Best objective: {best_f}")
    print(f"Best solution (first 10 dims): {best_x[:10]}")
    return best_x, best_f


if __name__ == '__main__':
    # Example run
    best_x, best_f = run_ga_on_cec1(ndim=30, pop_size=60, max_gen=100, seed=42)
    print("Done.")
