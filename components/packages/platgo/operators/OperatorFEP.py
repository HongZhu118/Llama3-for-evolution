import numpy as np
import math
import scipy.stats as st
from ..Population import Population


"""
 OperatorFEP - The operator of fast evolutionary programming.

   Off = OperatorFEP(P) uses the operator of fast evolutionary programming
   to generate offsprings based on the parents P. P should be an array of
   Population objects, and Off is also an array of Population objects. Each
   object of P is used to generate one offspring.

   Example:
       Off = OperatorFEP(Population)

 ------------------------------------ Reference -------------------------------
 X. Yao, Y. Liu, and G. Lin, Evolutionary programming made faster, IEEE
 Transactions on Evolutionary Computation, 1999, 3(2): 82-102.
 ------------------------------------------------------------------------------
"""


def OperatorFEP(pop) -> Population:
    """
    Parameter setting
    """
    PopulationDec = pop.decs
    N = PopulationDec.shape[0]  # noqa
    D = PopulationDec.shape[1]
    PopulationEta = np.random.random(size=(N, D))

    """
    Fast evolutionary programming
    """
    tau = 1 / math.sqrt(2 * math.sqrt(D))
    tau1 = 1 / math.sqrt(2 * D)
    GaussianRand = np.tile(np.random.randn(N, 1), [1, D])
    GaussianRandj = np.random.randn(N, D)
    CauchyRandj = st.t.rvs(1, size=(N, D))
    OffDec = PopulationDec + PopulationEta * CauchyRandj
    OffEta = PopulationEta * np.exp(tau1 * GaussianRand + tau * GaussianRandj)
    Offspring = Population(decs=OffDec)

    return Offspring, OffEta
