import numpy as np
from ..Population import Population


"""
 OperatorDE - The operator of differential evolution.
 Off = OperatorDE(P1,P2,P3,problem) uses the operator of differential evolution
 to generate offsprings based on the parents P1, P2, and P3. If P1, P2,
 and P3 are arrays of Population objects, then Off is also an array of
 Population objects; while if P1, P2, and P3 are matrices of decision
 variables, then Off is also a matrix of decision variables, i.e., the
 offsprings are not evaluated. Each object or row of P1, P2, and P3 is
 used to generate one offspring by P1 + 0.5*(P2-P3) and polynomial
 mutation.

 Off = OperatorDE(P1,P2,P3,problem,CR,F,proM,disM) specifies the parameters of
 operators, where CR and F are the parameters in differental evolution,
 proM is the expectation of the number of mutated variables, and disM is
 the distribution index of polynomial mutation.

 Example:
       Off = OperatorDE(Parent1,Parent2,Parent3,problem)
       Off = OperatorDE(Parent1.decs,Parent2.decs,Parent3.decs,problem,1,0.5,1,20)  # noqa

------------------------------------ Reference --------------------------------
 H. Li and Q. Zhang, Multiobjective optimization problems with complicated
 Pareto sets, MOEA/D and NSGA-II, IEEE Transactions on Evolutionary
 Computation, 2009, 13(2): 284-302.
 -----------------------------------------------------------------------------------
"""


def OperatorDE(Parent1, Parent2, Parent3, problem, *args):
    if len(args) > 3:
        CR = args[0]
        F = args[1]
        proM = args[2]
        disM = args[3]
    else:
        CR = 1
        F = 0.5
        proM = 1
        disM = 20
    if isinstance(Parent1, Population):
        calobj = True
        Parent1 = Parent1.decs
    else:
        calobj = False

    if isinstance(Parent2, Population):
        calobj = True
        Parent2 = Parent2.decs
    else:
        calobj = False

    if isinstance(Parent3, Population):
        calobj = True
        Parent3 = Parent3.decs
    else:
        calobj = False

    N = Parent1.shape[0]
    D = Parent1.shape[1]
    """
    Differental evolution
    """
    Site = np.random.random((N, D)) < CR
    Offspring = Parent1.copy()
    Offspring[Site] = Offspring[Site] + F * (Parent2[Site] - Parent3[Site])

    """
    Polynomial mutation
    """
    Lower = np.tile(problem.lb, (N, 1))
    Upper = np.tile(problem.ub, (N, 1))
    Site = np.random.random((N, D)) < proM / D
    mu = np.random.random((N, D))
    temp = np.logical_and(Site, mu <= 0.5)
    Offspring = np.minimum(np.maximum(Offspring, Lower), Upper)
    Offspring[temp] = Offspring[temp] + (Upper[temp] - Lower[temp]) * (
        (
            2 * mu[temp]
            + (1 - 2 * mu[temp])
            * (
                1
                - (Offspring[temp] - Lower[temp]) / (Upper[temp] - Lower[temp])
            )  # noqa
            ** (disM + 1)
        )  # noqa
        ** (1 / (disM + 1))
        - 1
    )  # noqa
    temp = np.logical_and(Site, mu > 0.5)  # noqa: E510
    Offspring[temp] = Offspring[temp] + (Upper[temp] - Lower[temp]) * (
        1
        - (  # noqa
            2 * (1 - mu[temp])
            + 2
            * (mu[temp] - 0.5)
            * (
                1
                - (Upper[temp] - Offspring[temp])
                / (Upper[temp] - Lower[temp])  # noqa
            )  # noqa
            ** (disM + 1)
        )  # noqa
        ** (1 / (disM + 1))
    )  # noqa
    if calobj:  # noqa: E510
        Offspring = Population(decs=Offspring)
    return Offspring
