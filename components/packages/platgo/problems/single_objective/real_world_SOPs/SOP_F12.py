import numpy as np
from ....Problem import Problem


class SOP_F12(Problem):
    type = {
        "n_obj": "single",
        "encoding": "real",
        "special": "expensive/none",
    }

    def __init__(self, in_optimization_problem={}) -> None:
        optimization_problem = {
            "name": "SOP_F12",
            "encoding": "real",
            "n_var": 30,
            "lower": "-50",
            "upper": "50",
            "n_obj": 1,
            "initFcn": [],
            "decFcn": [],
            "conFcn": [],
            "objFcn": [],
        }
        optimization_problem.update(in_optimization_problem)
        super(SOP_F12, self).__init__(optimization_problem)

    def compute(self, pop) -> None:
        objv = np.zeros((pop.decs.shape[0], 1))
        for i in range(pop.decs.shape[0]):
            objv[i, 0] = np.pi / 30 * (
                10 * np.sin(np.pi * pop.decs[i, 0]) ** 2
                + np.sum(
                    ((pop.decs[i, : ((pop.decs).shape[1] - 1)] - 1) ** 2)
                    * (1 + (10 * (np.sin(np.pi * pop.decs[i, 1:])) ** 2))
                )
                + ((pop.decs[i, (pop.decs).shape[1] - 1] - 1) ** 2)
            ) + np.sum(self.u(np.array([pop.decs[i, :]]), 10, 100, 4))
        pop.objv = objv
        pop.finalresult = np.zeros((pop.decs.shape[0], 1))
        pop.cv = np.zeros((pop.decs.shape[0], 1))

    def u(self, X, a, k, m):
        temp = np.abs(X) > a
        X[temp] = k * (np.abs(X[temp]) - a) ** m
        X[~temp] = 0
        return X
