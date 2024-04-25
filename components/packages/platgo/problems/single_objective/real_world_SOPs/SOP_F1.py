import numpy as np
from ....Problem import Problem


class SOP_F1(Problem):
    type = {
        "n_obj": "single",
        "encoding": "real",
        "special": "expensive/none",
    }

    def __init__(self, in_optimization_problem={}) -> None:
        optimization_problem = {
            "name": "SOP_F1",
            "encoding": "real",
            "n_var": 30,
            "lower": "-100",
            "upper": "100",
            "n_obj": 1,
            "initFcn": [],
            "decFcn": [],
            "conFcn": [],
            "objFcn": [],
        }
        optimization_problem.update(in_optimization_problem)
        super(SOP_F1, self).__init__(optimization_problem)

    def compute(self, pop) -> None:
        objv = np.zeros((pop.decs.shape[0], 1))
        for i in range(pop.decs.shape[0]):
            # objv[i, 0] = np.sum((pop.decs[i, :])**2)
            objv[i, 0] = np.sum(
                np.reshape(pop.decs[i, :] * pop.decs[i, :], (1, -1)), axis=1
            )
        pop.objv = objv
        pop.finalresult = np.zeros((pop.decs.shape[0], 1))
        pop.cv = np.zeros((pop.decs.shape[0], 1))
