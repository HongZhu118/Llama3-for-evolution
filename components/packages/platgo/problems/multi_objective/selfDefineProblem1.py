import numpy as np
from ...Problem import Problem
from ... import Population
from scipy.spatial.distance import cdist


class selfDefineProblem1(Problem):
    type = {
        "n_obj": {"multi"},
        "encoding": {"real"},
        "special": "none"
    }

    def __init__(self, in_optimization_problem={}) -> None:
        optimization_problem = {
            "name": "selfDefineProblem1",
            "encoding": "real",
            "n_var": 6,
            "lower": "0",
            "upper": "1",
            "n_obj": 2,
            "num_c": 6,
            "initFcn": [],
            "decFcn": [],
            "objFcn": [],  # noqa
            "conFcn": ["0"]
        }
        optimization_problem.update(in_optimization_problem)
        self.num_c = optimization_problem["num_c"]
        super(selfDefineProblem1, self).__init__(optimization_problem)

    def init_pop(self, N: int = None):
        if N is None:
            N = self.pop_size
        data = self.data[0]
        low, high = np.min(data[:, :-1]), np.max(data[:, :-1])
        dec = np.random.uniform(low, high, size=(N, 2 * self.num_c))
        return Population(decs=dec)

    def compute(self, pop) -> None:
        objv = []
        for dec in pop.decs:
            d = np.empty((len(self.data[0]), 0))
            for i in range(int(len(dec) / 2)):
                d = np.concatenate(
                    [d, cdist(np.array([[dec[2 * i], dec[2 * i + 1]]]), self.data[0][:, :-1]).reshape(-1, 1)],
                    axis=1)
            objv1 = np.sum(np.min(d, axis=1))
            c_mat = np.concatenate([self.data[0], np.argmin(d, axis=1).reshape(-1, 1)], axis=1)

            r_objv = []
            for i in range(int(len(dec) / 2)):
                r_objv.append(np.sum(c_mat[c_mat[:, -1] == i][:, 2]))
            objv2 = (max(r_objv) - min(r_objv))
            objv.append([objv1, objv2])
        pop.objv = np.array(objv)
        pop.cv = np.zeros((pop.pop_size, self.n_constr))
