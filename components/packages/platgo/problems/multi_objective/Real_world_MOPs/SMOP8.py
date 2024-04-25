
import numpy as np
from ....Problem import Problem


# objFcn 返回的是一维数组
class SMOP8(Problem):
    type = {
        "n_obj": {"multi", "many"},
        "encoding": "real",
        "special": {"large/none", "sparse/none", "expensive/none"},
    }

    def __init__(self, in_optimization_problem={}) -> None:  # noqa
        optimization_problem = {
            "name": "SMOP8",
            "encoding": "real",
            "n_var": 100,
            "lower": "0",
            "upper": "1",
            "n_obj": 2,
            "initFcn": [],
            "decFcn": [],
            "conFcn": ["0"],
            "objFcn": [],
            "theta": 0.1,
        }
        optimization_problem.update(in_optimization_problem)
        if 0 <= optimization_problem["theta"] <= 1:
            self.theta = optimization_problem["theta"]
        elif optimization_problem["theta"] < 0:
            self.theta = 0
        else:
            self.theta = 1
        super(SMOP8, self).__init__(optimization_problem)
        self.lb = np.hstack((np.zeros(optimization_problem["n_obj"]-1), np.zeros(optimization_problem["n_var"] - optimization_problem["n_obj"] + 1) - 1))  # noqa
        self.ub = np.hstack((np.zeros(optimization_problem["n_obj"]-1)+1, np.zeros(optimization_problem["n_var"] - optimization_problem["n_obj"] + 1) + 2))  # noqa

    def compute(self, pop) -> None:
        X = pop.decs
        N = X.shape[0]
        D = X.shape[1]
        M = self.n_obj
        K = int(np.ceil(self.theta * (D - M + 1)))
        temp1 = np.sum(self.g3(X[:, M-1: M+K-1], np.mod(X[:, M:M+K]+np.pi, 2)), axis=1, keepdims=True)  # noqa
        temp2 = np.sum(self.g3(X[:, M+K-1: -1], 0.9*X[:, M+K:]), axis=1, keepdims=True)  # noqa
        g = temp1 + temp2
        ones_matrix = np.ones((N, 1))
        f = np.fliplr(np.cumprod(np.hstack([ones_matrix, np.cos(X[:, :self.n_obj - 1]*np.pi/2)]), axis=1)) * np.hstack(  # noqa
            [ones_matrix, np.sin(X[:, range(self.n_obj - 2, -1, -1)]*np.pi/2)]) * np.tile(1 + g/(self.n_var-self.n_obj+1), (1, self.n_obj))  # noqa
        pop.objv = f
        pop.finalresult= np.zeros((pop.pop_size, self.n_constr))
        cv = np.zeros((pop.pop_size, self.n_constr))
        pop.cv = cv

    def g3(self, x, t):
        g = 4 - (x - t) - 4/np.exp(100*(x-t)**2)
        return g
