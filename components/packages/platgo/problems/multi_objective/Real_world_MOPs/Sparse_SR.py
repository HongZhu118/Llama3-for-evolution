import numpy as np
import os
from ....Problem import Problem
import scipy.io as sio
from scipy.linalg import orth


class Sparse_SR(Problem):
    type = {
        "n_obj": "multi",
        "encoding": "real",
        "special": {"large/none", "sparse/none", "expensive/none"},
    }

    def __init__(self, in_optimization_problem={}) -> None:
        optimization_problem = {
            "name": "Sparse_SR",
            "encoding": "real",
            "n_var": 1024,
            "lower": "-1",
            "upper": "1",
            "n_obj": 2,
            "lenSig": 1024,
            "lenObs": 480,
            "sparsity": 260,
            "sigma": 0,
            "initFcn": [],
            "decFcn": [],
            "objFcn": [],
            "conFcn": [],
        }
        optimization_problem.update(in_optimization_problem)
        N = optimization_problem["lenSig"]
        M = optimization_problem["lenObs"]
        K = optimization_problem["sparsity"]
        datasetName = "Dataset_SR-N{:d}-M{:d}-K{:d}-sigma{:.2f}.mat".format(  # noqa
            N, M, K, optimization_problem["sigma"]
        )
        fileName = "D:\\mathmodel\\mathmodel2023\\mathmodel2023\\components\\resources\\Real_world_MOPs\\"+"Dataset_SR-N{:d}-M{:d}-K{:d}-sigma{:.2f}.mat".format(
            N, M, K, optimization_problem["sigma"])

        if os.path.exists(fileName):
            load_data = sio.loadmat(fileName)
            self.A = load_data["A"]
            self.b = load_data["b"]
            self.x_true = load_data["x_true"]
        else:
            # Generate dataset
            self.A, self.b, self.x_true = self.inst_gen(
                N, M, K, optimization_problem["sigma"]
            )
            sio.savemat(
                fileName,
                {
                    "A": self.A,
                    "b": self.b,
                    "x_true": self.x_true,
                    "N": N,
                    "M": M,
                    "K": K,
                    "sigma": optimization_problem["sigma"],
                },
            )

        # Parameter setting
        self.n_obj = 2
        self.n_var = optimization_problem["n_var"] = N
        x = self.x_true[self.x_true != 0]
        self.lb = optimization_problem["lower"] = np.floor(
            np.tile(np.mean(x) - 3 * np.std(x), (1, self.n_var))
        )
        self.ub = optimization_problem["upper"] = np.ceil(
            np.ones((1, self.n_var)) * (np.mean(x) + 3 * np.std(x))
        )
        x = np.zeros((1, N))
        self.Total = np.linalg.norm(np.dot(self.A, x.T) - self.b)
        super(Sparse_SR, self).__init__(optimization_problem)

    def compute(self, pop) -> None:
        PopDec = pop.decs != 0
        PopObj = np.zeros((PopDec.shape[0], 2))
        N, D = np.shape(PopDec)

        PopObj[:, 0] = np.sum(PopDec != 0, 1) / D

        for i in range(N):
            PopObj[i, 1] = (
                np.linalg.norm(
                    np.dot(self.A, PopDec[i, :].T.reshape(-1, 1)) - self.b
                )
                / self.Total
            )

        pop.objv = PopObj
        pop.finalresult = np.zeros((pop.decs.shape[0], 1))
        pop.cv = np.zeros((pop.decs.shape[0], 1))

    def inst_gen(self, N, M, K, sigma):

        x_true = np.zeros((N, 1))
        q = np.random.permutation(N)

        x_true[q[0:K]] = 2 * np.random.rand(K, 1)

        err = sigma * np.random.rand(M, 1)

        A = np.random.rand(M, N)

        A = orth(A.T).T
        b = np.dot(A, x_true)

        normA = np.linalg.norm(A, ord=2)
        A = A / normA
        b = b / normA + err

        return A, b, x_true
