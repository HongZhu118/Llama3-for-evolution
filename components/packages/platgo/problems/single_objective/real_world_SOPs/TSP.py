import numpy as np
import os

import pandas as pd
from scipy.spatial.distance import cdist
from ....Problem import Problem

# import scipy.io as sio


class TSP(Problem):
    type = {
        "n_obj": "single",
        "encoding": "permutation",
        "special": "large/none",
    }

    def __init__(self, in_optimization_problem={}) -> None:
        # load_fn = os.path.join(
        #     os.path.dirname(__file__),
        #     "../../../../../../resources/real_world_SOPs/TSP-D30.mat")
        # if os.path.isfile(load_fn):
        #     load_data = sio.loadmat(load_fn)
        #     R = np.array(load_data["R"], dtype=float)
        # else:
        #     R = np.random.rand(in_optimization_problem.get("n_var"), 2)
        #     sio.savemat(load_fn, {"R": R})

        # load_data = sio.loadmat(load_fn)
        # R = load_data["R"]
        # self.C = cdist(R, R)
        optimization_problem = {
            "name": "TSP",
            "encoding": "permutation",
            "n_var": 30,
            "lower": "0",
            "upper": "1",
            "n_obj": 1,
            "initFcn": [],
            "decFcn": [],
            "objFcn": [],
            "conFcn": [],
        }
        optimization_problem.update(in_optimization_problem)
        str_temp = str(optimization_problem["n_var"])
        # path = os.path.join(
        #     os.path.dirname(__file__),
        #     "resources\\real_world_SOPs",
        # )  # noqa
        path = "D:\\LLM\\Llama\\components\\resources\\real_world_SOPs"
        path += "\\TSPdata-R" + str_temp + ".csv"
        if os.path.isfile(path):
            R = np.loadtxt(path, delimiter=",")
            self.C = cdist(R, R)
        else:
            R = np.random.rand(int(optimization_problem["n_var"]), 2)
            pd.DataFrame(R).to_csv(path, index=False)
            self.C = cdist(R, R)  # noqa
            np.savetxt(path, R, fmt="%f", delimiter=",")
        super(TSP, self).__init__(optimization_problem)

    def compute(self, pop) -> None:
        # load_fn = os.path.join(
        #     os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        #     "real_world_SOPs/dec.mat",
        # )
        # load_data = sio.loadmat(load_fn)
        # pop.decs = np.array(load_data["PopDec"], dtype=int)

        objv = np.zeros((pop.decs.shape[0], 1))
        for i in range(pop.decs.shape[0]):
            for j in range(pop.decs.shape[1] - 1):
                objv[i, 0] = (
                    objv[i, 0]
                    + self.C[int(pop.decs[i, j]), int(pop.decs[i, j + 1])]
                )
            objv[i, 0] = (
                objv[i, 0] + self.C[int(pop.decs[i, -1]), int(pop.decs[i, 0])]
            )
        pop.objv = objv
        # 下面这两行一定要加
        pop.finalresult=np.zeros((pop.decs.shape[0], 1))
        pop.cv = np.zeros((pop.decs.shape[0], 1))
