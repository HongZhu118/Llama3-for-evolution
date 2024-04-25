# from pickle import TRUE
# from matplotlib import axis
import numpy as np
import os

# from sympy import re
from ....Problem import Problem
import scipy.io as sio
import networkx as nx
import matplotlib.pyplot as plt


class Sparse_CD(Problem):
    type = {
        "n_obj": "multi",
        "encoding": "label",
        "special": {"large/none", "sparse/none", "expensive/none"},
    }

    def __init__(self, in_optimization_problem={}) -> None:
        optimization_problem = {
            "name": "Sparse_CD",
            "encoding": "label",
            "n_var": 100,
            "lower": "0",
            "upper": "1",
            "n_obj": 2,
            "dataNo": 0,
            "initFcn": [],
            "decFcn": [],
            "objFcn": [],
            "conFcn": [],
        }
        '''
        load_fn = os.path.join(
            os.path.dirname(__file__),
            "../../../../../../resources/Real_world_MOPs/Dataset_CD.mat")
        '''
        # % Load data
        self.dataNo = optimization_problem["dataNo"]
        # load_data = sio.loadmat(load_fn)
        load_data = sio.loadmat("components/resources/Real_world_MOPs/Dataset_CD.mat")
        self.Dataset = load_data["Dataset"]
        #self.Dataset = load_data["PopDec"]
        string = ["Karate", "Dolphin", "Polbook", "Football"]
        self.Adj = self.Dataset[string[self.dataNo]][0, 0]
        temp = self.random_walk_distance(self.Adj)
        self.ACT = temp ** (-2)
        self.G = self.adMmatrix2Img(self.Adj)
        # % Parameter setting
        optimization_problem["n_obj"] = 2
        optimization_problem["n_var"] = self.Adj.shape[1]
        optimization_problem["upper"] = str(self.Adj.shape[1]-1)

        # % Maximum and minimum objective values for normalization
        C = self.Decoding(
            np.zeros((1, optimization_problem["n_var"])),
            self.ACT,
            optimization_problem["n_var"],
        )
        self.MaxKKM = self.KKM(self.Adj, C)
        self.MinRC = self.RC(self.Adj, C)
        C = self.Decoding(
            np.arange(self.Adj.shape[1]).reshape(1, -1),
            self.ACT,
            optimization_problem["n_var"],
        )
        self.MinKKM = self.KKM(self.Adj, C)
        self.MaxRC = self.RC(self.Adj, C)
        optimization_problem.update(in_optimization_problem)
        super(Sparse_CD, self).__init__(optimization_problem)

    def compute(self, pop) -> None:
        # PopDec = (pop.decs)!=0
        PopDec = pop.decs

        objv = np.zeros((pop.decs.shape[0], self.n_obj))
        for i in range(len(objv)):
            C = self.Decoding(
                PopDec[i, :].reshape(1, -1), self.ACT, self.n_var
            )
            objv[i, 0] = self.KKM(self.Adj, C)
            objv[i, 1] = self.RC(self.Adj, C)

        objv[:, 0] = (objv[:, 0] - self.MinKKM) / (self.MaxKKM - self.MinKKM)
        objv[:, 1] = (objv[:, 1] - self.MinRC) / (self.MaxRC - self.MinRC)

        pop.objv = objv
        cv = np.zeros((pop.decs.shape[0], 1))
        pop.cv = cv

    def RC(self, Adj, C):
        # % Calculate the ratio cut
        cs = 0
        de = 0
        clu_num = len(C)
        for i in range(clu_num):
            if i in C:
                s_index = C[i]
                s = np.zeros((len(s_index), len(s_index)), dtype=np.int64)
                for m in range(len(s_index)):
                    for n in range(len(s_index)):
                        s[m, n] = Adj[m, n]
                s_cardinality = len(s_index)
                if s_cardinality > 0:
                    kins_sum = 0
                    kouts_sum = 0
                    for j in range(s_cardinality):
                        kins = sum(s[j, :])
                        ksum = sum(Adj[s_index[j], :])
                        kouts = ksum - kins
                        kins_sum = kins_sum + kins
                        kouts_sum = kouts_sum + kouts
                        de = kouts_sum

                    cf_s = de * 1.0 / (s_cardinality)
                    cs = cs + cf_s
            else:
                continue
        return cs

    def KKM(self, Adj, C):
        # % Calculate the kernel k-means
        cf = 0
        ec = 0
        numVar = Adj.shape[0]
        clu_num = len(C)
        for i in range(clu_num):
            if i in C:
                s_index = C[i]
                s = np.zeros((len(s_index), len(s_index)), dtype=np.int64)
                for m in range(len(s_index)):
                    for n in range(len(s_index)):
                        s[m, n] = Adj[m, n]
                s_cardinality = len(s_index)
                if s_cardinality > 0:
                    kins_sum = 0
                    for j in range(s_cardinality):
                        kins = sum(s[j, :])
                        kins_sum = kins_sum + kins
                        ec = kins_sum
                    cf_s = ec * 1.0 / (s_cardinality)
                    cf = cf + cf_s
            else:
                continue
        cf = 2 * (numVar - len(C)) - cf
        return cf

    def Decoding(self, PopDec, ACT, M):
        PopDec = np.array(PopDec, dtype=int)
        C_Community = {}
        for i in range(M):
            if len(np.where(PopDec == i)[1]) > 0:
                C_Community[i] = np.where(PopDec == i)[1]
        return C_Community

    def adMmatrix2Img(self, matrix):
        plt.figure()
        G = nx.Graph()
        n = len(matrix)
        point = []
        for i in range(n):
            point.append(i)
        G.add_nodes_from(point)
        edglist = []
        for i in range(n):
            for k in range(i + 1, n):
                if matrix[i][k] > 0:
                    edglist.append((i, k))
        G.add_edges_from(edglist)
        position = nx.circular_layout(G)
        nx.draw_networkx_nodes(G, position, nodelist=point, node_color="y")
        nx.draw_networkx_edges(G, position)
        nx.draw_networkx_labels(G, position)
        # plt.show()
        return G

    def random_walk_distance(self, Adj):
        # % Random walk based distance
        n = len(Adj)
        Adj = np.array(Adj, dtype=int)
        D = np.diag(np.sum(Adj, axis=1))
        D = np.array(D, dtype=int)
        Laplace = D - Adj
        e = np.ones((n, 1))
        degree = np.sum(np.sum(Adj, axis=1))
        L_p = np.linalg.inv((Laplace - ((e * e.T) / n))) + ((e * e.T) / n)
        # L_p = (Laplace - ((e * e.T) / n)) **( -1 )+ ((e * e.T) / n)
        ACT = np.zeros((n, n))
        for i in range(n):
            AI = L_p[i, :].reshape(1, -1)
            for j in range(n):
                BI = L_p[j, :].reshape(1, -1)
                CI = AI - BI
                ACT[i, j] = degree * (CI[:, i] - CI[:, j])
                if i == j:
                    ACT[i, j] = 1

        return ACT
