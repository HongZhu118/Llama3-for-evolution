"""
------------------------------- Reference --------------------------------
 Y. Zhang, Y. Tian, and X. Zhang, Improved SparseEA for sparse large-scale
 multi-objective optimization problems, Complex & Intelligent Systems,
 2021.
"""

import numpy as np

from .. import GeneticAlgorithm, utils, Population


class SparseEA2(GeneticAlgorithm):
    type = {
        "n_obj": "multi",
        "encoding": {"real", "binary"},
        "special": {"large/none", "constrained/none", "sparse"},
    }

    def __init__(
        self,
        pop_size,
        options,
        optimization_problem,
        control_cb,
        max_fe=10000,
        name="SparseEA2",
        show_bar=False,
        sim_req_cb=None,
        debug=False,
    ):
        super(SparseEA2, self).__init__(
            pop_size,
            options,
            optimization_problem,
            control_cb,
            max_fe=max_fe,
            name=name,
            show_bar=show_bar,
            sim_req_cb=sim_req_cb,
            debug=debug,
        )

    def run_algorithm(self):
        # Calculate the fitness of each decision variable
        TDec = np.array([])
        TMask = np.array([])
        TempPop = np.array([])
        Fitness = np.zeros((self.problem.n_var))
        if self.problem.encoding != "binary":
            REAL = True
        for i in range(1 + 4 * int(REAL)):
            if REAL:
                Dec = np.random.uniform(
                    low=np.tile(self.problem.lb, (self.problem.n_var, 1)),
                    high=np.tile(self.problem.ub, (self.problem.n_var, 1)),
                    size=(self.problem.n_var, self.problem.n_var),
                )
            else:
                Dec = np.ones((self.problem.n_var, self.problem.n_var))
            Mask = np.eye(self.problem.n_var)
            Pop = Population(decs=Dec * Mask)
            self.cal_obj(Pop)
            if len(TDec) == 0:
                TDec = Dec
            else:
                TDec = np.vstack((TDec, Dec))
            if len(TMask) == 0:
                TMask = Mask
            else:
                TMask = np.vstack((TMask, Mask))
            if len(TempPop) == 0:
                TempPop = Pop
            else:
                TempPop = TempPop + Pop
            frontno, _ = utils.nd_sort(np.hstack((Pop.objv, Pop.cv)), np.inf)
            Fitness = Fitness + frontno
        # Generate initial Pop
        if REAL:
            Dec = np.random.uniform(
                low=np.tile(self.problem.lb, (self.problem.pop_size, 1)),
                high=np.tile(self.problem.ub, (self.problem.pop_size, 1)),
                size=(self.problem.pop_size, self.problem.n_var),
            )
        else:
            Dec = np.ones((self.problem.pop_size, self.problem.n_var))
        Mask = np.zeros((self.problem.pop_size, self.problem.n_var))
        for i in range(self.problem.pop_size):
            Mask[
                i,
                utils.tournament_selection(
                    2,
                    int(np.ceil(np.random.random(1) * self.problem.n_var)),
                    Fitness,
                ),
            ] = 1
        Pop = Population(decs=Dec * Mask)
        self.cal_obj(Pop)
        Pop, Dec, Mask, FrontNo, CrowdDis = self.EnvironmentalSelection(
            Pop + TempPop,
            np.vstack((Dec, TDec)),
            np.vstack((Mask, TMask)),
            self.problem.pop_size,
        )

        while self.not_terminal(Pop):
            MatingPool = utils.tournament_selection(
                2, 2 * self.problem.pop_size, FrontNo, -CrowdDis
            )
            OffDec, OffMask = self.Operator(
                Dec[MatingPool, :], Mask[MatingPool, :], Fitness, REAL
            )
            Offspring = Population(decs=(OffDec * OffMask))
            self.cal_obj(Offspring)
            [Pop, Dec, Mask, FrontNo, CrowdDis] = self.EnvironmentalSelection(
                Pop + Offspring,
                np.vstack((Dec, OffDec)),
                np.vstack((Mask, OffMask)),
                self.problem.pop_size,
            )
        return Pop

    def Operator(self, ParentDec, ParentMask, Fitness, REAL):
        # Parameter setting
        N = ParentDec.shape[0]
        Parent1Dec = ParentDec[: N // 2, :]
        Parent2Dec = ParentDec[N // 2: N // 2 * 2, :]
        Parent1Mask = ParentMask[: ParentMask.shape[0] // 2, :]
        Parent2Mask = ParentMask[
            ParentMask.shape[0] // 2: ParentMask.shape[0] // 2 * 2, :
        ]
        # Crossover and mutation for dec
        if REAL:
            OffDec, groupIndex, chosengroups = self.GLP_OperatorGAhalf(
                Parent1Dec, Parent2Dec, 4
            )
        # Crossover for mask
        OffMask = Parent1Mask.copy()
        for i in range(int(N / 2)):
            if np.random.random(1) < 0.5:
                temp_index = np.argwhere(
                    np.logical_and(
                        Parent1Mask[i, :], np.logical_not(Parent2Mask[i, :])
                    )
                    != 0
                )
                index = temp_index.flatten()
                if len(index) == 0:
                    OffMask
                else:
                    index = index[self.TS(-Fitness.flatten()[index])]
                    OffMask[i, index] = 0
            else:
                temp_index = np.argwhere(
                    np.logical_and(
                        np.logical_not(Parent1Mask[i, :]), Parent2Mask[i, :]
                    )
                    != 0
                )
                index = temp_index.flatten()
                if len(index) == 0:
                    OffMask
                else:
                    index = index[self.TS(Fitness.flatten()[index])]
                    OffMask[i, index] = Parent2Mask[i, index]
        # Mutation for mask
        if REAL:
            chosenindex = groupIndex == chosengroups
            for i in range(int(N / 2)):
                if np.random.random(1) < 0.5:
                    index = np.argwhere(
                        np.logical_and(OffMask[i, :], chosenindex[i, :])
                    )
                    if len(index) == 0:
                        OffMask
                    else:
                        index = index[self.TS(-Fitness.flatten()[index])]
                        OffMask[i, index] = 0
                else:
                    index = np.argwhere(
                        np.logical_and(
                            np.logical_not(OffMask[i, :]), chosenindex[i, :]
                        )
                    )
                    if len(index) == 0:
                        OffMask
                    else:
                        index = index[self.TS(Fitness.flatten()[index])]
                        OffMask[i, index] = 1
        return OffDec, OffMask

    def TS(self, Fitness):
        if len(Fitness) == 0:
            index = np.array([])
        else:
            index = utils.tournament_selection(2, 1, Fitness)
        return index

    def EnvironmentalSelection(self, Pop, Dec, Mask, N):
        # Delete duplicated solutions
        _, uni = np.unique(Pop.objv, return_index=True, axis=0)
        Pop = Pop[uni]
        Dec = Dec[uni, :]
        Mask = Mask[uni, :]
        N = np.minimum(N, len(Pop))
        # Non-dominated sorting
        FrontNo, MaxFNo = utils.nd_sort(Pop.objv, Pop.cv, N)
        Next = FrontNo < MaxFNo
        # Calculate the crowding distance of each solution
        CrowdDis = utils.crowding_distance(Pop.objv, FrontNo)
        # Select the solutions in the last front
        # based on their crowding distances
        temp_Last = np.argwhere(FrontNo == MaxFNo)
        Last = temp_Last.flatten()
        Rank = np.argsort(CrowdDis[Last])[::-1]
        Next[Last[Rank[: N - np.sum(Next)]]] = True
        # Population for next generation
        Pop = Pop[Next]
        FrontNo = FrontNo[Next]
        CrowdDis = CrowdDis[Next]
        Dec = Dec[Next, :]
        Mask = Mask[Next, :]
        return Pop, Dec, Mask, FrontNo, CrowdDis

    def GLP_OperatorGAhalf(self, Parent1, Parent2, numberOfGroups):
        # Parameter setting
        proC = 1
        disC = 20
        disM = 20
        N, D = Parent1.shape
        # Genetic operators for real encoding
        # Simulated binary crossover
        beta = np.zeros((N, D))
        mu = np.random.random((N, D))
        beta[mu <= 0.5] = (2 * mu[mu <= 0.5]) ** (1 / (disC + 1))
        beta[mu > 0.5] = (2 - 2 * mu[mu > 0.5]) ** (-1 / (disC + 1))
        beta = beta * (-1) ** np.random.randint(0, 2, (N, D))
        beta[np.random.random((N, D)) < 0.5] = 1
        beta[np.tile(np.random.random((N, 1)) > proC, (1, D))] = 1
        Offspring = np.vstack(
            ((Parent1 + Parent2) / 2 + beta * (Parent1 - Parent2) / 2)
        )
        # Polynomial mutation
        Lower = np.tile(self.problem.lb, (N, 1))
        Upper = np.tile(self.problem.ub, (N, 1))
        outIndexList, _ = self.CreateGroups(numberOfGroups, Offspring, D)
        chosengroups = np.random.randint(
            numberOfGroups, size=(outIndexList.shape[0], 1)
        )
        Site = outIndexList == chosengroups
        mu = np.random.random((N, 1))
        mu = np.tile(mu, (1, D))
        temp = np.logical_and(Site, mu <= 0.5)
        Offspring = np.minimum(np.maximum(Offspring, Lower), Upper)
        Offspring[temp] = Offspring[temp] + (Upper[temp] - Lower[temp]) * (
            (
                2 * mu[temp]
                + (1 - 2 * mu[temp])
                * (
                    1
                    - (Offspring[temp] - Lower[temp])
                    / (Upper[temp] - Lower[temp])
                )
                ** (disM + 1)
            )
            ** (1 / (disM + 1))
            - 1
        )
        temp = np.logical_and(Site, mu > 0.5)
        Offspring[temp] = Offspring[temp] + (Upper[temp] - Lower[temp]) * (
            1
            - (
                2 * (1 - mu[temp])
                + 2
                * (mu[temp] - 0.5)
                * (
                    1
                    - (Upper[temp] - Offspring[temp])
                    / (Upper[temp] - Lower[temp])
                )
                ** (disM + 1)
            )
            ** (1 / (disM + 1))
        )
        return Offspring, outIndexList, chosengroups

    def CreateGroups(self, numberOfGroups, xPrime, numberOfVariables):
        # Creat groups by ordered grouping
        outIndexArray = np.array([])
        numberOfGroupsArray = np.array([])
        noOfSolutions = xPrime.shape[0]
        for sol in range(noOfSolutions):
            varsPerGroup = numberOfVariables // numberOfGroups
            vars = xPrime[sol, :]
            I = np.argsort(vars)  # noqa E741
            outIndexList = np.ones((1, numberOfVariables)).flatten()
            for i in range(1, numberOfGroups):
                outIndexList[
                    I[((i - 1) * varsPerGroup): i * varsPerGroup]
                ] = i
            outIndexList[
                I[((numberOfGroups - 1) * varsPerGroup) + 1:]
            ] = numberOfGroups
            if len(outIndexArray) == 0:
                outIndexArray = outIndexList
            else:
                outIndexArray = np.vstack((outIndexArray, outIndexList))
            if len(numberOfGroupsArray) == 0:
                numberOfGroupsArray = np.array([numberOfGroups])
            else:
                numberOfGroupsArray = np.hstack(
                    (numberOfGroupsArray, np.array([numberOfGroups]))
                )
        return outIndexArray, numberOfGroupsArray
