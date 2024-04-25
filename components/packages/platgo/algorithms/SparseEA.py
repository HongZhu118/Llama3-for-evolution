"""
------------------------------- Reference --------------------------------
 Y. Tian, X. Zhang, C. Wang, and Y. Jin, An evolutionary algorithm for
 large-scale sparse multi-objective optimization problems, IEEE
 Transactions on Evolutionary Computation, 2020, 24(2): 380-393.
"""

import numpy as np

from .. import GeneticAlgorithm, utils, operators, Population


class SparseEA(GeneticAlgorithm):
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
        max_fe=100000,
        name="SparseEA",
        show_bar=False,
        sim_req_cb=None,
        debug=False,
    ):
        super(SparseEA, self).__init__(
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
        Fitness = np.zeros(self.problem.n_var)
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
        N, D = ParentDec.shape
        Parent1Mask = ParentMask[: int(N / 2), :]
        Parent2Mask = ParentMask[int(N / 2):, :]
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
        for i in range(int(N / 2)):
            if np.random.random(1) < 0.5:
                index = np.argwhere(OffMask[i, :] != 0)
                if len(index) == 0:
                    OffMask
                else:
                    index = index[self.TS(-Fitness.flatten()[index])]
                    OffMask[i, index] = 0
            else:
                index = np.argwhere(np.logical_not(OffMask[i, :]) != 0)
                if len(index) == 0:
                    OffMask
                else:
                    index = index[self.TS(Fitness.flatten()[index])]
                    OffMask[i, index] = 1
        # Crossover and mutation for dec
        if REAL:
            OffDec = operators.OperatorGAhalf(ParentDec, self.problem)
        else:
            OffDec = np.ones((int(N / 2), D))

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
