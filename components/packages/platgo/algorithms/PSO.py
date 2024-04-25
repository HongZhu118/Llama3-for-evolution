import numpy as np

from components.packages.platgo import operators, utils

from .. import GeneticAlgorithm


class PSO(GeneticAlgorithm):
    type = {
        "n_obj": "single",
        "encoding": "real",
        "special": {"large/none", "constrained/none"}
    }

    def __init__(
        self,
        pop_size,
        options,
        optimization_problem,
        control_cb,
        max_fe=10000,
        name="PSO",
        show_bar=False,
        sim_req_cb=None,
        debug=False,
        W=0.4
    ):
        super(PSO, self).__init__(
            pop_size,
            options,
            optimization_problem,
            control_cb,
            max_fe=10000,
            name=name,
            show_bar=show_bar,
            sim_req_cb=sim_req_cb,
            debug=debug
        )
        self.W = W

    def run_algorithm(self):
        W = self.W
        Pop = self.problem.init_pop()
        self.cal_obj(Pop)
        Pbest = Pop.copy()
        best = np.argmin(utils.fitness_single(Pbest))
        Gbest = Pbest[np.array([best])]
        while self.not_terminal(Pop):
            Pop = operators.OperatorPSO(Pop, Pbest, Gbest, W)
            self.cal_obj(Pop)
            replace = utils.fitness_single(Pbest) > utils.fitness_single(Pop)
            Pbest[replace] = Pop[replace]
            best = np.argmin(utils.fitness_single(Pbest))
            Gbest = Pbest[np.array([best])]
            print(np.min(Pop.objv))
        return Pop
