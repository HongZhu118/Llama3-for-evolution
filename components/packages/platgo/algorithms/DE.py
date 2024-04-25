from ..utils.selections.tournament_selection import tournament_selection
from ..operators.OperatorDE import OperatorDE
from ..utils.fitness_single import fitness_single
from ..GeneticAlgorithm import GeneticAlgorithm


class DE(GeneticAlgorithm):
    type = {
        "n_obj": "single",
        "encoding": "real",
        "special": {"large/none", "constrained/none"},
    }

    def __init__(
        self,
        pop_size,
        options,
        optimization_problem,
        simulation_request_callback,
        max_fe=100000,
        CR=1,
        F=0.5,
        name=None,
        show_bar=False,
        sim_req_cb=None,
        debug=False,
    ) -> None:

        super(DE, self).__init__(
            pop_size,
            options,
            optimization_problem,
            simulation_request_callback,
            max_fe=max_fe,
            name=name,
            show_bar=show_bar,
            sim_req_cb=sim_req_cb,
            debug=debug,
        )
        self.CR = CR if CR is not None else 0.9
        self.F = F if F is not None else 0.5

    def run_algorithm(self):
        """
         main function for Different Evolution
         if population is None, generate a new population with N
        :param N: population size
        :param population: population to be optimized
        :return:
        """
        pop = self.problem.init_pop()
        self.cal_obj(pop)

        while self.not_terminal(pop):
            matingpool = tournament_selection(
                2, 2 * self.problem.pop_size, fitness_single(pop)
            )  # noqa
            offspring = OperatorDE(
                pop,
                pop[matingpool[: self.problem.pop_size]],
                pop[matingpool[self.problem.pop_size :]],  # noqa
                self.problem,
                self.CR,
                self.F,
                1,
                20,
            )  # noqa
            self.cal_obj(offspring)  # 计算子代种群目标函数值
            replace = fitness_single(pop) > fitness_single(offspring)
            pop[replace] = offspring[replace]

        return pop
