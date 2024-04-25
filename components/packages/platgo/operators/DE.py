import numpy as np
from typing import Union

from ..Population import Population


class DE:

    """
    差分变异算子类

    当传入的是一个种群时，对整个种群做变异
    当传入的是3个一维矩阵时，视为3个个体，对个体做变异
    """

    def __init__(self, F: float = 0.5):
        self.F = F

    def _mut_individual(
        self, dec1: np.ndarray, dec2: np.ndarray, dec3: np.ndarray
    ):
        """
        对单个个体变异，参数是数组类型
        :param dec1:
        :param dec2:
        :param dec3:
        :return:
        """
        return dec1 + self.F * (dec2 - dec3)

    def _mut_pop(self, pop: Population) -> Population:
        """
        对种群内所有个体变异
        :param pop:
        :return:
        """
        decs1 = pop.decs.copy()
        decs2 = pop.decs.copy()
        decs3 = pop.decs.copy()
        np.random.shuffle(decs2)
        np.random.shuffle(decs3)
        new_decs = decs1 + self.F * (decs2 - decs3)
        new_pop = Population(decs=new_decs)

        return new_pop

    def _mut_three_pop(
        self, pop1: Population, pop2: Population, pop3: Population
    ) -> Population:
        decs1 = pop1.decs
        decs2 = pop2.decs
        decs3 = pop3.decs
        new_decs = decs1 + self.F * (decs2 - decs3)
        new_pop = Population(decs=new_decs)
        return new_pop

    def __call__(
        self,
        p1: Union[Population, np.ndarray],
        p2: Union[Population, np.ndarray] = None,
        p3: Union[Population, np.ndarray] = None,
    ) -> Union[Population, np.ndarray]:
        if p2 is None:  # mutate for a population
            return self._mut_pop(p1)
        elif isinstance(p2, np.ndarray):  # mutate for a individual
            return self._mut_individual(p1, p2, p3)
        else:  # mutate for three specified population
            return self._mut_three_pop(p1, p2, p3)
