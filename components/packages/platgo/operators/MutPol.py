import typing as t
import numpy as np

from ..Problem import Problem
from ..Population import Population


class MutPol:
    """

    多项式变异算子类

    $v'_k = v_k + \delta \cdot (up_k - low_k)$， 其中  # noqa

    $$
    \delta=\left\{\begin{array}{l}  # noqa
    {\left[2\mu+(1-2\mu)\left(1-\delta_{1}\right)^{\eta_{m}+1}\right]^{\frac{1}{\eta_{m}+1}}-1, \text{ if } \mu \leq 0.5} \\  # noqa
    1-\left[2\cdot(1-\mu)+2\cdot(\mu-0.5)\left(1-\delta_{2}\right)^{\eta_{m}+1}\right]^{\frac{1}{\eta_{m}+1}}, \text { if } \mu>0.5  # noqa
    \end{array}\right.  # noqa
    $$

    $\delta_1 = (v_k-low_k)/(up_k-low_k)$,   $\delta_2 = (up_k-v_k)/(up_k-low_k)$  # noqa
    $\{up,low\}$ 是变量的上下界，  $\mu$是区间 $[0,1]$ 的随机数，$\eta_m$ 是分布指数， $v$是父代。  # noqa
    """

    def __init__(self, problem: Problem, proM: int = 1, disM: int = 20) -> None:  # noqa: E501
        self.problem = problem
        self.lower = problem.lb
        self.upper = problem.ub
        # self.lower = problem.borders[0]
        # self.upper = problem.borders[1]
        self.proM = proM
        self.disM = disM

    def _mut_pop(self, pop: Population) -> Population:
        N, D = pop.decs.shape
        lower = np.tile(self.lower, (N, 1))
        upper = np.tile(self.upper, (N, 1))
        site = np.random.rand(N, D) < self.proM / D
        mu = np.random.rand(N, D)
        self.problem.fix_decs(pop)
        delta_1 = (pop.decs - lower) / (upper - lower)
        assert (delta_1 >= 0).all()
        delta_2 = (upper - pop.decs) / (upper - lower)
        assert (delta_2 >= 0).all()
        delta = (2 * mu + (1 - 2 * mu) * (1 - delta_1) ** (self.disM + 1)) ** (
            1 / (self.disM + 1)
        ) - 1
        temp = mu > 0.5
        delta[temp] = (
            1
            - (2 * (1 - mu) + 2 * (mu - 0.5) * (1 - delta_2) ** (self.disM + 1))  # noqa: E501
            ** (1 / (self.disM + 1))
        )[temp]
        decs = pop.decs + delta * (upper - lower)
        decs[~site] = pop.decs[~site]
        new_pop = Population(decs=decs)
        return new_pop

    def __call__(
        self, pop: t.Union[Population, np.ndarray]
    ) -> t.Union[Population, np.ndarray]:
        if type(pop) == np.ndarray:
            raise NotImplementedError("mutate for ndarray has not yet completed")  # noqa: E501
        elif type(pop) == Population:
            return self._mut_pop(pop)
        else:
            raise TypeError("the parameter must be population or ndarray")
