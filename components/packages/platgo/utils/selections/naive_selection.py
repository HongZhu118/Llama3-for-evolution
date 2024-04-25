import numpy as np
from ...Population import Population


def naive_selection(pop: Population, N: int):
    idx = np.argsort(pop.objv, axis=0)
    idx = np.sort(idx[:N], axis=0)
    return idx
