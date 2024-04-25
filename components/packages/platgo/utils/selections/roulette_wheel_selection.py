import numpy as np


"""
 roulette_wheel_selection - Roulette-wheel selection.

    P = roulette_wheel_selection(N,pop,fitness) returns the indices of N
    solutions by roulette-wheel selection based on fitness. A smaller
    fitness value indicates a larger probability to be selected.

    Example:
        P = roulette_wheel_selection(100,pop,FrontNo)
"""


def roulette_wheel_selection(N, pop=None, fitness=None) -> np.ndarray:
    """
    :param N:需要选择的个体数目
    :param pop:种群
    :param fitness:多目标函数的适应度值
    :return:被选择个体的索引值
    """
    assert pop or np.all(fitness), "pop and fitness can't be both None"
    if pop is None:
        # 多目标函数应有自己的适应值
        fitness = fitness
    else:
        # 单目标的适应值就是函数目标值
        fitness = pop.objv
    # 将适应值转化为一维
    fitness = fitness.reshape(1, -1)[0]
    fitness = fitness + min(min(fitness), 0)
    fitness = 1/fitness
    fitness = fitness/np.sum(fitness)
    index = np.random.choice(np.arange(len(fitness)), N, replace=True, p=fitness)  # noqa
    return index
