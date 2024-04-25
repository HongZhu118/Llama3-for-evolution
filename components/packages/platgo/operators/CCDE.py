import numpy as np


"""
 Differential evolution operator used in CCGDE3
"""


def CCDE(
    Parent1, Parent2, Parent3, lower, upper, *args
):
    if len(args) > 2:
        CR = args[0]
        F = args[1]
        proM = args[2]
        disM = args[3]
    else:
        CR = 0.5
        F = 0.5
        proM = 1
        disM = 20

    N = Parent1.shape[0]
    D = Parent1.shape[1]
    """
    Differental evolution
    """
    Site = np.random.rand(N, D) < CR
    Offspring = Parent1
    Offspring[Site] = Offspring[Site] + F * (Parent2[Site] - Parent3[Site])

    """
    Polynomial mutation
    """
    Lower = np.tile(lower, (N, 1))
    Upper = np.tile(upper, (N, 1))
    Site = np.random.random((N, D)) < proM / D
    mu = np.random.random((N, D))
    temp = np.logical_and(Site, mu <= 0.5)
    Offspring = np.minimum(np.maximum(Offspring, Lower), Upper)
    Offspring[temp] = Offspring[temp] + (Upper[temp] - Lower[temp]) * (
        (
            2 * mu[temp]
            + (1 - 2 * mu[temp])
            * (  # noqa
                1
                - (Offspring[temp] - Lower[temp]) / (Upper[temp] - Lower[temp])
            )
            ** (disM + 1)
        )
        ** (1 / (disM + 1))
        - 1
    )  # noqa
    temp = np.logical_and(Site, mu > 0.5)  # noqa: E510
    Offspring[temp] = Offspring[temp] + (Upper[temp] - Lower[temp]) * (
        1
        - (
            2 * (1 - mu[temp])
            + 2
            * (mu[temp] - 0.5)
            * (  # noqa
                1
                - (Upper[temp] - Offspring[temp]) / (Upper[temp] - Lower[temp])
            )
            ** (disM + 1)
        )
        ** (1 / (disM + 1))
    )  # noqa
    return Offspring # noqa