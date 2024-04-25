import numpy as np


from ..common.commons import SimParams


def ZDT1(x):
    x = np.array(x)
    f1 = x[0]
    g = 1 + 9 * np.mean(x[1:])
    h = 1 - np.sqrt(f1 / g)
    f2 = g * h

    cv = 0
    out = {
        SimParams.OutputObjective: [f1, f2],
        SimParams.OutputConstraint: [cv]}

    return out


def ZDT2(x):
    out = 0
    return out


def ZDT3(x):
    out = 0
    return out


def ZDT4(x):
    out = 0
    return out


def ZDT5(x):
    out = 0
    return out


def ZDT6(x):
    out = 0
    return out


def CF1(x):
    x = np.array(x)

    j1 = np.arange(2, len(x), 2)
    j2 = np.arange(1, len(x), 2)
    objv1 = x[0] + 2 * np.mean(
        (x[j1] - x[0] ** (0.5 * (1 + 3 * (j1 - 1) / (5 - 2)))) ** 2)
    objv2 = 1 - x[:, 0] + 2 * np.mean(
        (x[j2] - x[0] ** (0.5 * (1 + 3 * (j2 - 1) / (5 - 2)))) ** 2)

    cv = 1 - objv1 - objv2 + np.abs(np.sin(10 * np.pi * (objv1 - objv2 + 1)))

    out = {
        SimParams.OutputObjective: [objv1, objv2],
        SimParams.OutputConstraint: [cv]}
    return out


def Schwefel(x):
    x = np.array(x)
    objv = -np.sum(x * np.sin(np.sqrt(abs(x))))

    cv = 0
    out = {SimParams.OutputObjective: [objv], SimParams.OutputConstraint: [cv]}
    return out


def SOP_F1(x):
    x = np.array(x)
    f1 = np.sum(np.square(x))

    cv = 0
    out = {SimParams.OutputObjective: [f1], SimParams.OutputConstraint: [cv]}
    return out


def SOP_F2(x):
    x = np.array(x)
    objv = np.sum(np.abs(x)) + np.prod(np.abs(x))

    cv = 0
    out = {SimParams.OutputObjective: [objv], SimParams.OutputConstraint: [cv]}
    return out


def SOP_F3(x):
    x = np.array(x)
    objv = np.sum(np.square(np.cumsum(x)))

    cv = 0
    out = {SimParams.OutputObjective: [objv], SimParams.OutputConstraint: [cv]}
    return out


def SOP_F4(x):
    x = np.array(x)
    objv = np.abs(x).max(1)

    cv = 0
    out = {SimParams.OutputObjective: [objv], SimParams.OutputConstraint: [cv]}
    return out


def SOP_F5(x):
    # TODO, need to convert
    x = np.array(x)
    objv = np.array(
        [
            np.sum(
                100 * (x[1:] - x[: len(x)] ** 2) ** 2 + (x[: len(x)] - 1) ** 2
            )
        ]
    ).T

    cv = np.zeros((objv.shape[0], 1))
    out = {SimParams.OutputObjective: objv, SimParams.OutputConstraint: cv}
    return out


def SOP_F6(x):
    x = np.array(x)
    objv = np.sum(np.floor(x + 0.5) ** 2)

    cv = 0
    out = {SimParams.OutputObjective: [objv], SimParams.OutputConstraint: [cv]}
    return out


def TSP(x):
    out = 0
    return out


def CVRP(x):
    out = 0
    return out


def DTLZ1(x):
    # TODO, need to convert
    M = 3
    D = M + 4
    g = 100 * (
        D
        - M
        + 1
        + np.sum(
            ((x[:, M:] - 0.5) ** 2 - np.cos(20 * np.pi * (x[:, M:] - 0.5))),
            axis=1,
            keepdims=True,
        )
    )
    ones_matrix = np.ones((x.shape[0], 1))
    objv = (
        0.5
        * np.tile(1 + g, (1, M))
        * np.fliplr(
            np.cumprod(np.hstack((ones_matrix, x[:, : M - 1])), axis=1)
        )
        * np.hstack((ones_matrix, 1 - x[:, range(M - 2, -1, -1)]))
    )
    cv = np.zeros((objv.shape[0], 1))
    out = {SimParams.OutputObjective: objv, SimParams.OutputConstraint: cv}
    return out


def DTLZ2(x):
    # TODO, need to convert
    M = 3
    # D = M + 9
    g = np.sum((x[:, M:] - 0.5) ** 2, axis=1, keepdims=True,)
    ones_matrix = np.ones((x.shape[0], 1))
    objv = (
        np.tile(1 + g, (1, M))
        * np.fliplr(
            np.cumprod(
                np.hstack((ones_matrix, np.cos(x[:, : M - 1] * np.pi / 2))),
                axis=1,
            )
        )  # noqa
        * np.hstack(
            (ones_matrix, np.sin(x[:, range(M - 2, -1, -1)] * np.pi / 2))
        )  # noqa
    )
    cv = np.zeros((objv.shape[0], 1))
    out = {SimParams.OutputObjective: objv, SimParams.OutputConstraint: cv}
    return out


def MMF1(x):
    out = 0
    return out


def MMF2(x):
    out = 0
    return out


def MMF3(x):
    out = 0
    return out


def MMF4(x):
    out = 0
    return out


def MMF5(x):
    out = 0
    return out


def MMF6(x):
    out = 0
    return out


def MMF7(x):
    out = 0
    return out


def MMF8(x):
    out = 0
    return out


def WFG1(x):
    out = 0
    return out


def WFG2(x):
    out = 0
    return out


def WFG3(x):
    out = 0
    return out


def WFG4(x):
    out = 0
    return out


def WFG5(x):
    out = 0
    return out


def WFG6(x):
    out = 0
    return out


def WFG7(x):
    out = 0
    return out


def WFG8(x):
    out = 0
    return out


def WFG9(x):
    out = 0
    return out


def MAF10(x):
    out = 0
    return out


def MAF11(x):
    out = 0
    return out


def MAF12(x):
    out = 0
    return out


def Sparse_CN(x):
    out = 0
    return out


def Sparse_CD(x):
    out = 0
    return out


def CDsingle(x):
    out = 0
    return out


def LSMOP1(x):
    # TODO, need to convert
    # Parameter setting
    nk = 5
    M = 3
    D = 100 * M
    # lower = np.zeros((1, D))
    # upper = np.hstack((np.ones((1, M - 1)), 10 * np.ones((1, D - M + 1))))
    # Calculate the number of variables in each subcomponent
    c = np.array([3.8 * 0.1 * (1 - 0.1)])
    for i in range(1, M):
        c = np.hstack((c, 3.8 * c[c.shape[0] - 1] * (1 - c[c.shape[0] - 1])))
    sublen = np.floor(c / np.sum(c) * (D - M + 1) / nk)
    len = np.hstack((0, np.cumsum(sublen * nk)))
    # Calculate objective values
    N, D = x.shape
    x[:, np.arange(M - 1, D)] = (
        1 + np.tile(np.arange(M - 1, D) / D, (N, 1))
    ) * x[:, np.arange(M - 1, D)] - np.tile(
        np.array([x[:, 0] * 10]).T, (1, D - M + 1)
    )
    G = np.zeros((N, M))
    for i in range(0, M + 1, 2):
        for j in range(1, nk + 1):
            G[:, i] = G[:, i] + np.sum(
                (
                    x[
                        :,
                        int(len[i])
                        + M
                        - 1
                        + (j - 1) * int(sublen[i])
                        + 1: int(len[i])
                        + M
                        - 1
                        + j * int(sublen[i])
                        + 1,
                    ]
                )
                ** 2,
                axis=1,
            )
    for i in range(1, M, 2):
        for j in range(1, nk + 1):
            G[:, i] = G[:, i] + np.sum(
                (
                    x[
                        :,
                        int(len[i])
                        + M
                        - 1
                        + (j - 1) * int(sublen[i])
                        + 1: int(len[i])
                        + M
                        - 1
                        + j * int(sublen[i])
                        + 1,
                    ]
                )
                ** 2,
                axis=1,
            )
    G = G / np.tile(sublen, (N, 1)) / nk
    objv = (
        (1 + G)
        * np.fliplr(
            np.cumprod(np.hstack((np.ones((N, 1)), x[:, : M - 1])), axis=1)
        )
        * np.hstack((np.ones((N, 1)), 1 - x[:, range(M - 2, -1, -1)]))
    )
    cv = np.zeros((objv.shape[0], 1))
    out = {SimParams.OutputObjective: objv, SimParams.OutputConstraint: cv}
    return out


def SMOP1(x):
    # TODO, need to convert
    theta = 0.1
    M = 2
    D = 100
    decs = x
    K = np.ceil(theta * (D - M + 1)).astype(int)
    g = np.sum(
        g1(decs[:, M - 1: M + K - 1], np.pi / 3), axis=1, keepdims=True
    ) + np.sum(
        g2(decs[:, M + K - 1:], 0), axis=1, keepdims=True
    )  # noqa
    objv = (
        np.tile(1 + g / (D - M + 1), (1, M))
        * np.fliplr(
            np.cumprod(
                np.hstack(
                    (np.ones(shape=(decs.shape[0], 1)), decs[:, 0: M - 1])
                ),
                axis=1,
            )
        )
        * np.hstack(
            (np.ones(shape=(decs.shape[0], 1)), 1 - decs[:, M - 2:: -1])
        )
    )  # noqa
    cv = np.zeros((objv.shape[0], 1))
    out = {SimParams.OutputObjective: objv, SimParams.OutputConstraint: cv}
    return out


def g1(x, t):
    return (x - t) ** 2


def g2(x, t):
    return 2 * (x - t) ** 2 + np.sin(2 * np.pi * (x - t)) ** 2


def DOC1(x):
    # TODO, need to convert
    objv1 = x[0]
    objv2 = (
        5.3578547 * x[3] ** 2
        + 0.8356891 * x[1] * x[5]
        + 37.293239 * x[1]
        - 40792.141
        + 30665.5386717834
        + 1
    ) * (
        1
        - np.sqrt(x[0])
        / (
            5.3578547 * x[3] ** 2
            + 0.8356891 * x[1] * x[5]
            + 37.293239 * x[1]
            - 40792.141
            + 30665.5386717834
            + 1
        )
    )  # noqa

    cv1 = np.maximum(-(objv1 ** 2 + objv2 ** 2 - 1), 0)  # noqa
    cv2 = (
        85.334407
        + 0.0056858 * x[2] * x[5]
        + 0.0006262 * x[1] * x[4]
        - 0.0022053 * x[3] * x[5]
        - 92
    )  # noqa
    cv3 = (
        -85.334407
        - 0.0056858 * x[2] * x[5]
        - 0.0006262 * x[1] * x[4]
        + 0.0022053 * x[3] * x[5]
    )  # noqa
    cv4 = (
        80.51249
        + 0.0071317 * x[2] * x[5]
        + 0.0029955 * x[1] * x[2]
        + 0.0021813 * x[3] ** 2
        - 110
    )  # noqa
    cv5 = (
        -80.51249
        - 0.0071317 * x[2] * x[5]
        - 0.0029955 * x[1] * x[2]
        - 0.0021813 * x[3] ** 2
        + 90
    )  # noqa
    cv6 = (
        9.300961
        + 0.0047026 * x[3] * x[5]
        + 0.0012547 * x[1] * x[3]
        + 0.0019085 * x[3] * x[4]
        - 25
    )  # noqa
    cv7 = (
        -9.300961
        - 0.0047026 * x[3] * x[5]
        - 0.0012547 * x[1] * x[3]
        - 0.0019085 * x[3] * x[4]
        + 20
    )  # noqa

    objv = np.vstack((objv1, objv2)).T
    cv = np.vstack((cv1, cv2, cv3, cv4, cv5, cv6, cv7)).T
    out = {
        SimParams.OutputObjective: objv,
        SimParams.OutputConstraint: cv,
    }  # noqa
    return out


def C1_DTLZ1(x):
    # TODO, need to convert
    D = 7
    M = 3
    g = 100 * (
        D
        - M
        + 1
        + np.sum(
            (x[:, M - 1:] - 0.5) ** 2
            - np.cos(20 * np.pi * (x[:, M - 1:] - 0.5)),
            axis=1,
        )
    )
    objv = (
        0.5
        * np.tile(1 + g, (1, M))
        * np.fliplr(
            np.cumprod(
                np.hstack((np.ones((x.shape[0], 1)), x[:, : M - 1])), axis=1
            )
        )
        * np.hstack((np.ones((x.shape[0], 1)), 1 - x[:, M - 2: 0: -1]))
    )
    cv = objv[:, M - 1] / 0.6 + np.sum(objv[:, : M - 1] / 0.5, axis=1) - 1
    out = {
        SimParams.OutputObjective: objv,
        SimParams.OutputConstraint: cv,
    }  # noqa
    return out


def x2_function(x):
    # TODO, need to convert
    x = np.array(x)
    objv = np.array([np.square(x)]).T
    cv = np.zeros((objv.shape[0], 1))
    out = {SimParams.OutputObjective: objv, SimParams.OutputConstraint: cv}
    return out


def MW1(x):
    out = 0
    return out


def MW2(x):
    out = 0
    return out


def MW3(x):
    out = 0
    return out


def MW4(x):
    out = 0
    return out


def MW5(x):
    out = 0
    return out


def MW6(x):
    out = 0
    return out


def MW7(x):
    out = 0
    return out


def MW8(x):
    out = 0
    return out


def MW9(x):
    out = 0
    return out


def MW10(x):
    out = 0
    return out


def MW11(x):
    out = 0
    return out


def MW12(x):
    out = 0
    return out


def MW13(x):
    out = 0
    return out


def MW14(x):
    out = 0
    return out


def SMOP2(x):
    out = 0
    return out


def SMOP3(x):
    out = 0
    return out


def SMOP4(x):
    out = 0
    return out


def SMOP5(x):
    out = 0
    return out


def SMOP6(x):
    out = 0
    return out


def SMOP7(x):
    out = 0
    return out


def SMOP8(x):
    out = 0
    return out


def FSP(x):
    out = 0
    return out


def volume_func(sim_params):
    length = sim_params.pop("length")
    height = sim_params.pop("height")
    width = sim_params.pop("width")

    volume = length * height * width
    surface = 2 * (length*height + height*width + width*length)

    result = sim_params
    result.update({"Volume": volume, "Surface": surface})
    return result


PROBLEM_ROUTER = {
    "zdt1": ZDT1,
    "zdt2": ZDT2,
    "zdt3": ZDT3,
    "zdt4": ZDT4,
    "zdt5": ZDT5,
    "zdt6": ZDT6,
    "cf1": CF1,
    "schwefel": Schwefel,
    "sop_f1": SOP_F1,
    "sop_f2": SOP_F2,
    "sop_f3": SOP_F3,
    "sop_f4": SOP_F4,
    "sop_f5": SOP_F5,
    "sop_f6": SOP_F6,
    "dtlz1": DTLZ1,
    "dtlz2": DTLZ2,
    "tsp": TSP,
    "lsmop1": LSMOP1,
    "smop1": SMOP1,
    "smop2": SMOP2,
    "smop3": SMOP3,
    "smop4": SMOP4,
    "smop5": SMOP5,
    "smop6": SMOP6,
    "smop7": SMOP7,
    "smop8": SMOP8,
    "doc1": DOC1,
    "c1_dtlz1": C1_DTLZ1,
    "cf1": CF1,
    "x2": x2_function,
    "MW1": MW1,
    "MW2": MW2,
    "MW3": MW3,
    "MW4": MW4,
    "MW5": MW5,
    "MW6": MW6,
    "MW7": MW7,
    "MW8": MW8,
    "MW9": MW9,
    "MW10": MW10,
    "MW11": MW11,
    "MW12": MW12,
    "MW13": MW13,
    "MW14": MW14,
    "fsp": FSP,
}
