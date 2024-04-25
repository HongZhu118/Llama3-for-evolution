import numpy as np
import math
from ..Population import Population
import re
import ollama
"""
 OperatorGA - Crossover and mutation operators of genetic algorithm.

    Off = OperatorGA(P,problem) uses genetic operators to generate offsprings
    based on the parents P. If P is an array of Population objects, then Off is
    also an array of Population objects; while if P is a matrix of decision
    variables, then Off is also a matrix of decision variables, i.e., the
    offsprings are not evaluated. P is split into two subsets P1 and P2
    with the same size, where each object or row of P1 and P2 is used to
    generate two offsprings. Different operators are used for real, binary,
    and permutation based encodings.

    Off = OperatorGA(P,problem,proC,disC,proM,disM) specifies the parameters of
    operators, where proC is the probability of crossover, disC is the
    distribution index of simulated binary crossover, proM is the
    expectation of the number of mutated variables, and disM is the
    distribution index of polynomial mutation.

    Example:
       Offspring = OperatorGA(Parent,problem)
       Offspring = OperatorGA(Parent.decs,problem,1,20,1,20)

    See also OperatorGAhalf

 ------------------------------------ Reference -------------------------------
 [1] K. Deb, K. Sindhya, and T. Okabe, Self-adaptive simulated binary
 crossover for real-parameter optimization, Proceedings of the Annual
 Conference on Genetic and Evolutionary Computation, 2007, 1187-1194.
 [2] K. Deb and M. Goyal, A combined genetic adaptive search (GeneAS) for
 engineering design, Computer Science and informatics, 1996, 26: 30-45.
 [3] L. Davis, Applying adaptive algorithms to epistatic domains,
 Proceedings of the International Joint Conference on Artificial
 Intelligence, 1985, 162-164.
 [4] D. B. Fogel, An evolutionary approach to the traveling salesman
 problem, Biological Cybernetics, 1988, 60(2): 139-144.
 ------------------------------------------------------------------------------
"""


def OperatorGA(pop, problem, *args) -> Population:
    if len(args) > 0:
        proC = args[0]
        disC = args[1]
        proM = args[2]
        disM = args[3]
    else:
        proC = 1
        disC = 20
        proM = 1
        disM = 20
    if isinstance(pop, Population):
        calobj = True
        pop = pop.decs
    else:
        calobj = False
    pop1 = pop[0 : math.floor(pop.shape[0] / 2), :]  # noqa
    pop2 = pop[
        math.floor(pop.shape[0] / 2) : math.floor(pop.shape[0] / 2)  # noqa
        * 2,  # noqa
        :,  # noqa
    ]  # noqa
    N = pop1.shape[0]  # noqa
    D = pop1.shape[1]

    if problem.encoding == "binary":
        """
         Genetic operators for binary encoding
         Uniform crossover
        """
        k = np.random.random((N, D)) < 0.5
        k[np.tile(np.random.random((N, 1)) > proC, (1, D))] = False
        Offspring1 = pop1.copy()
        Offspring2 = pop2.copy()
        Offspring1[k] = pop2[k]
        Offspring2[k] = pop1[k]
        Offspring = np.vstack((Offspring1, Offspring2))
        """
        Bit-flip mutation
        """
        Site = np.random.random((2 * N, D)) < proM / D
        Offspring[Site] = ~Offspring[Site].astype(bool)

    elif problem.encoding == "label":
        """
         Genetic operators for label encoding
         Uniform crossover
        """
        k = np.random.random((N, D)) < 0.5
        k[np.tile(np.random.random((N, 1)) > proC, (1, D))] = False
        Offspring1 = pop1.copy()
        Offspring2 = pop2.copy()
        Offspring1[k] = pop2[k]
        Offspring2[k] = pop1[k]
        Offspring = np.vstack((Offspring1, Offspring2))
        """
        Bit-flip mutation
        """
        Site = np.random.random((2 * N, D)) < proM / D
        Rand = np.random.randint(0, high=D, size=(2 * N, D))
        Offspring[Site] = Rand[Site]

        """
        Repair
        """
        for i in range(2 * N):
            Off = np.zeros((1, D)).astype(int)
            while not (Off > 0).all():
                fOff = (np.ones((Off.shape)) - Off).astype(int)
                if np.size(np.where(fOff > 0)[1]) > 0:
                    x = np.where(fOff > 0)[1][0]
                    Off[0, Offspring[i, :] == Offspring[i, x]] = (
                        np.max(Off) + 1
                    )
                else:
                    continue
            Offspring[i, :] = Off

    elif problem.encoding == "permutation":
        """
         Genetic operators for permutation based encoding
         Order crossover
        """
        Offspring = np.vstack((pop1, pop2))
        k = np.random.randint(0, high=D, size=2 * N)
        for i in range(N):  # noqa
            Offspring[i, k[i] + 1 :] = np.setdiff1d(  # noqa
                pop2[i, :], pop1[i, : k[i] + 1], True
            )  # noqa
            Offspring[i + N, k[i] + 1 :] = np.setdiff1d(  # noqa
                pop1[i, :], pop2[i, : k[i] + 1], True
            )  # noqa


        """
         Genetic operators for permutation based encoding
         Order crossover
        """
        # Offspring = np.vstack((pop1, pop2))
        # k = np.random.randint(0, high=D, size=2 * N)
        # for i in range(N):  # noqa
        #     Offspring[i, k[i] + 1 :] = np.setdiff1d(  # noqa
        #         pop2[i, :], pop1[i, : k[i] + 1], True
        #     )  # noqa
        #     Offspring[i + N, k[i] + 1 :] = np.setdiff1d(  # noqa
        #         pop1[i, :], pop2[i, : k[i] + 1], True
        #     )  # noqa
        # index = np.random.randint(0, 2*N)
        # Offspring = [pop[index]]
        # Offspring = []
        # while len(Offspring) < len(pop):
        #     # k = np.random.randint(0, 2*N)
        #     p1 = np.random.randint(0, N - 1)
        #     p2 = np.random.randint(N, 2 * N)
        #     ind = np.random.randint(0, D - 1)
        #     my_string1 = "[" + ",".join(str(element) for element in pop[p1]) + "]"
        #     my_string2 = "[" + ",".join(str(element) for element in pop[p2]) + "]"
        #     prompt_content_crossover = "I have two lists S1:" + my_string1 + " and S2:" + my_string2 + \
        #                                "Convert string S1 to list of integer numbers S1, string S2 to list of integer numbers S2." \
        #                                "Save the first " + str(ind + 1) + " elements in S1 as list L1,The length of listing L1 should be" + str(ind + 1) + "." \
        #                                 "Remove the same element in S2 as in L1 and save the remaining elements in S2 as the list L2," \
        #                                 "The length of listing L2 should be" + str(D - ind - 1) + \
        #                                "Merge L1 and S2 into a new list Offspring." \
        #                                "Please return the final value of list L1,L2,Offspring directly." \
        #                                "Do not give additional explanations."

            #     prompt_content_crossover2 = "I have a string S1:"+str(pop)+",  identify string S1 and convert S1 to np.array array pop"\
            # "Please randomly select two different elements pop1 and pop2"\
            # "Save the first "+ str(k+1) + " elements in pop1 as list L1,The length of listing L1 should be"+ str(k+1)+"." \
            # "Remove the same element in pop2 as in L1 and save the remaining elements in pop2 as the list L2,The length of listing L2 should be"+ str(len(pop[0])-k-1)+\
            # "Merge L1 and pop2 into a new list Offspring."\
            # "Please return the final value of list L1,L2,Offspring directly." \
            # "Do not give additional explanations."

        #     Offspring_one = get_response(prompt_content_crossover)
        #     list_strings = re.findall(r'\[([^\]]*)\]', Offspring_one)
        #     # 将字符串转换为列表
        #     One_offspring_list = []
        #     for list_string in list_strings:
        #         One_offspring_list.append([int(x) for x in list_string.split(',')])
        #     offspring_array = np.array(One_offspring_list[2])
        #     set_offspring_array = set(offspring_array)
        #     if len(offspring_array) == len(pop) and len(set_offspring_array) == len(offspring_array):
        #         Offspring.append(offspring_array)
        # Offspring = np.array(Offspring)

        """
         Slight mutation
        """
        k = np.random.randint(0, high=D, size=2 * N) + 1
        s = np.random.randint(0, high=D, size=2 * N) + 1

        for i in range(2 * N):
            if s[i] < k[i]:
                Offspring[i, :] = np.hstack(
                    (
                        np.hstack(
                            (
                                np.hstack(
                                    (
                                        Offspring[i, : s[i] - 1],
                                        Offspring[i, k[i] - 1],
                                    )
                                ),  # noqa
                                Offspring[i, s[i] - 1 : k[i] - 1],  # noqa
                            )
                        ),
                        Offspring[i, k[i] :],  # noqa
                    )
                )  # noqa
            elif s[i] > k[i]:
                Offspring[i, :] = np.hstack(
                    (
                        np.hstack(
                            (
                                np.hstack(
                                    (
                                        Offspring[i, : k[i] - 1],  # noqa
                                        Offspring[i, k[i] : s[i] - 1],  # noqa
                                    )
                                ),  # noqa
                                Offspring[i, k[i] - 1],
                            )
                        ),
                        Offspring[i, s[i] - 1 :],  # noqa
                    )
                )

    elif problem.encoding == "two_permutation":
        """
         Genetic operators for two_permutation based encoding
         Order crossover
        """
        Offspring = np.vstack((pop1, pop2))
        pop1_1 = pop1[:, : int(D / 2)]  # 前段 # noqa
        pop1_2 = pop1[:, int(D / 2) :]  # 后段 # noqa
        pop2_1 = pop2[:, : int(D / 2)]  # 前段 # noqa
        pop2_2 = pop2[:, int(D / 2) :]  # 后段 # noqa
        k = np.random.randint(0, high=int(D / 2), size=2 * N)
        k1 = np.random.randint(0, high=int(D / 2), size=2 * N)
        for i in range(N):  # noqa
            Offspring[i, k[i] + 1 : int(D / 2)] = np.setdiff1d(  # noqa
                pop2_1[i, :], pop1_1[i, : k[i] + 1], True
            )  # noqa
            Offspring[i + N, k[i] + 1 : int(D / 2)] = np.setdiff1d(  # noqa
                pop1_1[i, :], pop2_1[i, : k[i] + 1], True
            )  # noqa
            Offspring[i, k1[i] + 1 + int(D / 2) :] = np.setdiff1d(  # noqa
                pop2_2[i, :], pop1_2[i, : k1[i] + 1], True
            )  # noqa
            Offspring[i + N, k1[i] + 1 + int(D / 2) :] = np.setdiff1d(  # noqa
                pop1_2[i, :], pop2_2[i, : k1[i] + 1], True
            )  # noqa

        """
        Slight mutation
        """
        k = np.random.randint(0, high=int(D / 2), size=2 * N) + 1
        k1 = np.random.randint(0, high=int(D / 2), size=2 * N) + 1
        s = np.random.randint(0, high=int(D / 2), size=2 * N) + 1
        s1 = np.random.randint(0, high=int(D / 2), size=2 * N) + 1
        for i in range(2 * N):
            # 前段
            if s[i] < k[i]:
                Offspring[i, : int(D / 2)] = np.hstack(
                    (
                        np.hstack(
                            (
                                np.hstack(
                                    (
                                        Offspring[i, : s[i] - 1],
                                        Offspring[i, k[i] - 1],
                                    )
                                ),
                                Offspring[i, s[i] - 1 : k[i] - 1],  # noqa
                            )
                        ),  # noqa
                        Offspring[i, k[i] : int(D / 2)],  # noqa
                    )
                )
            elif s[i] > k[i]:
                Offspring[i, : int(D / 2)] = np.hstack(
                    (
                        np.hstack(
                            (
                                np.hstack(
                                    (
                                        Offspring[i, : k[i] - 1],
                                        Offspring[i, k[i] : s[i] - 1],  # noqa
                                    )
                                ),
                                Offspring[i, k[i] - 1],
                            )
                        ),  # noqa
                        Offspring[i, s[i] - 1 : int(D / 2)],  # noqa
                    )
                )
            # 后段
            if s1[i] < k1[i]:
                Offspring[i, int(D / 2) :] = np.hstack(  # noqa
                    (
                        np.hstack(
                            (
                                np.hstack(
                                    (
                                        Offspring[
                                            i,  # noqa
                                            int(D / 2) : s1[i]  # noqa
                                            - 1
                                            + int(D / 2),
                                        ],
                                        Offspring[i, k1[i] - 1 + int(D / 2)],
                                    )
                                ),
                                Offspring[
                                    i,
                                    s1[i]
                                    - 1
                                    + int(D / 2) : k1[i]  # noqa
                                    - 1
                                    + int(D / 2),
                                ],
                            )
                        ),  # noqa
                        Offspring[i, k1[i] + int(D / 2) :],  # noqa
                    )
                )
            elif s1[i] > k1[i]:  # noqa
                Offspring[i, int(D / 2) :] = np.hstack(  # noqa
                    (
                        np.hstack(
                            (
                                np.hstack(
                                    (
                                        Offspring[
                                            i,
                                            int(D / 2) : k1[i]  # noqa
                                            - 1
                                            + int(D / 2),
                                        ],
                                        Offspring[
                                            i,
                                            k1[i]
                                            + int(D / 2) : s1[i]  # noqa
                                            - 1
                                            + int(D / 2),
                                        ],
                                    )
                                ),
                                Offspring[i, k1[i] - 1 + int(D / 2)],
                            )
                        ),  # noqa
                        Offspring[i, s1[i] - 1 + int(D / 2) :],  # noqa
                    )
                )

    elif problem.encoding == "vrp":
        CUSNUM = max(pop[0])
        Offspring = pop
        Offspring_hat = Offspring[:, 1:-1]
        # 2-opt
        for i in range(N):
            count = 1
            # vrp编码转为序列编码,便于进行交叉
            for j in range(len(Offspring_hat[i])):
                if Offspring_hat[i][j] == 0:
                    Offspring_hat[i][j] = CUSNUM + count
                    count = count + 1
            # 2-opt
            k11 = np.random.randint(0, D - 2)
            k12 = np.random.randint(0, D - 2)
            start = min(k11, k12)
            end = max(k11, k12)
            if start != end:
                Offspring_hat[i][start : end + 1] = np.flipud(  # noqa
                    Offspring_hat[i][start : end + 1]  # noqa
                )
            # swap()
            k21 = np.random.randint(0, D - 2)
            k22 = np.random.randint(0, D - 2)
            if k21 != k22:
                Offspring_hat[i][k21], Offspring_hat[i][k22] = (
                    Offspring_hat[i][k22],
                    Offspring_hat[i][k21],
                )
            # 序列编码解码回vrp编码
            for j in range(len(Offspring_hat[i])):
                if Offspring_hat[i][j] > CUSNUM:
                    Offspring_hat[i][j] = 0
            Offspring[i] = np.hstack((0, np.hstack((Offspring_hat[i], 0))))

    else:
        """
         Genetic operators for real encoding
         Simulated binary crossover
        """
        offspring = []
        while len(offspring) < pop.shape[0]:
            i = 0
            response = ollama.chat(model='llama3', messages=[
                {
                    'role': 'user',
                    'content': f"""
                                I have two existing 1 by {D} dimensional numpy array P={pop1[i]} and O={pop2[i]}.\
                                Please return two numpy array L and K with the same size of P that is totally different from O and P but can be motivated from them.\
                                Please use the format:
                                L=<L> 
                                K=<K>
                                Do not give additional explanations.If you return code, give the results of your code run, and output a specific list
                                """
                },
            ])
            r = response['message']['content']
            float_pattern = r'\b\d+\.\d+\b'
            text = re.findall(float_pattern, r)[-12:]
            float_values = [float(match) for match in text]
            if len(float_values) == pop1.shape[1]*2:
                i += 1
                off1 = float_values[:6]
                off2 = float_values[6:12]
                offspring.append(off1)
                offspring.append(off2)
        Offspring = np.array(offspring)



        # beta = np.zeros((N, D))
        # mu = np.random.random((N, D))
        # beta[mu <= 0.5] = (2 * mu[mu <= 0.5]) ** (1 / (disC + 1))
        # beta[mu > 0.5] = (2 - 2 * mu[mu > 0.5]) ** (-1 / (disC + 1))
        # beta = beta * (-1) ** np.random.randint(0, 2, (N, D))
        # beta[np.random.random((N, D)) < 0.5] = 1
        # beta[np.tile(np.random.random((N, 1)) > proC, (1, D))] = 1
        # Offspring = np.vstack(
        #     (
        #         (pop1 + pop2) / 2 + beta * (pop1 - pop2) / 2,
        #         (pop1 + pop2) / 2 - beta * (pop1 - pop2) / 2,
        #     )
        # )  # noqa
        # """
        # Polynomial mutation
        # """
        # Lower = np.tile(problem.lb, (2 * N, 1))
        # Upper = np.tile(problem.ub, (2 * N, 1))
        # Site = np.random.random((2 * N, D)) < proM / D
        # mu = np.random.random((2 * N, D))
        # temp = np.logical_and(Site, mu <= 0.5)
        # Offspring = np.minimum(np.maximum(Offspring, Lower), Upper)
        # Offspring[temp] = Offspring[temp] + (Upper[temp] - Lower[temp]) * (
        #     (
        #         2 * mu[temp]
        #         + (1 - 2 * mu[temp])
        #         * (
        #             1
        #             - (Offspring[temp] - Lower[temp])
        #             / (Upper[temp] - Lower[temp])  # noqa
        #         )
        #         ** (disM + 1)
        #     )
        #     ** (1 / (disM + 1))
        #     - 1
        # )  # noqa
        # temp = np.logical_and(Site, mu > 0.5)  # noqa: E510
        # Offspring[temp] = Offspring[temp] + (Upper[temp] - Lower[temp]) * (
        #     1
        #     - (
        #         2 * (1 - mu[temp])
        #         + 2
        #         * (mu[temp] - 0.5)
        #         * (
        #             1
        #             - (Upper[temp] - Offspring[temp])
        #             / (Upper[temp] - Lower[temp])
        #         )  # noqa
        #         ** (disM + 1)
        #     )
        #     ** (1 / (disM + 1))
        # )  # noqa

    if calobj:  # noqa: E510
        Offspring = Population(decs=Offspring)
    return Offspring
