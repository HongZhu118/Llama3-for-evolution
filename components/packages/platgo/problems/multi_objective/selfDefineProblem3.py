import numpy as np
import pandas as pd
from ...Problem import Problem
import math
from ... import Population
from scipy.spatial.distance import cdist


class selfDefineProblem3(Problem):
    type = {
        "n_obj": {"multi"},
        "encoding": {"real"},
        "special": "none"
    }

    def __init__(self, in_optimization_problem={}) -> None:
        optimization_problem = {
            "name": "selfDefineProblem3",
            "encoding": "real",
            "n_var": 2,
            "lower": "0",
            "upper": "1",
            "n_obj": 2,
            "height": 4,
            "width": 5,
            "data_list": [],
            "initFcn": [],
            "decFcn": [],
            "objFcn": [],  # noqa
            "conFcn": ["0"]
        }
        optimization_problem.update(in_optimization_problem)
        self.height = optimization_problem["height"]
        self.width = optimization_problem["width"]
        self.data_list = optimization_problem["data_list"]
        self.overlap20 = []
        super(selfDefineProblem3, self).__init__(optimization_problem)

    def Point_Heigh(self, Point: list, Param: list):
        Max_Deep = 197.2 / 1852
        # 求点到线的距离
        Distance = abs(Param[0] * Point[0] + Param[1] * Point[1] + Param[2]) / pow(Param[0] ** 2 + Param[1] ** 2, 0.5)
        return Max_Deep - Distance / pow(3, 0.5)


    def compute(self, pop) -> None:
        objv = []

        for dec in pop.decs:
            flag = False
            dec[2] = 0
            # 目标值：
            Object_Length = 0
            Object_outArea_Cover = 0
            # Object_Over20 = float('inf')
            # 参数定义：角度，第一条测线与边界的距离，测线之间的距离，测量区域的长（起点边），测量区域的长（测线方向平行边），海底地图数据表
            v = dec[0]
            Theta = dec[1]
            pop_d = dec[2:]
            Length = self.width
            Width = self.height
            Data = self.data[0]
            Max_Deep = 197.2 / 1852
            W_Max = pow(3, 0.5) * Max_Deep * 2

            Cover_point = []

            # 测线斜率
            l_k = -np.tan(v * np.pi / 180)
            # 边界方向
            if v <= 90:
                S_start = [0, 0]
                S_end = [Length, Width]
                # 起始方程A, B, C
                Line_start_param = [l_k, -1, 0]
                # 最大边界距离
                Bound_max = abs(
                    Line_start_param[0] * S_end[0] + Line_start_param[1] * S_end[1] + Line_start_param[2]) / pow(
                    Line_start_param[0] ** 2 + Line_start_param[1] ** 2, 0.5)
            else:
                S_start = [0, Width]
                S_end = [Length, 0]
                # 起始方程A, B, C
                Line_start_param = [l_k, -1, Width]
                Bound_max = abs(
                    Line_start_param[0] * S_end[0] + Line_start_param[1] * S_end[1] + Line_start_param[2]) / pow(
                    Line_start_param[0] ** 2 + Line_start_param[1] ** 2, 0.5)

            # 计算测线方程以及测线总长度
            Line_list_ABC = []
            Bound_line = Theta

            for i_d in pop_d:
                Bound_line += i_d
                if Bound_line > Bound_max:
                    break

                if v <= 90:
                    l_b = Bound_line / np.cos(v * np.pi / 180)
                else:
                    l_b = Width - Bound_line / np.cos((180 - v) * np.pi / 180)

                # 直线方程：Ax + By + C = 0
                l_A = l_k
                l_B = -1
                l_C = l_b
                Line_list_ABC.append([l_A, l_B, l_C])

                # 计算航线长度
                Point = []
                res_0_y = l_k * 0 + l_b
                if 0 <= res_0_y <= 4:
                    Point.append([0, res_0_y])

                res_5_y = l_k * 5 + l_b
                if 0 <= res_5_y <= 4:
                    Point.append([5, res_0_y])

                res_x_0 = (0 - l_b) / l_k
                if 0 <= res_x_0 <= 5:
                    Point.append([res_x_0, 0])

                res_x_4 = (4 - l_b) / l_k
                if 0 <= res_x_4 <= 5:
                    Point.append([res_x_4, 4])

                Point = np.unique(Point, axis=0)
                if len(Point) != 2:
                    print("该测线不符合实际")
                    flag = True
                    break
                else:
                    Object_Length += pow((Point[0][0] - Point[1][0]) ** 2 + (Point[0][1] - Point[1][1]) ** 2, 0.5)

            if flag:
                objv.append([99999999, 1])
                continue

            for i_line in Line_list_ABC:
                for i_x in range(0, 251):
                    for i_y in range(0, 201):
                        # 计算点到直线的距离
                        point_line_Dis = abs(i_line[0] * i_x * 0.02 + i_line[1] * i_y * 0.02 + i_line[2]) / pow(
                            i_line[0] ** 2 + i_line[1] ** 2, 0.5)
                        if point_line_Dis > W_Max / 2:
                            continue
                        else:
                            point_heigh = self.Point_Heigh(list([i_x * 0.02, i_y * 0.02]), i_line)
                            if Data[i_y][i_x] / 1852 <= point_heigh:
                                # 可以扫描到，加入到扫描区域中
                                Cover_point.append([i_x, i_y])
                            else:
                                # 不能扫描到
                                continue

            All = np.unique(Cover_point, axis=0)

            Object_outArea_Cover = 1 - len(All) / (200 * 250)

            objv.append([Object_Length, Object_outArea_Cover])
        pop.objv = np.array(objv)
        pop.cv = np.zeros((pop.pop_size, self.n_constr))

