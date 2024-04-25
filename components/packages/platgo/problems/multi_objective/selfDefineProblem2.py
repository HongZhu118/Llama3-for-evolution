import numpy as np
import pandas as pd
from ...Problem import Problem
from ... import Population
from scipy.spatial.distance import cdist


class selfDefineProblem2(Problem):
    type = {
        "n_obj": {"multi"},
        "encoding": {"real"},
        "special": "none"
    }

    def __init__(self, in_optimization_problem={}) -> None:
        optimization_problem = {
            "name": "selfDefineProblem1",
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
        super(selfDefineProblem2, self).__init__(optimization_problem)

    def compute(self, pop) -> None:
        objv = []

        for dec in pop.decs:
            Theta = dec[0]
            d = dec[1]
            Length = self.width
            Width = self.height
            Data = self.data[0]
            Max_Deep = 197.2 / 1852
            # 计算起点坐标点
            Coord_point_list_x = []
            Temp_start = Theta
            Start_Num = 0
            while Temp_start < Length:
                # 加入起点坐标
                Coord_point_list_x.append([Temp_start, 0])
                Start_Num += 1
                Temp_start += d
            # 计算总测线长度目标值：
            Object_Length = Start_Num * Width
            # 左侧光束方向向量
            D_vector_l = [-pow(3, 0.5), 0, -1]
            # 右侧光束方向向量
            D_vector_r = [pow(3, 0.5), 0, -1]

            Cover_point = list([] for _ in range(0, Start_Num + 1))
            res = list([0, 0.0] for _ in range(0, Start_Num + 1))

            for i_step in range(0, 402, 2):
                # 计算每一步的测量区域
                # 该时刻船的位置
                for ind in range(len(Coord_point_list_x)):
                    Temp_cover_point = []

                    # 光束函数(x0+at, y0+bt, z0+ct) --->(x0, y0, z0):船此刻的坐标
                    x0 = Coord_point_list_x[ind][0]
                    y0 = i_step / 100
                    z0 = Max_Deep
                    Light_L = [[x0, D_vector_l[0]], [y0, D_vector_l[1]], [z0, D_vector_l[0]]]
                    Light_R = [[x0, D_vector_r[0]], [y0, D_vector_r[1]], [z0, D_vector_r[0]]]

                    Width_w = Max_Deep * pow(3, 0.5)
                    Bound_l = x0 - Width_w
                    temp_bound_l = x0 - 0.02
                    Bound_r = x0 + Width_w
                    temp_bound_r = x0 + 0.02

                    Temp_cover_point.append([round(x0 * 100 / 2), i_step / 2])

                    while Bound_l <= temp_bound_l:
                        if temp_bound_l < 0:
                            break
                        else:
                            # 边界x，y坐标
                            B_x = temp_bound_l
                            B_y = i_step / 2
                            # --->海底区域真实值
                            Deep_true = Data[B_y][round(B_x * 100 / 2)]
                            # --->海底区域此刻光束边界值
                            temp_t_l = (B_x - Light_L[0][0]) / Light_L[0][1]
                            Deep_cap = Light_L[2][0] + Light_L[2][1] * temp_t_l

                            if Deep_true > Deep_cap:
                                # 照不到
                                break
                            else:
                                Temp_cover_point.append([round(B_x * 100 / 2), B_y])
                                temp_bound_l -= 0.02

                    while temp_bound_r <= Bound_r:
                        if temp_bound_r > 5:
                            break
                        else:
                            # 边界x，y坐标
                            B_x = temp_bound_r
                            B_y = i_step / 2
                            # --->海底区域真实值
                            Deep_true = Data[B_y][round(B_x * 100 / 2)]
                            # --->海底区域此刻光束边界值
                            temp_t_r = (B_x - Light_R[0][0]) / Light_R[0][1]
                            Deep_cap = Light_R[2][0] + Light_R[2][1] * temp_t_r

                            if Deep_true > Deep_cap:
                                # 照不到
                                break
                            else:
                                Temp_cover_point.append([round(B_x * 100 / 2), B_y])
                                temp_bound_l += 0.02

                    Cover_point[ind] += Temp_cover_point
                    if ind != 0:
                        # res = [_ for _ in Cover_point[ind] if _ in Cover_point[ind - 1]]
                        # res_num = len(res)
                        cover_ratio = 1 - d / (temp_bound_r - temp_bound_l)
                        if cover_ratio > 0.2:
                            if res[ind][0] == 0:
                                res[ind][0] = 1
                            else:
                                res[ind][1] += 0.02
                        else:
                            res[ind][0] = 0

            Object_Over20 = 0
            for _ in range(len(res)):
                Object_Over20 += res[_][1]
            All = []
            for ind in range(0, Start_Num + 1):
                All += Cover_point[ind]

            All = np.unique(All, axis=0)

            Object_Area_Cover = 1-len(All) / (201 * 251)
            objv.append([Object_Length,Object_Area_Cover])
            self.overlap20.append(Object_Over20)
        pop.objv = np.array(objv)
        pop.cv = np.zeros((pop.pop_size, self.n_constr))

    # # 参数: 侧线之间的距离
    # def Get_Object(Param_Theta: float, Param_d: float, Param_x: float, Param_y: float, Param_data: pd.DataFrame()):
    #     # 目标值：
    #     Object_Length = float('inf')
    #     Object_Area_Cover = float('inf')
    #     Object_Next_cover = float('inf')
    #     # 参数定义：第一条测线与边界的距离，测线之间的距离，测量区域的长（起点边），测量区域的长（测线方向平行边），海底地图数据表
    #     Theta = Param_Theta
    #     d = Param_d
    #     Length = Param_x
    #     Width = Param_y
    #     Data = Param_data
    #     Max_Deep = 197.2 / 1852
    #     # 计算起点坐标点
    #     Coord_point_list_x = []
    #     Temp_start = Theta
    #     Start_Num = 0
    #     while (Temp_start < Length):
    #         # 加入起点坐标
    #         Coord_point_list_x.append([Temp_start, 0])
    #         Start_Num += 1
    #         Temp_start += d
    #     # 计算总测线长度目标值：
    #     Object_Length = Start_Num * Width
    #
    #     # 左侧光束方向向量
    #     D_vector_l = [-pow(3, 0.5), 0, -1]
    #     # 右侧光束方向向量
    #     D_vector_r = [pow(3, 0.5), 0, -1]
    #
    #     Cover_point = list([] for _ in range(0, Start_Num + 1))
    #     for i_step in range(0, 402, 2):
    #         # 计算每一步的测量区域
    #         # 该时刻船的位置
    #         for ind in range(len(Coord_point_list_x)):
    #             Temp_cover_point = []
    #
    #             # 光束函数(x0+at, y0+bt, z0+ct) --->(x0, y0, z0):船此刻的坐标
    #             x0 = Coord_point_list_x[ind][0]
    #             y0 = i_step / 100
    #             z0 = Max_Deep
    #             Light_L = [[x0, D_vector_l[0]], [y0, D_vector_l[1]], [z0, D_vector_l[0]]]
    #             Light_R = [[x0, D_vector_r[0]], [y0, D_vector_r[1]], [z0, D_vector_r[0]]]
    #
    #             Width_w = Max_Deep * pow(3, 0.5)
    #             Bound_l = x0 - Width_w
    #             temp_bound_l = x0 - 0.02
    #             Bound_r = x0 + Width_w
    #             temp_bound_r = x0 + 0.02
    #
    #             Temp_cover_point.append([round(x0 * 100 / 2), i_step / 2])
    #
    #             while Bound_l <= temp_bound_l:
    #                 if temp_bound_l < 0:
    #                     break
    #                 else:
    #                     # 边界x，y坐标
    #                     B_x = temp_bound_l
    #                     B_y = i_step / 2
    #                     # --->海底区域真实值
    #                     Deep_true = Data[B_y][round(B_x * 100 / 2)]
    #                     # --->海底区域此刻光束边界值
    #                     temp_t_l = (B_x - Light_L[0][0]) / Light_L[0][1]
    #                     Deep_cap = Light_L[2][0] + Light_L[2][1] * temp_t_l
    #
    #                     if Deep_true > Deep_cap:
    #                         # 照不到
    #                         break
    #                     else:
    #                         Temp_cover_point.append([round(B_x * 100 / 2), B_y])
    #                         temp_bound_l -= 0.02
    #
    #             while temp_bound_r <= Bound_r:
    #                 # 边界x，y坐标
    #                 B_x = temp_bound_r
    #                 B_y = i_step / 2
    #                 # --->海底区域真实值
    #                 Deep_true = Data[B_y][round(B_x * 100 / 2)]
    #                 # --->海底区域此刻光束边界值
    #                 temp_t_r = (B_x - Light_R[0][0]) / Light_R[0][1]
    #                 Deep_cap = Light_R[2][0] + Light_R[2][1] * temp_t_r
    #
    #                 if Deep_true > Deep_cap:
    #                     # 照不到
    #                     break
    #                 else:
    #                     Temp_cover_point.append([round(B_x * 100 / 2), B_y])
    #                     temp_bound_l += 0.02
    #
    #             Cover_point[ind] += Temp_cover_point
    #             if ind != 0:
    #                 res = [_ for _ in Cover_point[ind] if _ in Cover_point[ind - 1]]
    #                 res_num = len(res)
    #                 cover_ratio = 1 - d / (temp_bound_r - temp_bound_l)
    #
    #     All = np.array([])
    #     for ind in range(0, Start_Num + 1):
    #         All += Cover_point[ind]
    #
    #     All = np.unique(All, axis=0)
    #
    #     Object_Area_Cover = len(All) / (200 * 250)
    #
    #     return Object_Length, Object_Area_Cover
