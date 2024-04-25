import random
import datetime

import numpy as np
import pandas as pd

from ....Problem import Problem
from .....platgo import Population


class SatellkiteRCPSP(Problem):
    """
    type = {"n_obj": "single", "encoding": "real"}
    """

    def __init__(self, in_optimization_problem, debug=True) -> None:
        optimization_problem = {
            "mode": 0,
            "encoding": "real",
            "n_obj": 1,
        }
        optimization_problem.update(in_optimization_problem)
        super(SatellkiteRCPSP, self).__init__(
            optimization_problem, debug=debug
        )

    # def init_pop(self, N: int = None):
    #     if N is None:
    #         N = self.pop_size
    #     # 将lb矩阵向0维坐标方向重复n次
    #     lb = np.tile(self.lb, (N, 1))
    #     # 将ub矩阵向0维坐标方向重复n次
    #     ub = np.tile(self.ub, (N, 1))
    #     decs = np.random.randint(lb, ub)  # todo 这里种群初始化的时候要考虑编码方式
    #     return Population(decs=decs)

    def fix_decs(self, pop: Population):
        for i in range(len(pop)):
            for j in range(pop.decs.shape[1]):
                if self.lb[j] <= pop.decs[i][j] <= self.ub[j]-1:
                    continue
                elif self.ub[j]-1 < pop.decs[i][j] < self.ub[j]:
                    pop.decs[i][j] = int(pop.decs[i][j])
                else:
                    pop.decs[i][j] = np.random.uniform(self.lb[j], self.ub[j])
        return pop

    def compute(self, pop) -> None:
        objv = np.zeros((pop.decs.shape[0], 1))
        finalresult = np.empty((pop.decs.shape[0], 1), dtype=np.object)
        print("done")
        for i, x in enumerate(pop.decs):
            objv[i], finalresult[i] = self.main(x)
        pop.objv = objv
        pop.finalresult = finalresult
        pop.cv = np.zeros((pop.pop_size, self.n_constr))

    def main(self, x):
        data = self.data[0]
        task_time_dict = dict()  # 存储每个任务的开始时间和结束时间
        task_done_dict = dict()  # 存储每个任务是否做过
        device_time_dict = dict()  # 统计每个设备资源的占用时间段
        worker_time_dict = dict()  # 统计每个人员资源的占用时间段

        while len(task_done_dict) < (len(data["scheduling_task"])):
            for i in range(len(x)):
                task_id = data["scheduling_task"][i]["task_id"]
                flag_pre = False
                if task_id in task_done_dict:  # 当前任务做过
                    continue
                else:  # 当前任务没有做过
                    start_time_list = list()  # 统计该任务的可能的开始时间
                    start_time = int(data["scheduling_task"][i]["start_time"])
                    start_time_list.append(start_time)
                    using_device = list(data["scheduling_task"][i]["using_device"].keys())[0]  # 当前任务占用的设备
                    if using_device in device_time_dict:
                        start_time = device_time_dict[using_device][-1][1]
                        start_time_list.append(start_time)
                    using_worker = list(data["scheduling_task"][i]["using_worker"].keys())[0]  # 当前任务占用的人员
                    if using_worker in worker_time_dict:
                        start_time = worker_time_dict[using_worker][-1][1]
                        start_time_list.append(start_time)
                    start_time = max(start_time_list)
                    end_time = start_time + x[i]
                    if "pre_task" not in data["scheduling_task"][i]:  # 该任务没有前置任务的约束
                        task_time_dict[task_id] = [start_time, end_time]
                        device_time_dict.setdefault(using_device, []).append([start_time, end_time])
                        worker_time_dict.setdefault(using_worker, []).append([start_time, end_time])
                        task_done_dict[task_id] = True
                    else:  # 该任务有前置任务的约束
                        pre_task_dict = data["scheduling_task"][i]["pre_task"]
                        for key, value in pre_task_dict.items():
                            if key not in task_done_dict:
                                flag_pre = True
                                break
                        if flag_pre:  # 该任务的前置任务没有做完，跳到下一个任务
                            continue
                        else:  # 该任务的前置任务做完
                            start_time_range = list()
                            point_type_dict = data["scheduling_task"][i]["point_type"]
                            for key, value in pre_task_dict.items():
                                if point_type_dict[key] == 0:  # 头->头
                                    start_time_former = task_time_dict[key][0] + int(pre_task_dict[key][0])
                                    start_time_latter = task_time_dict[key][0] + int(pre_task_dict[key][1])
                                    start_time_range.append([start_time_former, start_time_latter])
                                if point_type_dict[key] == 2:  # 尾->头
                                    start_time_former = task_time_dict[key][1] + int(pre_task_dict[key][0])
                                    start_time_latter = task_time_dict[key][1] + int(pre_task_dict[key][1])
                                    start_time_range.append([start_time_former, start_time_latter])
                            if len(start_time_range) > 0:
                                min_range = [start_time_range[0][0], start_time_range[0][1]]
                                for m in range(len(start_time_range)):
                                    for n in range(m + 1, len(start_time_range)):
                                        a1 = min_range[0]
                                        b1 = min_range[1]
                                        a2 = start_time_range[n][0]
                                        b2 = start_time_range[n][1]
                                        if a1 <= b2 and b1 >= a2:
                                            min_range[0] = a1 if a2 > a1 else a1
                                            min_range[1] = b2 if b1 > b2 else b1
                                        elif a2 <= b1 and b2 >= a1:
                                            min_range[0] = a2 if a1 > a2 else a1
                                            min_range[1] = b1 if b2 > b1 else b2
                                        else:
                                            return 100000, task_time_dict
                                start_time = max(start_time, min_range[0])
                                end_time = start_time + x[i]
                            for key, value in pre_task_dict.items():
                                if point_type_dict[key] in [1, 3]:  # 头->尾
                                    pre_task_range_former = int(pre_task_dict[key][0])
                                    if pre_task_dict[key][1] == "inf":
                                        pre_task_range_latter = np.inf
                                    else:
                                        pre_task_range_latter = int(pre_task_dict[key][1])
                                    if point_type_dict[key] == 1:
                                        if pre_task_range_latter >= end_time - task_time_dict[key][
                                            0] >= pre_task_range_former:  # 满足约束条件
                                            task_time_dict[task_id] = [start_time, end_time]
                                            task_done_dict[task_id] = True
                                        else:
                                            return 100000, task_time_dict
                                    if point_type_dict[key] == 3:
                                        if pre_task_range_latter >= end_time - task_time_dict[key][
                                            1] >= pre_task_range_former:  # 满足约束条件
                                            task_time_dict[task_id] = [start_time, end_time]
                                            task_done_dict[task_id] = True
                                        else:
                                            return 100000, task_time_dict
                            task_time_dict[task_id] = [start_time, end_time]
                            device_time_dict.setdefault(using_device, []).append([start_time, end_time])
                            worker_time_dict.setdefault(using_worker, []).append([start_time, end_time])
                            task_done_dict[task_id] = True
        print(x)
        task_time_dict = dict(sorted(task_time_dict.items(), key=lambda x: (x[1][0], x[1][1])))
        print(task_time_dict)
        return np.max(sum(list(task_time_dict.values()), [])), f"{task_time_dict}"
