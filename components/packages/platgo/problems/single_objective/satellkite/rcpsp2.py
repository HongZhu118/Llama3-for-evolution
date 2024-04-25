import random
import datetime
import copy
import numpy as np
import pandas as pd

from ....Problem import Problem
from .....platgo import Population


class SatellkiteRCPSP2(Problem):
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
        super(SatellkiteRCPSP2, self).__init__(
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
                if self.lb[j] <= pop.decs[i][j] <= self.ub[j] - 1:
                    continue
                elif self.ub[j] - 1 < pop.decs[i][j] < self.ub[j]:
                    pop.decs[i][j] = int(pop.decs[i][j])
                else:
                    pop.decs[i][j] = np.random.uniform(self.lb[j], self.ub[j])
        return pop

    def compute(self, pop) -> None:
        objv = np.zeros((pop.decs.shape[0], 1))
        finalresult = np.empty((pop.decs.shape[0], 1), dtype=np.object_)
        print("done")
        for i, x in enumerate(pop.decs):
            objv[i], finalresult[i] = self.main(x)
        pop.objv = objv
        pop.finalresult = finalresult
        pop.cv = np.zeros((pop.pop_size, self.n_constr))

    def main(self, x):
        data = self.data[0]
        resources = self.data[1]
        resources_info = dict()
        '''
          "resource_id": "11023",
          "resource_type": "consumed",
          "resource_num": 60,
          "load": [0, 1, 2, 3]
        '''
        for r in resources["resources"]:
            resources_info[r["resource_id"]] = [r["resource_type"], r["resource_num"], r["load"]]
        num = 0  # 用于统计完成的任务数
        # x = [0.14309352, 7.51917858, 0.72907644, 10.57966519, 4.18900198, 5.83425065, 18.87411393, 7.65927247]
        # x = [0.68878772, 6.96734306, 3.74064235, 16, 9.92159917, 4.36575706, 17.62169372, 9.18693719]

        task_time_dict = dict()  # 存储每个任务的开始时间和结束时间
        task_done_dict = dict()  # 存储满足约束类型任务的任务
        task2resources = dict()  # 存储每个任务所用资源及数量
        resources2task = dict()  # 存储每种资源对应的任务
        for i in range(0, len(x), 2):
            task_id = data["scheduling_task"][int(i / 2)]["task_id"]
            task_time_dict[task_id] = [x[i], x[i] + x[i + 1]]
            using_resources = data["scheduling_task"][int(i / 2)]["using_resources"]
            for key, value in using_resources.items():
                task2resources.setdefault(task_id, []).append([key, value])
                resources2task.setdefault(key, {})[task_id] = value
        for j in range(len(data["scheduling_task"])):
            task_id = data["scheduling_task"][j]["task_id"]
            start_time = x[2 * j]  # 当前任务的开始时间
            end_time = start_time + x[2 * j + 1]  # 当前任务的结束时间
            if "constrained_task" in data["scheduling_task"][j]:  # 存在约束类型的任务
                constrained_num = list()
                for key, value in data["scheduling_task"][j]["constrained_task"].items():  # 遍历其约束类型的任务
                    if value[1] == "inf":
                        value[1] = np.inf
                    else:
                        value[1] = int(value[1])
                    if int(value[2]) == 0:  # 开始时间->开始时间
                        # 用当前任务的开始时间减去约束任务的开始时间 判断是否满足约束
                        if task_time_dict[key][0] == -1:
                            constrained_num.append(1)
                        else:
                            if int(value[0]) <= start_time - task_time_dict[key][0] <= value[1]:
                                constrained_num.append(1)
                            elif start_time - task_time_dict[key][0] < int(value[0]):
                                task_time_dict[task_id] = [-1,-1]
                            elif start_time - task_time_dict[key][0]>value[1]:
                                task_time_dict[task_id] = [-1,-1]
                    elif int(value[2]) == 1:  # 开始时间->结束时间
                        # 用当前任务的结束时间减去约束任务的开始时间 判断是否满足约束
                        if task_time_dict[key][0] == -1:
                            constrained_num.append(1)
                        else:
                            if int(value[0]) <= end_time - task_time_dict[key][0] <= value[1]:
                                constrained_num.append(1)
                            elif end_time - task_time_dict[key][0] < int(value[0]):
                                task_time_dict[task_id] = [-1,-1]
                            elif end_time - task_time_dict[key][0]>value[1]:
                                task_time_dict[task_id] = [-1,-1]
                    elif int(value[2]) == 2:  # 结束时间->开始时间
                        # 用当前任务的开始时间减去约束任务的结束时间 判断是否满足约束
                        if task_time_dict[key][1] == -1:
                            constrained_num.append(1)
                        else:
                            if int(value[0]) <= start_time - task_time_dict[key][1] <= value[1]:
                                constrained_num.append(1)
                            elif start_time - task_time_dict[key][1] < int(value[0]):
                                task_time_dict[task_id] = [-1, -1]
                            elif start_time - task_time_dict[key][1] > value[1]:
                                task_time_dict[task_id] = [-1, -1]
                    else:  # 结束时间->结束时间
                        # 用当前任务的结束时间减去约束任务的结束时间 判断是否满足约束
                        if task_time_dict[key][1] == -1:
                            constrained_num.append(1)
                        else:
                            if int(value[0]) <= end_time - task_time_dict[key][1] <= value[1]:
                                constrained_num.append(1)
                            elif end_time - task_time_dict[key][1] < int(value[0]):
                                task_time_dict[task_id] = [-1, -1]
                            elif end_time - task_time_dict[key][1] > value[1]:
                                task_time_dict[task_id] = [-1, -1]
                if sum(constrained_num) == len(data["scheduling_task"][j]["constrained_task"]):  # 该任务的所有约束均已满足
                    num += 1
                    task_done_dict[task_id] = [start_time, end_time]
            else:  # 该任务没有约束类型的任务
                num += 1
                task_done_dict[task_id] = [start_time, end_time]
        # 进行资源约束的检测
        task_done_dict = dict(sorted(task_done_dict.items(), key=lambda x: (x[1][0], x[1][1])))  # 根据开始时间进行排序
        task_done_list = list(task_done_dict.keys())
        task_done_dict2 = dict()  # 存储满足资源约束的最终的任务
        for key in task_done_list:  # 对当前可行任务进行遍历
            resources_info2 = copy.deepcopy(resources_info)
            if key in task_done_dict2:  #
                continue
            using_resources = task2resources[key]  # 获取该任务的使用资源
            del_task = list()  # 记录需要删除的任务
            for us in range(len(using_resources)):  # 对该任务的资源集合进行遍历，找出每种资源用在哪些设备上
                r_id = using_resources[us][0]  # 资源id
                t_id = list(resources2task[r_id].keys())  # 找到使用该资源的所有任务
                t_id_other = [x for x in t_id if x in task_done_list]  # 使用该资源的所有任务与可行任务的交集
                t_id_other.remove(key)  # 删除当前任务，不与自身相比
                resource_num = resources_info2[r_id][1]  # 所用资源的总量
                resource_num -= using_resources[us][1]  # 减去当前任务所用的资源数
                for _ in t_id_other:
                    if _ in task_done_dict2:
                        continue
                    t_start = task_done_dict[_][0]  # 其余任务的开始时间
                    t_end = task_done_dict[_][1]  # 其余任务的结束时间
                    # 其他任务的所用资源数量
                    using_resource = resources2task[r_id][_]
                    if resources_info2[r_id][0] == "consumed":
                        resource_num -= using_resource  # 减去其余任务所用的资源数
                        if resource_num >= 0:  # 资源满足
                            resources_info2[r_id][1] = resource_num  # 消耗型资源库的进行更新
                    else:  # 可重用型资源判断时间是否冲突
                        if task_done_dict[key][0] < t_end and task_done_dict[key][1] > t_start:  # 时间冲突
                            resource_num -= using_resource  # 减去其余任务所用的资源数
                        # else:  # 没有时间冲突进行资源还原
                        #     resource_num += using_resource  # 加上其余任务所用的资源数
                    if resource_num < 0:  # 资源不满足
                        del_task.append(_)  # 当前任务不能做
                        resource_num += using_resource  # 加上其余任务所用的资源数
            if len(del_task) != 0:
                for d in set(del_task):
                    task_done_list.remove(d)
            task_done_dict2[key] = task_done_dict[key]
        return len(data["scheduling_task"]) - len(task_done_dict2), f"{task_done_dict2}"
