import random
import datetime
import copy
import numpy as np
import pandas as pd

from components.packages.platgo.Problem import Problem
from components.packages.platgo import Population


class Daily_Planning(Problem):
    """
    type = {"n_obj": "single", "encoding": "real"}
    """

    def __init__(self, in_optimization_problem, debug=True) -> None:
        optimization_problem = {
            "mode": 0,
            "encoding": "permutation",
            "n_obj": 3,
        }

        optimization_problem.update(in_optimization_problem)
        self.task_start_time = optimization_problem["task_start_time"]
        self.index_task = optimization_problem["index_task"]
        super(Daily_Planning, self).__init__(
            optimization_problem, debug=debug
        )

    def init_pop(self, N: int = None):
        if N is None:
            N = self.pop_size
        data = self.data[0]
        decs = list()
        for i in range(N):
            decs.append(random.sample(range(1, self.n_var+1), self.n_var))
        decs = np.array(decs)
        return Population(decs=decs)

    # def fix_decs(self, pop: Population):
    #     # 对边界进行修复
    #     for i in range(len(pop)):
    #         for j in range(pop.decs.shape[1]):
    #             if self.lb[j] <= pop.decs[i][j] <= self.ub[j]:
    #                 continue
    #             else:
    #                 pop.decs[i][j] = np.random.uniform(self.lb[j], self.ub[j])
    #     return pop

    def compute(self, pop) -> None:
        objv = np.zeros((pop.decs.shape[0], self.n_obj))
        finalresult = np.empty((pop.decs.shape[0], 1), dtype=np.object_)
        print("done")
        for i, x in enumerate(pop.decs):
            objv[i], finalresult[i] = self.main(x)
        pop.objv = objv
        pop.finalresult = finalresult
        pop.cv = np.zeros((pop.pop_size, self.n_constr))

    def main(self, x):
        data = self.data[0]
        task_info = dict()  # 根据数据集统计每个任务的信息
        task_main_action = dict()  # 统计每个任务执行了多少个主动作
        time_main_action = dict()
        task_index = dict()  # 任务id对序列的映射
        index_task = dict()  # 序列对任务id的映射
        task_index2_dict = dict()
        next_times = 0  # 统计切换次数
        for i in range(0, len(data["scheduling_task"])):
            task_id = data["scheduling_task"][i]["task_id"]
            task_index2_dict[task_id] = i+1
            task_index[task_id] = [k for k in range(i*30+1, (i+1)*30+1)]
            for _ in [k for k in range(i*30+1, (i+1)*30+1)]:
                index_task[_] = task_id
            task_info[task_id] = {
                                  "time_window": data["scheduling_task"][i]["time_window"],
                                  "main_action_num": data["scheduling_task"][i]["main_action_num"],
                                  "main_action_time": data["scheduling_task"][i]["main_action_time"],
                                  "end_action": data["scheduling_task"][i]["end_action"],
                                  "next_action": data["scheduling_task"][i]["next_action"]
                                  }
            task_main_action[task_id] = 0
        start_time = task_info[index_task[int(x[0])]]["time_window"][0][0]  # 获取基准开始时间
        pre_task = float()  # 前一个动作
        for i in range(0, len(x)):
            time_window = task_info[index_task[int(x[i])]]["time_window"]  # 该任务的可执行窗口集合
            if i == 0:  # 判断第一个动作一般可行，主要进行基准时间的更新
                for t in range(0, len(time_window)):
                    start_time = task_info[index_task[int(x[i])]]["time_window"][t][0]  # 该动作对应任务的第一个执行时间窗口的开始时间
                    if start_time + task_info[index_task[int(x[i])]]["main_action_time"] < time_window[t][
                        0]:  # 仍然没有到达该任务的开始时间
                        task_main_action[index_task[int(x[i])]] += 1  # 该任务的主动作加1
                        time_main_action[x[i]] = [time_window[t][0]]  # 该主动作的开始时间
                        start_time = time_window[t][0] + task_info[index_task[int(x[i])]][
                            "main_action_time"]  # 更新基准开始时间
                        time_main_action[x[i]] += [start_time]  # 该主动作的结束时间
                        start_time += task_info[index_task[int(x[i])]]["end_action"]  # 加上后处理动作的时间
                        pre_task = x[i]
                        break
                    if time_window[t][0] <= start_time + task_info[index_task[int(x[i])]]["main_action_time"] <= time_window[t][
                        1]:
                        time_main_action[x[i]] = [start_time]  # 该主动作的开始时间
                        task_main_action[index_task[int(x[i])]] += 1  # 该任务的主动作加1
                        start_time += task_info[index_task[int(x[i])]]["main_action_time"]  # 基准开始时间+主动作时间
                        time_main_action[x[i]] += [start_time]  # 该主动作的结束时间
                        start_time += task_info[index_task[int(x[i])]]["end_action"]  # 加上后处理动作的时间
                        pre_task = x[i]
                        break  # 找到一个可执行的窗口直接跳出
            else:  # 判断第二个及以后的动作
                if index_task[int(pre_task)] == index_task[int(x[i])]:  # 没有发生任务的切换
                    for t in range(0, len(time_window)):  # 进行第二个动作，仍然对该动作对应的任务时间窗口列表进行遍历
                        if start_time + task_info[index_task[int(x[i])]]["main_action_time"] < time_window[t][0]:  # 仍然没有到达该任务的开始时间
                            task_main_action[index_task[int(x[i])]] += 1  # 该任务的主动作加1
                            time_main_action[x[i]] = [time_window[t][0]]  # 该主动作的开始时间
                            start_time = time_window[t][0] + task_info[index_task[int(x[i])]][
                                "main_action_time"]  # 更新基准开始时间
                            time_main_action[x[i]] += [start_time]  # 该主动作的结束时间
                            start_time += task_info[index_task[int(x[i])]]["end_action"]  # 加上后处理动作的时间
                            pre_task = x[i]  # 记录下切换前的任务
                            break
                        if time_window[t][0] <= start_time + task_info[index_task[int(x[i])]]["main_action_time"] <= time_window[t][
                            1]:  # 当前任务在可执行的窗口范围为内
                            time_main_action[x[i]] = [start_time]  # 该主动作的开始时间
                            task_main_action[index_task[int(x[i])]] += 1  # 该任务的主动作加1
                            start_time += task_info[index_task[int(x[i])]]["main_action_time"]  # 基准开始时间+主动作时间
                            time_main_action[x[i]] += [start_time]  # 该主动作的结束时间
                            start_time += task_info[index_task[int(x[i])]]["end_action"]  # 加上后处理动作的时间
                            pre_task = x[i]  # 记录下切换前的任务
                            break  # 找到一个可执行的窗口直接跳出
                else:  # 发生了任务的切换
                    Flag_window = False
                    start_time -= task_info[index_task[int(pre_task)]]["end_action"]  # 需要先减去上一个任务的后处理时间,让上一个任务退回到主动作完成
                    # 上一个任务的切换时长和后处理时长取其中较大的
                    # 1.当切换时长大于后处理时,执行后处理的同时执行切换,后处理执行完毕,仍然执行切换
                    # 2.当后处理时长大于切换时长时,执行后处理的同时执行切换,切换后,后处理未完成仍需要等待后处理完成
                    temp_time = max(task_info[index_task[int(pre_task)]]["end_action"], task_info[index_task[int(pre_task)]]["next_action"][int(task_index2_dict[index_task[int(x[i])]])-1])
                    for t in range(len(time_window)):
                        if start_time + temp_time < time_window[t][0]:  # 任务的切换后，仍然没有到达该任务的开始时间
                            if time_window[t][0] + task_info[index_task[int(x[i])]]["main_action_time"] > time_window[t][1]:
                                continue
                            task_main_action[index_task[int(x[i])]] += 1  # 该任务的主动作加1
                            time_main_action[x[i]] = [time_window[t][0]]  # 该主动作的开始时间
                            start_time = time_window[t][0] + task_info[index_task[int(x[i])]][
                                "main_action_time"]  # 更新基准开始时间
                            time_main_action[x[i]] += [start_time]  # 该主动作的结束时间
                            start_time += task_info[index_task[int(x[i])]]["end_action"]  # 加上后处理动作的时间
                            next_times += 1  # 切换成功，切换次数加1
                            pre_task = x[i]  # 记录下切换前的任务
                            Flag_window = True   # 可用窗口已找到
                            break
                        elif time_window[t][0] <= start_time + temp_time <= time_window[t][1]:  # 切换后超过该任务的开始时间
                            if time_window[t][0] <= start_time + temp_time + task_info[index_task[int(x[i])]]["main_action_time"] <= time_window[t][
                                1]:  # 当前任务在可执行的窗口范围为内
                                task_main_action[index_task[int(x[i])]] += 1  # 该任务的主动作加1
                                start_time += temp_time  # 切换成功加上temp_time作为该任务的开始时间
                                time_main_action[x[i]] = [start_time]  # 该主动作的开始时间
                                start_time += task_info[index_task[int(x[i])]]["main_action_time"]  # 基准开始时间+主动作时间
                                time_main_action[x[i]] += [start_time]  # 该主动作的结束时间
                                start_time += task_info[index_task[int(x[i])]]["end_action"]  # 加上后处理动作的时间
                                next_times += 1  # 切换成功，切换次数加1
                                pre_task = x[i]
                                Flag_window = True
                                break
                    if not Flag_window:  # 找不到可用的时间窗口，切换不了，start_time进行还原
                        start_time += task_info[index_task[int(x[i - 1])]]["end_action"]
        gantt_dict = dict()
        for key, value in time_main_action.items():
            if key in index_task:
                gantt_dict.setdefault(self.index_task[key], []).append(
                    value
                )
        task_main_action_list = list(task_main_action.values())
        # 使用filter函数去除值为0的元素
        time_main_action2 = dict()
        for key, value in time_main_action.items():
            time_main_action2[str(key)] = value
        # import json
        # with open("file_jin.txt", "w") as file:
        #     json.dump(time_main_action2, file)
        task_main_action_list = list(filter(lambda x: x != 0, task_main_action_list))
        obj1 = self.n_var - sum(task_main_action_list)  # 统计总的主动作[最大化]->[600-(每个任务的主动作之和)]
        obj2 = 1-(sum(task_main_action_list)/30/len(task_main_action_list))  # 任务的总计完成度[最大化]->[1-每个任务的完成度之和的平均值]
        obj3 = next_times * 40  # 总的切换时长[最小化]
        # obj2 = 0  # 任务的总计完成度[最大化]->[1-每个任务的完成度之和的平均值]
        # obj3 = 0  # 总的切换时长[最小化]
        return np.array([obj1, obj2, obj3]), f'{time_main_action}'
