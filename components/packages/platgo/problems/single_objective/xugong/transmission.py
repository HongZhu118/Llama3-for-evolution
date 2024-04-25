import random
import datetime

import numpy as np
import pandas as pd

from ....Problem import Problem
from .....platgo import Population


class XugongTransmission(Problem):
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
        self.worker_index2string = optimization_problem["worker_index2string"]
        self.worker_string2index = optimization_problem["worker_string2index"]
        super(XugongTransmission, self).__init__(
            optimization_problem, debug=debug
        )

    def init_pop(self, N: int = None):
        if N is None:
            N = self.pop_size

        data = self.data[0]
        dec = []
        for k in range(N):
            dec1 = []
            for i in range(len(data["scheduling_task"])):
                experiment_i = random.choice(
                    data["scheduling_task"][i]["available_device"]
                )
                pop_i = random.choice(
                    data["scheduling_task"][i]["available_worker"]
                )
                dec1 += [experiment_i, pop_i]
            dec += [dec1]
        dec = np.array(dec, dtype=np.float64)
        return Population(decs=dec)

    def fix_decs(self, pop):
        data = self.data[0]
        for i in range(len(pop)):
            for j in range(len(data["scheduling_task"])):
                if (
                    pop.decs[i][2 * j]
                    not in data["scheduling_task"][j]["available_device"]
                ):
                    pop.decs[i][2 * j] = random.choice(
                        data["scheduling_task"][j]["available_device"]
                    )
                if (
                    pop.decs[i][2 * j + 1]
                    not in data["scheduling_task"][j]["available_worker"]
                ):
                    pop.decs[i][2 * j + 1] = random.choice(
                        data["scheduling_task"][j]["available_worker"]
                    )
        return pop

    def compute(self, pop) -> None:
        objv = np.zeros((pop.decs.shape[0], 1))
        for i, x in enumerate(pop.decs):
            objv[i] = self.main(x)
        pop.objv = objv
        pop.cv = np.zeros((pop.pop_size, self.n_constr))

    def pan(
        self,
        device,
        worker,
        data,
        using_worker,
        using_device,
        init_time,
        time_tmp,
        time1,
        task_type,
        time_list,
        pop_dict,
        worker_string2index,
        worker_index2string,
        pre,
    ):
        """
        判断待排任务与固定任务是否冲突，时间冲突则进行调整
        :param device: 待排任务占用的设备id
        :param worker: 待排任务占用的人员id
        :param data: 数据集
        :param using_worker: 每个固定任务所需人员id
        :param using_device: 每个固定任务所需设备id
        :param init_time: 数据集中给出的开始时间
        :param time_tmp: 该待排任务的结束时间
        :param time1: 该待排任务的实验时长
        :param task_type: 实验类型
        :param time_list: # 每个设备的使用时间段
        :param pop_dict:  # 每个人员的占用时间段
        :param worker_string2index
        :param worker_index2string
        :param pre
        :return: 该待排任务的在解决冲突后的开始时间和结束时间
        """
        time = []
        if pre:  # 当前待排任务顺序靠后
            time = pre  # 将顺序在前的任务的时间加入
        # 在固定任务中找出与当前待排任务相同的人员的开始时间和结束时间
        # 其实也在固定任务中找出当前待排任务所需人员的所有的工作时间段
        if task_type == "可靠性":
            for i in np.where(using_worker == str(int(worker)))[0]:
                time.append(
                    [
                        data["scheduled_task"][i]["start_time"],
                        data["scheduled_task"][i]["end_time"],
                    ]
                )
        else:
            for j in worker_string2index[
                worker_index2string[str(int(worker))]
            ]:
                for i in np.where(using_worker == str(int(j)))[0]:
                    time.append(
                        [
                            data["scheduled_task"][i]["start_time"],
                            data["scheduled_task"][i]["end_time"],
                        ]
                    )

        # 在固定任务中找出与当前待排任务相同的设备的开始时间和结束时间
        # 其实也在固定任务中找出当前待排任务所需设备的所有工作时间段
        for i in np.where(using_device == str(int(device)))[0]:
            time.append(
                [
                    data["scheduled_task"][i]["start_time"],
                    data["scheduled_task"][i]["end_time"],
                ]
            )
        if str(int(device)) in time_list:
            for i in time_list[str(int(device))]:
                time.append(i)
        if str(int(worker)) in pop_dict:
            for i in pop_dict[str(int(worker))]:
                time.append(i)
        time.sort(key=lambda x: (x[0], x[1]))
        flag = True  # 判断待排任务是否固定 True为不固定
        # 根据时间表调整当前待排任务的开始时间和结束时间
        init_time, time_tmp = self.juddge(init_time, time_tmp, time1)
        for i in range(len(time)):
            # 判断当前的待排任务与固定任务的时间是否冲突
            if (
                time[i][0] < init_time < time[i][1]
                or time[i][0] < str(time_tmp) < time[i][1]
            ):
                init_time = time[i][1]  # 时间冲突，将待排任务的开始时间设置发生冲突固定任务的结束时间
            elif init_time <= time[i][0] and str(time_tmp) >= time[i][1]:
                init_time = time[i][1]  # 时间冲突，将待排任务的开始时间设置发生冲突固定任务的结束时间
            elif (
                str(time_tmp) <= time[i][0]
            ):  # 待排任务的结束时间在固定任务开始时间之前，此时待排任务的位置就可以固定了
                flag = False  # 待排任务和固定任务没有发生时间冲突
            # 解决时间冲突后，计算当前待排任务的实际的开始时间和结束时间
            init_time, time_tmp = self.juddge(init_time, time_tmp, time1)
            if not flag:  # flag为False说明待排任务已经解决与固定任务的时间冲突，且位置已经固定
                break
        return init_time, time_tmp

    """

    def pan(
        self,
        device,
        worker,
        data,
        using_worker,
        using_device,
        init_time,
        time_tmp,
        time1,
        time_list,
        pop_dict,
    ):
        time = []
        for i in np.where(using_worker == str(int(worker)))[0]:
            time.append(
                [
                    data["scheduled_task"][i]["start_time"],
                    data["scheduled_task"][i]["end_time"],
                ]
            )  # noqa
        for i in np.where(using_device == str(int(device)))[0]:
            time.append(
                [
                    data["scheduled_task"][i]["start_time"],
                    data["scheduled_task"][i]["end_time"],
                ]
            )
        if str(device) in time_list:
            for i in time_list[str(device)]:
                time.append(i)
        if str(worker) in pop_dict:
            for i in pop_dict[str(worker)]:
                time.append(i)
        time.sort(key=lambda x: (x[0], x[1]))
        flag = True
        init_time, time_tmp = self.juddge(init_time, time_tmp, time1)
        for i in range(len(time)):
            if (
                time[i][0] <= init_time <= time[i][1]
                or time[i][0] <= str(time_tmp) <= time[i][1]
            ):
                init_time = time[i][1]
            elif str(time_tmp) <= time[i][0]:
                flag = False
            elif init_time <= time[i][0] and str(time_tmp) >= time[i][1]:
                init_time = time[i][1]
            init_time, time_tmp = self.juddge(init_time, time_tmp, time1)
            if not flag:
                break
        return init_time, time_tmp
"""

    def juddge(self, init_time, time_tmp, time1):
        if "02-16" <= init_time[5:10] <= "07-14":
            if (
                "08:30:00" <= init_time[11:] < "12:00:00"
                or "13:30:00" <= init_time[11:] < "18:00:00"
            ):
                init_time = init_time
            elif "12:00:00" <= init_time[11:] < "13:30:00":
                init_time = init_time[:11] + "13:30:00"
            elif "00:00:00" <= init_time[11:] < "08:30:00":
                init_time = init_time[:11] + "08:30:00"
            else:
                init_time = (
                    str(
                        datetime.datetime.strptime(init_time[:10], "%Y-%m-%d")
                        + datetime.timedelta(days=1)
                    )[:11]
                    + "08:30:00"
                )
        elif "07-15" <= init_time[5:10] <= "09-30":
            if (
                "08:30:00" <= init_time[11:] < "12:00:00"
                or "14:00:00" <= init_time[11:] < "18:00:00"
            ):
                init_time = init_time
            elif "12:00:00" <= init_time[11:] < "14:00:00":
                init_time = init_time[:11] + "14:00:00"
            elif "00:00:00" <= init_time[11:] < "08:30:00":
                init_time = init_time[:11] + "08:30:00"
            else:
                init_time = (
                    str(
                        datetime.datetime.strptime(init_time[:10], "%Y-%m-%d")
                        + datetime.timedelta(days=1)
                    )[:11]
                    + "08:30:00"
                )
        elif "10-01" <= init_time[5:10] <= "11-15":
            if (
                "08:30:00" <= init_time[11:] < "12:00:00"
                or "13:30:00" <= init_time[11:] < "18:00:00"
            ):
                init_time = init_time
            elif "12:00:00" <= init_time[11:] < "13:30:00":
                init_time = init_time[:11] + "13:30:00"
            elif "00:00:00" <= init_time[11:] < "08:30:00":
                init_time = init_time[:11] + "08:30:00"
            else:
                init_time = (
                    str(
                        datetime.datetime.strptime(init_time[:10], "%Y-%m-%d")
                        + datetime.timedelta(days=1)
                    )[:11]
                    + "08:30:00"
                )
        else:
            if (
                "08:30:00" <= init_time[11:] < "12:00:00"
                or "13:30:00" <= init_time[11:] < "17:30:00"
            ):
                init_time = init_time
            elif "12:00:00" <= init_time[11:] < "13:30:00":
                init_time = init_time[:11] + "13:30:00"
            elif "00:00:00" <= init_time[11:] < "08:30:00":
                init_time = init_time[:11] + "08:30:00"
            else:
                init_time = (
                    str(
                        datetime.datetime.strptime(init_time[:10], "%Y-%m-%d")
                        + datetime.timedelta(days=1)
                    )[:11]
                    + "08:30:00"
                )
        time_tmp = datetime.datetime.strptime(
            init_time, "%Y-%m-%d %H:%M:%S"
        ) + datetime.timedelta(minutes=time1)
        return init_time, time_tmp

    def main(self, x):
        data = self.data[0]
        init_time = data.get("schedule_start_time", "2023-01-19 00:00:00")
        available_device = []
        available_pop = []
        i = max(len(data["scheduling_task"]), len(data["scheduled_task"]))
        for k in range(i):
            if k < len(data["scheduling_task"]):
                available_device.append(
                    data["scheduling_task"][k]["available_device"]
                )
                available_pop.append(
                    data["scheduling_task"][k]["available_worker"]
                )
            if k < len(data["scheduled_task"]):
                available_pop.append(data["scheduled_task"][k]["using_worker"])
                available_device.append(
                    data["scheduled_task"][k]["using_device"]
                )
        available_device = list(set(sum(available_device, [])))
        available_pop = list(set(sum(available_pop, [])))
        D = len(x)
        experiment_id = {}
        if len(data["scheduling_task"]) != 0:
            if data["scheduling_task"][0]["laboratory"] == "CD":
                for j in range(0, D, 2):
                    if (
                        "plan_start_time"
                        in data["scheduling_task"][int(j / 2)]
                    ):
                        tt = data["scheduling_task"][int(j / 2)][
                            "experiment_id"
                        ]
                        if (
                            data["scheduling_task"][int(j / 2)][
                                "experiment_id"
                            ]
                            not in experiment_id
                        ):
                            experiment_id.setdefault(tt, []).append(
                                (
                                    data["scheduling_task"][int(j / 2)][
                                        "plan_start_time"
                                    ][0:7],
                                    data["scheduling_task"][int(j / 2)][
                                        "task_order"
                                    ],
                                )
                            )
                        else:
                            if (
                                data["scheduling_task"][int(j / 2)][
                                    "plan_start_time"
                                ][0:7]
                                < experiment_id[
                                    data["scheduling_task"][int(j / 2)][
                                        "experiment_id"
                                    ]
                                ][0][0]
                            ):
                                tt = data["scheduling_task"][int(j / 2)][
                                    "experiment_id"
                                ]
                                experiment_id[tt] = [
                                    (
                                        data["scheduling_task"][int(j / 2)][
                                            "plan_start_time"
                                        ][0:7],
                                        data["scheduling_task"][int(j / 2)][
                                            "task_order"
                                        ],
                                    )
                                ]
                    else:
                        tt = data["scheduling_task"][int(j / 2)][
                            "experiment_id"
                        ]
                        if (
                            data["scheduling_task"][int(j / 2)][
                                "experiment_id"
                            ]
                            not in experiment_id
                        ):
                            experiment_id.setdefault(tt, []).append(
                                (
                                    "3000-13",
                                    data["scheduling_task"][int(j / 2)][
                                        "task_order"
                                    ],
                                )
                            )
        pop_i = []  # 对应实验台上实验人的情况
        time = []
        experiment = []
        for k in range(1, len(available_device) + 1):
            experiment_time = []
            people = []
            order = []
            seq1 = []
            t = []
            for j in range(0, D, 2):
                if x[j] == k:
                    people.append(x[j + 1])
                    experiment_time.append(
                        data["scheduling_task"][int(j / 2)]["task_duration"]
                    )
                    tt = data["scheduling_task"][int(j / 2)]["experiment_id"]
                    if (
                        "plan_start_time"
                        in data["scheduling_task"][int(j / 2)]
                    ):  # 有计划开始时间的待排任务
                        if (
                            data["scheduling_task"][int(j / 2)]["laboratory"]
                            == "CD"
                        ):  # 该待排任务属于传动实验室
                            # experiment_id[tt]->[('2023-03', 5)]
                            seq1.append(
                                (
                                    data["scheduling_task"][int(j / 2)][
                                        "task_sort"
                                    ],
                                    experiment_id[tt][0][0],
                                    experiment_id[tt][0][1],
                                    tt,
                                )
                            )
                        else:  # 该待排任务属于普通实验室
                            seq1.append(
                                (
                                    data["scheduling_task"][int(j / 2)][
                                        "task_sort"
                                    ],
                                    data["scheduling_task"][int(j / 2)][
                                        "plan_start_time"
                                    ][
                                        0:7
                                    ],  # noqa
                                    data["scheduling_task"][int(j / 2)][
                                        "task_order"
                                    ],
                                    tt,
                                )
                            )  # noqa
                    else:  # 没有计划开始时间的待排任务
                        if (
                            data["scheduling_task"][int(j / 2)]["laboratory"]
                            == "CD"
                        ):  # 该待排任务属于传动实验室
                            seq1.append(
                                (
                                    data["scheduling_task"][int(j / 2)][
                                        "task_sort"
                                    ],
                                    "3000-13",
                                    experiment_id[tt][0][1],
                                    tt,
                                )
                            )
                        else:  # 该待排任务属于普通实验室
                            seq1.append(
                                (
                                    data["scheduling_task"][int(j / 2)][
                                        "task_sort"
                                    ],
                                    "3000-13",
                                    data["scheduling_task"][int(j / 2)][
                                        "task_order"
                                    ],
                                    tt,
                                )
                            )
                    order.append(
                        data["scheduling_task"][int(j / 2)]["task_order"]
                    )
                    t.append(int(j / 2))
            # 根据元组第0个值升序排序，若第0个值相等则根据第1个值升序排序
            # 先按照年月份排序再按照优先级排序
            tmp1 = sorted(
                range(len(t)),
                key=lambda x: (seq1[x][0], seq1[x][1], seq1[x][2]),
            )
            pop_i.append(np.array(people)[tmp1])
            experiment.append(np.array(t)[tmp1])
            time.append(np.array(experiment_time)[tmp1])
        pop_i = pd.DataFrame(pop_i)
        experiment = pd.DataFrame(experiment)
        time = pd.DataFrame(time)

        time_list = (
            {}
        )  # 记录每个设备的占用时间段{'1': [['2023-03-30 10:20:00', '2023-03-30 11:10:00']]}
        pop_dict = (
            {}
        )  # 记录每个人员的占用时间段{'2': [['2023-03-30 10:20:00', '2023-03-30 11:10:00']]}
        experiment_done = {}  # 判断实验是否做过
        Flag = {}  # 存放已做实验的时间(key值为task_id)
        Flag_c = [0] * experiment.shape[0]  # 存放设备的第几个实验
        using_worker = []
        using_device = []
        if len(data["scheduled_task"]):
            using_worker = np.array(
                pd.DataFrame(data["scheduled_task"])["using_worker"]
            )
            using_device = np.array(
                pd.DataFrame(data["scheduled_task"])["using_device"]
            )
        for k in range(len(using_device)):
            using_worker[k] = using_worker[k][0]
            using_device[k] = using_device[k][0]

        # 记录同一experiment_id下的待排任务的task_id和task_sort
        same_experiment_id_task = {}
        for item in data["scheduling_task"]:
            same_experiment_id_task.setdefault(
                item["experiment_id"], []
            ).append((item["task_id"], item["task_sort"]))

        # experiment表的最后添加一列空值
        last_col = experiment.shape[1]
        experiment[last_col] = np.nan

        while len(experiment_done) != len(
            data["scheduling_task"]
        ):  # 确保所有的实验全部做完
            for j in range(experiment.shape[0]):  # 对每个设备进行遍历
                c = Flag_c[j]  # 存放当前设备的第几个实验
                if ~np.isnan(experiment[c][j]):
                    task_sort = data["scheduling_task"][int(experiment[c][j])][
                        "task_sort"
                    ]  # 待排任务的顺序
                    task_list = []
                    if task_sort > 1:  # 待排任务顺序靠后
                        task_flag = True  # 判断任务顺序合规
                        for item in same_experiment_id_task[
                            data["scheduling_task"][int(experiment[c][j])][
                                "experiment_id"
                            ]
                        ]:
                            if item[1] < task_sort:  # 找到顺序在前的项目的id
                                task_list.append(item[0])
                        for tid in task_list:
                            if tid not in Flag:  # 顺序在前的项目没有做
                                task_flag = False  # 不合规
                        if not task_flag:  # 不合规跳过该项目
                            continue
                    if c == 0:  # 先看每个设备上的第一个项目
                        if (
                            str(int(pop_i[c][j])) not in pop_dict
                        ):  # 该待排任务的所需人员不在pop_dict中
                            init_time1 = init_time
                        else:
                            if (
                                data["scheduling_task"][int(experiment[c][j])][
                                    "task_type"
                                ]
                                == "性能"
                            ):
                                init_time1 = init_time
                                for i in self.worker_string2index[
                                    self.worker_index2string[
                                        str(int(pop_i[c][j]))
                                    ]
                                ]:
                                    if str(i) in pop_dict:
                                        if (
                                            pop_dict[str(i)][-1][1]
                                            > init_time1
                                        ):
                                            init_time1 = pop_dict[str(i)][-1][
                                                1
                                            ]
                            else:
                                init_time1 = pop_dict[str(int(pop_i[c][j]))][
                                    -1
                                ][1]
                        pre = []  # 顺序在前待排任务的开始和结束时间
                        if len(task_list) != 0:
                            for tid in task_list:
                                pre.append(Flag[tid])
                            pre.sort(key=lambda x: (x[0], x[1]))
                            init_time1 = max(pre[-1][1], init_time1)
                        time_tmp = datetime.datetime.strptime(
                            init_time1, "%Y-%m-%d %H:%M:%S"
                        ) + datetime.timedelta(
                            minutes=data["scheduling_task"][
                                int(experiment[c][j])
                            ]["task_duration"]
                        )
                    else:  # 再看场地的第二个及以后的实验
                        if str(int(pop_i[c][j])) not in pop_dict:
                            init_time1 = time_list[str(j + 1)][-1][1]
                        else:
                            if (
                                data["scheduling_task"][int(experiment[c][j])][
                                    "task_type"
                                ]
                                == "性能"
                            ):
                                init_time1 = init_time
                                for i in self.worker_string2index[
                                    self.worker_index2string[
                                        str(int(pop_i[c][j]))
                                    ]
                                ]:
                                    if str(i) in pop_dict:
                                        if (
                                            pop_dict[str(i)][-1][1]
                                            > init_time1
                                        ):
                                            init_time1 = pop_dict[str(i)][-1][
                                                1
                                            ]
                            else:
                                init_time1 = pop_dict[str(int(pop_i[c][j]))][
                                    -1
                                ][1]
                            init_time1 = max(
                                time_list[str(j + 1)][-1][1], init_time1,
                            )
                            pre = []  # 顺序在前待排任务的开始和结束时间
                            if len(task_list) != 0:
                                for tid in task_list:
                                    pre.append(Flag[tid])
                                pre.sort(key=lambda x: (x[0], x[1]))
                                init_time1 = max(pre[-1][1], init_time1)
                        time_tmp = datetime.datetime.strptime(
                            init_time1, "%Y-%m-%d %H:%M:%S"
                        ) + datetime.timedelta(
                            minutes=data["scheduling_task"][
                                int(experiment[c][j])
                            ]["task_duration"]
                        )
                    Flag_c[j] += 1  # 第j个设备的项目加1

                    init_time1, time_tmp = self.pan(
                        j + 1,
                        pop_i[c][j],
                        data,
                        using_worker,
                        using_device,
                        init_time1,
                        time_tmp,
                        data["scheduling_task"][int(experiment[c][j])][
                            "task_duration"
                        ],
                        data["scheduling_task"][int(experiment[c][j])][
                            "task_type"
                        ],
                        time_list,
                        pop_dict,
                        self.worker_string2index,
                        self.worker_index2string,
                        pre,
                    )
                    # 更新设备占用表
                    time_list.setdefault(str(int(j + 1)), []).append(
                        [init_time1, str(time_tmp)]
                    )
                    # 更新人员占用表
                    if (
                        data["scheduling_task"][int(experiment[c][j])][
                            "task_type"
                        ]
                        == "性能"
                    ):
                        for i in self.worker_string2index[
                            self.worker_index2string[str(int(pop_i[c][j]))]
                        ]:
                            pop_dict.setdefault(str(i), []).append(
                                [init_time1, str(time_tmp)]
                            )
                    else:
                        pop_dict.setdefault(str(int(pop_i[c][j])), []).append(
                            [init_time1, str(time_tmp)]
                        )
                    # 记录已做的实验
                    experiment_done[str(int(experiment[c][j]))] = True
                    Flag[
                        data["scheduling_task"][int(experiment[c][j])][
                            "task_id"
                        ]
                    ] = [init_time1, str(time_tmp)]
        f = init_time
        for q in range(len(available_device)):
            if str(q) in time_list:
                if q == 0:
                    f = time_list[str(q)][-1][1]
                else:
                    if f <= time_list[str(q)][-1][1]:
                        f = time_list[str(q)][-1][1]
        f = datetime.datetime.strptime(
            f, "%Y-%m-%d %H:%M:%S"
        ) - datetime.datetime.strptime(init_time, "%Y-%m-%d %H:%M:%S")
        f = f.days * 24 * 3600 + f.seconds  # 秒当作目标值
        return f

    """
        def main(self, x):
        data = self.data[0]
        init_time = data.get("schedule_start_time", "2023-01-19 00:00:00")
        available_device = []
        available_pop = []
        i = max(len(data["scheduling_task"]), len(data["scheduled_task"]))
        for k in range(i):
            if k < len(data["scheduling_task"]):
                available_device.append(
                    data["scheduling_task"][k]["available_device"]
                )
                available_pop.append(
                    data["scheduling_task"][k]["available_worker"]
                )
            if k < len(data["scheduled_task"]):
                available_pop.append(data["scheduled_task"][k]["using_worker"])
                available_device.append(
                    data["scheduled_task"][k]["using_device"]
                )
        available_device = list(set(sum(available_device, [])))
        available_pop = list(set(sum(available_pop, [])))
        D = len(x)
        time_list = {}
        pop_dict = {}
        pop_i = []  # 对应实验台上实验人的情况
        time = []
        experiment = []
        experiment_id = {}
        if len(data["scheduling_task"]) != 0:
            if data["scheduling_task"][0]["laboratory"] == "CD":
                for j in range(0, D, 2):
                    if (
                        "plan_start_time"
                        in data["scheduling_task"][int(j / 2)]
                    ):
                        tt = data["scheduling_task"][int(j / 2)][
                            "experiment_id"
                        ]
                        if (
                            data["scheduling_task"][int(j / 2)][
                                "experiment_id"
                            ]
                            not in experiment_id
                        ):
                            experiment_id.setdefault(tt, []).append(
                                (
                                    data["scheduling_task"][int(j / 2)][
                                        "plan_start_time"
                                    ][0:7],
                                    data["scheduling_task"][int(j / 2)][
                                        "task_order"
                                    ],
                                )
                            )
                        else:
                            if (
                                data["scheduling_task"][int(j / 2)][
                                    "plan_start_time"
                                ][0:7]
                                < experiment_id[
                                    data["scheduling_task"][int(j / 2)][
                                        "experiment_id"
                                    ]
                                ][0][0]
                            ):
                                tt = data["scheduling_task"][int(j / 2)][
                                    "experiment_id"
                                ]
                                experiment_id[tt] = [
                                    (
                                        data["scheduling_task"][int(j / 2)][
                                            "plan_start_time"
                                        ][0:7],
                                        data["scheduling_task"][int(j / 2)][
                                            "task_order"
                                        ],
                                    )
                                ]
                    else:
                        tt = data["scheduling_task"][int(j / 2)][
                            "experiment_id"
                        ]
                        if (
                            data["scheduling_task"][int(j / 2)][
                                "experiment_id"
                            ]
                            not in experiment_id
                        ):
                            experiment_id.setdefault(tt, []).append(
                                (
                                    "3000-13",
                                    data["scheduling_task"][int(j / 2)][
                                        "task_order"
                                    ],
                                )
                            )
        for k in range(1, len(available_device) + 1):
            experiment_time = []
            people = []
            order = []
            seq1 = []
            t = []
            for j in range(0, D, 2):
                if x[j] == k:
                    people.append(x[j + 1])
                    experiment_time.append(
                        data["scheduling_task"][int(j / 2)]["task_duration"]
                    )
                    tt = data["scheduling_task"][int(j / 2)]["experiment_id"]
                    if (
                        "plan_start_time"
                        in data["scheduling_task"][int(j / 2)]
                    ):
                        if (
                            data["scheduling_task"][int(j / 2)]["laboratory"]
                            == "CD"
                        ):
                            seq1.append(
                                (
                                    experiment_id[tt][0][0],
                                    experiment_id[tt][0][1],
                                    tt,
                                )
                            )
                        else:
                            seq1.append(
                                (
                                    data["scheduling_task"][int(j / 2)][
                                        "plan_start_time"
                                    ][0:7],
                                    data["scheduling_task"][int(j / 2)][
                                        "task_order"
                                    ],
                                    tt,
                                )
                            )
                    else:
                        if (
                            data["scheduling_task"][int(j / 2)]["laboratory"]
                            == "CD"
                        ):
                            seq1.append(
                                ("3000-13", experiment_id[tt][0][1], tt)
                            )
                        else:
                            seq1.append(
                                (
                                    "3000-13",
                                    data["scheduling_task"][int(j / 2)][
                                        "task_order"
                                    ],
                                    tt,
                                )
                            )
                    order.append(
                        data["scheduling_task"][int(j / 2)]["task_order"]
                    )
                    t.append(int(j / 2))

            tmp1 = sorted(
                range(len(t)), key=lambda x: (seq1[x][0], seq1[x][1])
            )
            pop_i.append(np.array(people)[tmp1])
            time.append(np.array(experiment_time)[tmp1])
            experiment.append(np.array(t)[tmp1])
        time = pd.DataFrame(time)
        pop_i = pd.DataFrame(pop_i)
        experiment = pd.DataFrame(experiment)

        using_worker = []
        using_device = []
        if len(data["scheduled_task"]):
            using_worker = np.array(
                pd.DataFrame(data["scheduled_task"])["using_worker"]
            )
            using_device = np.array(
                pd.DataFrame(data["scheduled_task"])["using_device"]
            )
        for k in range(len(using_device)):
            using_worker[k] = using_worker[k][0]
            using_device[k] = using_device[k][0]
        for h in range(time.shape[1]):
            s = 1
            for h1 in range(time.shape[0]):
                if h == 0:
                    if ~np.isnan(pop_i[h][h1]):
                        if str(int(pop_i[h][h1])) not in pop_dict:
                            time_tmp = datetime.datetime.strptime(
                                init_time, "%Y-%m-%d %H:%M:%S"
                            ) + datetime.timedelta(
                                minutes=data["scheduling_task"][
                                    int(experiment[h][h1])
                                ]["task_duration"]
                            )
                            init_time1, time_tmp = self.pan(
                                h1 + 1,
                                pop_i[h][h1],
                                data,
                                using_worker,
                                using_device,
                                init_time,
                                time_tmp,
                                data["scheduling_task"][
                                    int(experiment[h][h1])
                                ]["task_duration"],
                                time_list,
                                pop_dict,
                            )
                            time_list.setdefault(str(int(s)), []).append(
                                [init_time1, str(time_tmp)]
                            )
                            pop_dict.setdefault(
                                str(int(pop_i[h][h1])), []
                            ).append([init_time1, str(time_tmp)])
                        else:
                            init_time1 = pop_dict[str(int(pop_i[h][h1]))][-1][
                                1
                            ]
                            time_tmp = datetime.datetime.strptime(
                                init_time1, "%Y-%m-%d %H:%M:%S"
                            ) + datetime.timedelta(
                                minutes=data["scheduling_task"][
                                    int(experiment[h][h1])
                                ]["task_duration"]
                            )
                            init_time1, time_tmp = self.pan(
                                h1 + 1,
                                pop_i[h][h1],
                                data,
                                using_worker,
                                using_device,
                                init_time1,
                                time_tmp,
                                data["scheduling_task"][
                                    int(experiment[h][h1])
                                ]["task_duration"],
                                time_list,
                                pop_dict,
                            )
                            time_list.setdefault(str(int(s)), []).append(
                                [init_time1, str(time_tmp)]
                            )
                            pop_dict.setdefault(
                                str(int(pop_i[h][h1])), []
                            ).append([init_time1, str(time_tmp)])
                    s += 1
                else:
                    if ~np.isnan(pop_i[h][h1]):
                        if str(int(pop_i[h][h1])) not in pop_dict:
                            init_time1 = time_list[str(s)][-1][1]
                            time_tmp = datetime.datetime.strptime(
                                init_time1, "%Y-%m-%d %H:%M:%S"
                            ) + datetime.timedelta(
                                minutes=data["scheduling_task"][
                                    int(experiment[h][h1])
                                ]["task_duration"]
                            )

                            init_time1, time_tmp = self.pan(
                                h1 + 1,
                                pop_i[h][h1],
                                data,
                                using_worker,
                                using_device,
                                init_time1,
                                time_tmp,
                                data["scheduling_task"][
                                    int(experiment[h][h1])
                                ]["task_duration"],
                                time_list,
                                pop_dict,
                            )
                            time_list.setdefault(str(int(s)), []).append(
                                [init_time1, str(time_tmp)]
                            )
                            pop_dict.setdefault(
                                str(int(pop_i[h][h1])), []
                            ).append([init_time1, str(time_tmp)])
                        else:
                            init_time1 = max(
                                time_list[str(s)][-1][1],
                                pop_dict[str(int(pop_i[h][h1]))][-1][1],
                            )
                            time_tmp = datetime.datetime.strptime(
                                init_time1, "%Y-%m-%d %H:%M:%S"
                            ) + datetime.timedelta(
                                minutes=data["scheduling_task"][
                                    int(experiment[h][h1])
                                ]["task_duration"]
                            )
                            init_time1, time_tmp = self.pan(
                                h1 + 1,
                                pop_i[h][h1],
                                data,
                                using_worker,
                                using_device,
                                init_time1,
                                time_tmp,
                                data["scheduling_task"][
                                    int(experiment[h][h1])
                                ]["task_duration"],
                                time_list,
                                pop_dict,
                            )
                            time_list.setdefault(str(int(s)), []).append(
                                [init_time1, str(time_tmp)]
                            )
                            pop_dict.setdefault(
                                str(int(pop_i[h][h1])), []
                            ).append([init_time1, str(time_tmp)])
                    s += 1
        f = init_time
        for q in range(len(available_device)):
            if str(q) in time_list:
                if q == 0:
                    f = time_list[str(q)][-1][1]
                else:
                    if f <= time_list[str(q)][-1][1]:
                        f = time_list[str(q)][-1][1]
        f = datetime.datetime.strptime(
            f, "%Y-%m-%d %H:%M:%S"
        ) - datetime.datetime.strptime(init_time, "%Y-%m-%d %H:%M:%S")
        f = f.days * 24 * 3600 + f.seconds  # 秒当作目标值
        return f

    """
