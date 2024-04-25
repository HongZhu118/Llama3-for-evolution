import random
import datetime

import numpy as np
import pandas as pd

from ....Problem import Problem
from .... import Population


class XugongComprehensive(Problem):
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
        super(XugongComprehensive, self).__init__(
            optimization_problem, debug=debug
        )

    def init_pop(self, N: int = None):
        if N is None:
            N = self.pop_size
        data1 = self.data[0]
        dec = []
        for k in range(N):
            dec1 = []
            for i in range(len(data1["scheduling_task"])):
                if "available_worker" in data1["scheduling_task"][i]:
                    pop_i = random.sample(
                        data1["scheduling_task"][i]["available_worker"],
                        k=data1["scheduling_task"][i]["needed_worker"],
                    )
                else:
                    pop_i = data1["scheduling_task"][i]["using_worker"]
                dec1.extend(pop_i)
            dec += [dec1]
        dec = np.array(dec, dtype=np.float64)
        return Population(decs=dec)

    def fix_decs(self, pop):
        data1 = self.data[0].copy()
        for i in range(pop.decs.shape[0]):
            nums = 0  # 第几个任务
            j = 0
            while j < len(pop.decs[1]):
                if "available_worker" in data1["scheduling_task"][nums]:
                    t = data1["scheduling_task"][nums]["available_worker"]
                    t1 = pop.decs[i][
                        j : j + data1["scheduling_task"][nums]["needed_worker"]
                    ]
                    # 不能出现可用人员外的人，也不可同一个人分配多次
                    if not all(i in t for i in t1) or len(t1) != len(set(t1)):
                        pop.decs[i][
                            j : j
                            + data1["scheduling_task"][nums]["needed_worker"]
                        ] = random.sample(
                            data1["scheduling_task"][nums]["available_worker"],
                            k=data1["scheduling_task"][nums]["needed_worker"],
                        )
                else:
                    t1 = data1["scheduling_task"][nums]["using_worker"]
                    for h in range(len(t1)):
                        pop.decs[i][j + h] = t1[h]
                j += len(t1)
                nums += 1

                print(f"decFcn: j={j} len(pop.decs[1])={len(pop.decs[1])}")
        return pop

    def compute(self, pop) -> None:
        objv = np.zeros((pop.decs.shape[0], 1))
        for i, x in enumerate(pop.decs):
            objv[i] = self.main(x)
        pop.objv = objv
        pop.cv = np.zeros((pop.pop_size, self.n_constr))

    def main(self, x):
        return self.run(x)

    def run(self, x):
        data1 = self.data[0].copy()
        init_time = data1["schedule_start_time"]
        time_dict = {}  # 场地时间
        pop_dict = {}  # 人员时间
        device_dict = {}  # 设备时间
        experiment_i = []
        pop_i = []  # 每个场地上人员情况
        field_nums = self.field_num(data1)
        residue_filed = {}
        print(x)
        # 统计分到同一个场地的任务
        for i in range(field_nums):
            j = 0
            nums = 0  # 第几个任务
            people = []
            experiment = []
            seq = []
            while j < len(x):
                if "available_worker" in data1["scheduling_task"][nums]:
                    t = data1["scheduling_task"][nums]["available_worker"]
                    j_ = data1["scheduling_task"][nums]["needed_worker"]
                else:
                    t = data1["scheduling_task"][nums]["using_worker"]
                    j_ = len(data1["scheduling_task"][nums]["using_worker"])
                if (
                    int(
                        list(
                            data1["scheduling_task"][nums][
                                "available_field"
                            ].keys()
                        )[0]
                    )
                    == i
                ):  # 第i个场地
                    residue_filed[i] = data1["scheduling_task"][nums][
                        "available_field"
                    ][str(i)]
                    if "available_worker" in data1["scheduling_task"][nums]:
                        people.append(
                            list(
                                x[
                                    j : j
                                    + data1["scheduling_task"][nums][
                                        "needed_worker"
                                    ]
                                ]
                            )
                        )
                    else:
                        people.append(
                            data1["scheduling_task"][nums]["using_worker"]
                        )
                    if "plan_start_time" in data1["scheduling_task"][nums]:
                        for h in range(j_):
                            seq.append(
                                (
                                    data1["scheduling_task"][nums][
                                        "plan_start_time"
                                    ][0:7],
                                    data1["scheduling_task"][nums][
                                        "task_order"
                                    ],
                                )
                            )
                            experiment.append(nums)
                    else:
                        for h in range(j_):
                            seq.append(
                                (
                                    "3000-13",
                                    data1["scheduling_task"][nums][
                                        "task_order"
                                    ],
                                )
                            )
                            experiment.append(nums)
                j += j_
                nums += 1
            people = sum(people, [])
            tmp = sorted(
                range(len(people)), key=lambda x: (seq[x][0], seq[x][1])
            )
            people = np.array(people, dtype=np.float64)[tmp]
            experiment = np.array(experiment)[tmp]
            pop_i.append(people)
            experiment_i.append(experiment)
        experiment_i = pd.DataFrame(experiment_i)
        pop_i = pd.DataFrame(pop_i)
        experiment = {}  # 判断实验是否做过
        Flag = {}  # 存放已做实验的时间
        Flag_i = [0] * field_nums  # 存放场地的第几个实验
        time_dict_i = {}  # 存放每个场地中最快结束和最慢结束的时间
        # 先看每个场地的第一个实验
        while len(experiment) != len(data1["scheduling_task"]):
            for j in range(experiment_i.shape[0]):
                i = Flag_i[j]
                if not np.isnan(experiment_i[i][j]):
                    if (
                        len(
                            data1["scheduling_task"][int(experiment_i[i][j])][
                                "pre_tasks"
                            ]
                        )
                        != 0
                    ):
                        if (
                            data1["scheduling_task"][int(experiment_i[i][j])][
                                "pre_tasks"
                            ][0]
                            not in Flag
                        ):
                            continue
                    if i == 0:
                        # 预估的结束时间
                        t = datetime.datetime.strptime(
                            init_time, "%Y-%m-%d %H:%M:%S"
                        ) + datetime.timedelta(
                            minutes=data1["scheduling_task"][
                                int(experiment_i[i][j])
                            ]["task_duration"]
                        )
                        start_time = init_time
                        if (
                            list(
                                data1["scheduling_task"][
                                    int(experiment_i[i][j])
                                ]["needed_field"].values()
                            )[0]
                            == residue_filed[j]
                        ):
                            num_flag = True
                        else:
                            num_flag = False
                    else:
                        #  实验做过了，要么没出现在experiment中，要么experiment中为True
                        if (
                            experiment_i[i][j] in experiment
                            and experiment[int(experiment_i[i][j])]
                        ):
                            Flag_i[j] += 1
                            continue
                            # 如果当前项目需要的场地数全部占用，则从最晚结束时间开始
                        if (
                            list(
                                data1["scheduling_task"][
                                    int(experiment_i[i][j])
                                ]["needed_field"].values()
                            )[0]
                            == residue_filed[j]
                        ):
                            num_flag = True
                            t = datetime.datetime.strptime(
                                time_dict_i[j]["end_slow"], "%Y-%m-%d %H:%M:%S"
                            ) + datetime.timedelta(
                                minutes=data1["scheduling_task"][
                                    int(experiment_i[i][j])
                                ]["task_duration"]
                            )
                            start_time = time_dict_i[j]["end_slow"]
                        else:  # 如果没有全部占用，则从最早结束时间开始
                            num_flag = False
                            t = datetime.datetime.strptime(
                                time_dict_i[j]["end_fast"], "%Y-%m-%d %H:%M:%S"
                            ) + datetime.timedelta(
                                minutes=data1["scheduling_task"][
                                    int(experiment_i[i][j])
                                ]["task_duration"]
                            )
                            start_time = time_dict_i[j]["end_fast"]
                    Flag_i[j] += 1
                    # 当前需要的人
                    if (
                        "needed_worker"
                        in data1["scheduling_task"][int(experiment_i[i][j])]
                    ):
                        people1 = list(
                            pop_i.iloc[j][
                                i : i
                                + data1["scheduling_task"][
                                    int(experiment_i[i][j])
                                ]["needed_worker"]
                            ]
                        )
                    else:
                        people1 = data1["scheduling_task"][
                            int(experiment_i[i][j])
                        ]["using_worker"]
                    device = list(
                        data1["scheduling_task"][int(experiment_i[i][j])][
                            "needed_equipment"
                        ].keys()
                    )
                    device1 = []
                    for h in range(len(device)):
                        device1 += [device[h].split(",")]
                    device1 = self.cal_cartesian_coord_2(device1)  # 笛卡尔积
                    time1 = []
                    for h in device1:
                        #  前置实验的时间
                        if not data1["scheduling_task"][
                            int(experiment_i[i][j])
                        ]["pre_tasks"]:
                            pre = []
                        else:
                            pre = Flag[
                                data1["scheduling_task"][
                                    int(experiment_i[i][j])
                                ]["pre_tasks"][0]
                            ]
                        start1, end1 = self.pan(
                            data1,
                            j,
                            people1,
                            time_dict,
                            pop_dict,
                            h,
                            device_dict,
                            start_time,
                            t,
                            pre,
                            residue_filed[j],
                            num_flag,
                        )
                        time1.append([start1, str(end1)])
                    # 选取最快的组合方式
                    # 更新设备表
                    for h1 in list(device1[np.argmin(np.array(time1)[:, 1])]):
                        device_dict.setdefault(h1, []).append(
                            time1[np.argmin(np.array(time1)[:, 1])]
                        )
                    # 更新人员表
                    for h in people1:
                        pop_dict.setdefault(h, []).append(
                            time1[np.argmin(np.array(time1)[:, 1])]
                        )
                    # 更新场地表
                    time_dict.setdefault(j, []).append(
                        time1[np.argmin(np.array(time1)[:, 1])]
                    )
                    experiment[int(experiment_i[i][j])] = True
                    Flag[
                        data1["scheduling_task"][int(experiment_i[i][j])][
                            "task_id"
                        ]
                    ] = time1[np.argmin(np.array(time1)[:, 1])]
                    if (
                        len(time_dict[j]) < residue_filed[j]
                    ):  # 当场地上任务数小于容量时，开始时间就是初始时间。
                        end_fast = init_time
                    else:
                        end_fast = np.array(time_dict[j])[
                            np.argmin(np.array(time_dict[j])[:, 1])
                        ][-1]
                    end_slow = np.array(time_dict[j])[
                        np.argmax(np.array(time_dict[j])[:, 1])
                    ][-1]
                    time_dict_i[j] = {
                        "end_fast": end_fast,
                        "end_slow": end_slow,
                    }
        f = init_time
        for i in range(field_nums):
            if i in time_dict_i:
                if f < time_dict_i[i]["end_slow"]:
                    f = time_dict_i[i]["end_slow"]
        f = datetime.datetime.strptime(
            f, "%Y-%m-%d %H:%M:%S"
        ) - datetime.datetime.strptime(init_time, "%Y-%m-%d %H:%M:%S")
        f = f.days * 24 * 3600 + f.seconds  # 秒当作目标值
        print("f", f)
        return f

    def cal_cartesian_coord_2(self, values):
        mesh = np.meshgrid(*values)
        cart = np.array(mesh).T.reshape(-1, len(values))
        return cart.tolist()

    def pan(
        self,
        data,
        field,
        worker,
        time_dict,
        pop_dict,
        device,
        device_dict,
        start,
        end,
        pre,
        residue_filed,
        num_flag,
    ):
        time = []
        if pre:
            time.append(pre)
        duration = end - datetime.datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
        for j in range(len(data["scheduled_task"])):
            if (
                int(list(data["scheduled_task"][j]["using_field"].keys())[0])
                == field
                and list(data["scheduled_task"][j]["using_field"].values())[0]
                == residue_filed
            ):
                time.append(
                    [
                        data["scheduled_task"][j]["start_time"],
                        data["scheduled_task"][j]["end_time"],
                    ]
                )
        for i in worker:
            for j in range(len(data["scheduled_task"])):
                if str(int(i)) in data["scheduled_task"][j]["using_worker"]:
                    time.append(
                        [
                            data["scheduled_task"][j]["start_time"],
                            data["scheduled_task"][j]["end_time"],
                        ]
                    )
            if i in pop_dict:
                time.append(pop_dict[i][-1])
        for i in device:
            for j in range(len(data["scheduled_task"])):
                if i in data["scheduled_task"][j]["using_equipment"]:
                    time.append(
                        [
                            data["scheduled_task"][j]["start_time"],
                            data["scheduled_task"][j]["end_time"],
                        ]
                    )
            if i in device_dict:
                for kk in range(len(device_dict[i])):
                    time.append(device_dict[i][kk])
        time.sort(key=lambda x: (x[0], x[1]))
        Flag = True
        while Flag:
            flag = True
            start, end = self.judege(start, end, duration)
            for i in range(len(time)):
                if (time[i][0] < start < time[i][1]) or (
                    time[i][0] < str(end) < time[i][1]
                ):
                    start = time[i][1]
                elif str(end) <= time[i][0]:
                    flag = False
                elif start <= time[i][0] and str(end) >= time[i][1]:
                    start = time[i][1]
                start, end = self.judege(start, end, duration)
                if not flag:
                    break
            time1 = [[start, str(end)]]
            if not num_flag:  # 没有全部占用的时候，需要考虑同场地，全部占用的时候，已经避开了场地其他
                # 任务，只需要跟固定任务比即可
                if field in time_dict:
                    time1 += time_dict[field]
            time1.sort(key=lambda x: (x[0], x[1]))
            for j in range(len(data["scheduled_task"])):
                if (
                    int(
                        list(data["scheduled_task"][j]["using_field"].keys())[
                            0
                        ]
                    )
                    == field
                ):
                    if (
                        int(
                            list(
                                data["scheduled_task"][j][
                                    "using_field"
                                ].values()
                            )[0]
                        )
                        != 1
                    ):
                        continue
                    #  t是固定任务的时间
                    t = [
                        data["scheduled_task"][j]["start_time"],
                        data["scheduled_task"][j]["end_time"],
                    ]
                    if not self.attack(
                        time1,
                        t,
                        int(
                            list(
                                data["scheduled_task"][j][
                                    "using_field"
                                ].values()
                            )[0]
                        ),
                        residue_filed,
                    ):
                        start = t[1]
                        end = (
                            datetime.datetime.strptime(
                                start, "%Y-%m-%d %H:%M:%S"
                            )
                            + duration
                        )
                        break
            else:
                Flag = False
        return start, end

    def judege(self, start, end, duration):
        if "02-16" < start[5:10] < "07-14":
            if (
                "08:30:00" <= start[11:] < "12:00:00"
                or "13:30:00" <= start[11:] < "18:00:00"
            ):
                start = start
            elif "12:00:00" <= start[11:] < "13:30:00":
                start = start[:11] + "13:30:00"
            elif "00:00:00" <= start[11:] < "08:30:00":
                start = start[:11] + "08:30:00"
            else:
                start = (
                    str(
                        datetime.datetime.strptime(start[:10], "%Y-%m-%d")
                        + datetime.timedelta(days=1)
                    )[:11]
                    + "08:30:00"
                )
        elif "07-15" < start[5:10] < "09-30":
            if (
                "08:30:00" <= start[11:] < "12:00:00"
                or "14:00:00" <= start[11:] < "18:00:00"
            ):
                start = start
            elif "12:00:00" <= start[11:] < "14:00:00":
                start = start[:11] + "14:00:00"
            elif "00:00:00" <= start[11:] < "08:30:00":
                start = start[:11] + "08:30:00"
            else:
                start = (
                    str(
                        datetime.datetime.strptime(start[:10], "%Y-%m-%d")
                        + datetime.timedelta(days=1)
                    )[:11]
                    + "08:30:00"
                )
        elif "10-01" < start[5:10] < "11-15":
            if (
                "08:30:00" <= start[11:] < "12:00:00"
                or "13:30:00" <= start[11:] < "18:00:00"
            ):
                start = start
            elif "12:00:00" <= start[11:] < "13:30:00":
                start = start[:11] + "13:30:00"
            elif "00:00:00" <= start[11:] < "08:30:00":
                start = start[:11] + "08:30:00"
            else:
                start = (
                    str(
                        datetime.datetime.strptime(start[:10], "%Y-%m-%d")
                        + datetime.timedelta(days=1)
                    )[:11]
                    + "08:30:00"
                )
        else:
            if (
                "08:30:00" <= start[11:] < "12:00:00"
                or "13:30:00" <= start[11:] < "17:30:00"
            ):
                start = start
            elif "12:00:00" <= start[11:] < "13:30:00":
                start = start[:11] + "13:30:00"
            elif "00:00:00" <= start[11:] < "08:30:00":
                start = start[:11] + "08:30:00"
            else:
                start = (
                    str(
                        datetime.datetime.strptime(start[:10], "%Y-%m-%d")
                        + datetime.timedelta(days=1)
                    )[:11]
                    + "08:30:00"
                )
        end = datetime.datetime.strptime(start, "%Y-%m-%d %H:%M:%S") + duration
        return start, end

    def field_num(self, data_set):
        field = 0
        field_map = []
        for item in data_set["scheduling_task"]:
            if list(item["available_field"].keys())[0] not in field_map:
                field_map.append(list(item["available_field"].keys())[0])
                field += 1
        for item in data_set["scheduled_task"]:
            if list(item["using_field"].keys())[0] not in field_map:
                field_map.append(list(item["using_field"].keys())[0])
                field += 1
        return field

    def attack(self, time, t, nums, residue_filed):
        s = 0
        for i in range(len(time)):
            if (
                (time[i][0] < t[0] < time[i][1])
                or (time[i][0] < t[1] < time[i][1])
                or (t[0] <= time[i][0] and t[1] >= time[i][1])
            ):
                s += 1
        if s + nums <= residue_filed:
            return True
        else:
            return False
