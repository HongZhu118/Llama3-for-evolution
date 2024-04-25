import sys
import os
from queue import Queue  # noqa
import random
import numpy as np
import pandas as pd
import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append("./components/packages")

from components.packages import platgo as pg  # noqa
from components.packages.platgo.algorithms import DE  # noqa
from components.packages.platgo.problems.single_objective.xugong.transmission import (  # noqa
    XugongTransmission,
)
from components.packages.common.external_optimization_problem import (  # noqa
    ext_opt_prob_cb,
)


def preprocessing(data_set):
    device_string2index = dict()
    worker_string2index = dict()
    device_index2string = dict()
    worker_index2string = dict()
    device_index = 1
    worker_index = 0
    tasks = len(data_set["scheduling_task"])

    for item in data_set["scheduling_task"]:
        for i, device_string in enumerate(item["available_device"]):
            if device_string not in device_string2index:
                device_string2index[device_string] = []
                device_index2string[str(device_index)] = []

                device_value = item["available_device"][device_string]
                for k in range(device_index, device_index + device_value):
                    device_string2index[device_string] += [k]
                    device_index2string[str(k)] = device_string
                    device_index += 1
            item["available_device"][device_string] = [
                str(i) for i in device_string2index[device_string]
            ]
        item["available_device"] = sum(
            list(map(lambda x: x, item["available_device"].values())), []
        )
        for i, worker_string in enumerate(item["available_worker"]):
            if worker_string not in worker_string2index:
                worker_string2index[worker_string] = []
                worker_index2string[str(worker_index)] = []

                worker_value = item["available_worker"][worker_string]
                for k in range(worker_index, worker_index + worker_value):
                    worker_string2index[worker_string] += [k]
                    worker_index2string[str(k)] = worker_string
                    worker_index += 1
            item["available_worker"][worker_string] = [
                str(i) for i in worker_string2index[worker_string]
            ]
        # 使用lambda函数结合map()方法获取字典中的所有值,并进行列表合并
        if item["task_type"] == "可靠性":
            item["available_worker"] = sum(
                list(map(lambda x: x, item["available_worker"].values())), []
            )
        else:
            item["available_worker"] = list(
                map(lambda x: x, item["available_worker"].values())
            )

    for item in data_set["scheduled_task"]:
        for i, device_string in enumerate(item["using_device"]):
            if device_string not in device_string2index:
                device_string2index[device_string] = []
                device_index2string[str(device_index)] = []
                device_value = item["using_device"][device_string]
                for k in range(device_index, device_index + device_value):
                    device_string2index[device_string] += [k]
                    device_index2string[str(k)] = device_string
                    device_index += 1

            item["using_device"] = [
                str(int(random.choice(device_string2index[device_string])))
            ]

        for i, worker_string in enumerate(item["using_worker"]):
            if worker_string not in worker_string2index:
                worker_string2index[worker_string] = []
                worker_index2string[str(worker_index)] = []
                worker_value = item["using_worker"][worker_string]
                for k in range(worker_index, worker_index + worker_value):
                    worker_string2index[worker_string] += [k]
                    worker_index2string[str(k)] = worker_string
                    worker_index += 1

            if item["task_type"] == "可靠性":
                item["using_worker"] = [
                    str(random.choice(worker_string2index[worker_string]))
                ]
            else:
                item["using_worker"] = [
                    str(i) for i in worker_string2index[worker_string]
                ]

    return (
        device_index2string,
        worker_index2string,
        device_index,
        worker_index,
        tasks,
    )


def postprocessing(
    data_set, algo_result, device_index2string, worker_index2string
):
    data = data_set
    init_time = data.get("schedule_start_time", "2023-01-19 00:00:00")
    algo_res = algo_result["data"]
    n_var = algo_result["n_var"]
    n_obj = algo_result["n_obj"]
    pop = np.array(algo_res)[1:, :n_var].astype(float).astype(int)
    # 最后一列是约束，倒数第二列是目标值
    objv_min = np.argmin(np.array(algo_res)[1:, n_var + n_obj - 1])
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
            available_device.append(data["scheduled_task"][k]["using_device"])
    available_device = list(set(sum(available_device, [])))
    available_pop = list(set(sum(available_pop, [])))
    D = len(pop[0])
    time_list = {}
    pop_dict = {}
    pop_i = []  # 对应实验台上实验人的情况
    time = []
    experiment = []
    experiment_id = {}
    if len(data["scheduling_task"]) != 0:
        if data["scheduling_task"][0]["laboratory"] == "CD":
            for j in range(0, D, 2):
                if "plan_start_time" in data["scheduling_task"][int(j / 2)]:
                    tt = data["scheduling_task"][int(j / 2)]["experiment_id"]
                    if (
                        data["scheduling_task"][int(j / 2)]["experiment_id"]
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
                    tt = data["scheduling_task"][int(j / 2)]["experiment_id"]
                    if (
                        data["scheduling_task"][int(j / 2)]["experiment_id"]
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
            if pop[objv_min][j] == k:
                people.append(pop[objv_min][j + 1])
                experiment_time.append(
                    data["scheduling_task"][int(j / 2)]["task_duration"]
                )
                tt = data["scheduling_task"][int(j / 2)]["experiment_id"]
                if "plan_start_time" in data["scheduling_task"][int(j / 2)]:
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
                        seq1.append(("3000-13", experiment_id[tt][0][1], tt))
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
                order.append(data["scheduling_task"][int(j / 2)]["task_order"])
                t.append(int(j / 2))

        tmp1 = sorted(range(len(t)), key=lambda x: (seq1[x][0], seq1[x][1]))
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
                        init_time1, time_tmp = pan(
                            h1 + 1,
                            pop_i[h][h1],
                            data,
                            using_worker,
                            using_device,
                            init_time,
                            time_tmp,
                            data["scheduling_task"][int(experiment[h][h1])][
                                "task_duration"
                            ],
                            time_list,
                            pop_dict,
                        )
                        time_list.setdefault(str(int(s)), []).append(
                            [init_time1, str(time_tmp)]
                        )
                        pop_dict.setdefault(str(int(pop_i[h][h1])), []).append(
                            [init_time1, str(time_tmp)]
                        )
                    else:
                        init_time1 = pop_dict[str(int(pop_i[h][h1]))][-1][1]
                        time_tmp = datetime.datetime.strptime(
                            init_time1, "%Y-%m-%d %H:%M:%S"
                        ) + datetime.timedelta(
                            minutes=data["scheduling_task"][
                                int(experiment[h][h1])
                            ]["task_duration"]
                        )
                        init_time1, time_tmp = pan(
                            h1 + 1,
                            pop_i[h][h1],
                            data,
                            using_worker,
                            using_device,
                            init_time1,
                            time_tmp,
                            data["scheduling_task"][int(experiment[h][h1])][
                                "task_duration"
                            ],
                            time_list,
                            pop_dict,
                        )
                        time_list.setdefault(str(int(s)), []).append(
                            [init_time1, str(time_tmp)]
                        )
                        pop_dict.setdefault(str(int(pop_i[h][h1])), []).append(
                            [init_time1, str(time_tmp)]
                        )
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

                        init_time1, time_tmp = pan(
                            h1 + 1,
                            pop_i[h][h1],
                            data,
                            using_worker,
                            using_device,
                            init_time1,
                            time_tmp,
                            data["scheduling_task"][int(experiment[h][h1])][
                                "task_duration"
                            ],
                            time_list,
                            pop_dict,
                        )
                        time_list.setdefault(str(int(s)), []).append(
                            [init_time1, str(time_tmp)]
                        )
                        pop_dict.setdefault(str(int(pop_i[h][h1])), []).append(
                            [init_time1, str(time_tmp)]
                        )
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
                        init_time1, time_tmp = pan(
                            h1 + 1,
                            pop_i[h][h1],
                            data,
                            using_worker,
                            using_device,
                            init_time1,
                            time_tmp,
                            data["scheduling_task"][int(experiment[h][h1])][
                                "task_duration"
                            ],
                            time_list,
                            pop_dict,
                        )
                        time_list.setdefault(str(int(s)), []).append(
                            [init_time1, str(time_tmp)]
                        )
                        pop_dict.setdefault(str(int(pop_i[h][h1])), []).append(
                            [init_time1, str(time_tmp)]
                        )
                s += 1
    outcome = []
    for h in range(time.shape[0]):
        for h1 in range(time.shape[1]):
            if ~np.isnan(experiment.loc[h][h1]):
                tmp = {
                    "experiment_id": data["scheduling_task"][
                        int(experiment.loc[h][h1])
                    ]["experiment_id"],
                    "laboratory": data["scheduling_task"][
                        int(experiment.loc[h][h1])
                    ]["laboratory"],
                    "task_id": data["scheduling_task"][
                        int(experiment.loc[h][h1])
                    ]["task_id"],
                    "start_time": time_list[str(h + 1)][h1][0],
                    "end_time": time_list[str(h + 1)][h1][1],
                    "using_device": [device_index2string[str(h + 1)]],
                    "using_worker": [
                        worker_index2string[str(int(pop_i.loc[h][h1]))]
                    ],
                }
                outcome.append(tmp)
    # outcome = test(outcome, data)
    return outcome


def pan(
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
    """
    判断待排任务与固定任务是否冲突，时间冲突则进行调整
    :param device: 待排任务占用的设备id
    :param worker: 待排任务占用的人员id
    :param data: 数据集
    :param using_worker: 每个固定任务所需人员id
    :param using_device: 每个固定任务所需设备id
    :param init_time: 待排任务预计的开始时间
    :param time_tmp: 该待排任务的结束时间
    :param time1: 该待排任务的实验时长
    :param time_list: # 每个设备的使用时间段
    :param pop_dict:  # 每个人员的占用时间段
    :return: 该待排任务的在解决冲突后的开始时间和结束时间
    """
    time = []
    # 在固定任务中找出与当前待排任务相同的人员的开始时间和结束时间
    # 其实也在固定任务中找出当前待排任务所需人员的所有的工作时间段

    for i in worker:  # 对当前待排任务所需人员遍历
        for j in range(len(using_worker)):
            if str(int(i)) in using_worker[j]:  # 判断待排任务所需人员是否与固定任务冲突
                time.append(
                    [
                        data["scheduled_task"][j]["start_time"],
                        data["scheduled_task"][j]["end_time"],
                    ]
                )
        if str(int(i)) in pop_dict:  # 如果当前待排任务与前面排好的任务所需人员相同
            for k in pop_dict[str(int(i))]:
                time.append(k)
    # 在固定任务中找出与当前待排任务相同的设备的开始时间和结束时间
    # 其实也在固定任务中找出当前待排任务所需设备的所有工作时间段
    for i in np.where(using_device == [str(int(device))])[0]:
        time.append(
            [
                data["scheduled_task"][i]["start_time"],
                data["scheduled_task"][i]["end_time"],
            ]
        )
    if str(int(device)) in time_list:
        for i in time_list[str(int(device))]:
            time.append(i)
    time.sort(key=lambda x: (x[0], x[1]))
    flag = True  # 判断待排任务是否固定 True为不固定
    # 根据时间表调整当前待排任务的开始时间和结束时间
    init_time, time_tmp = juddge(init_time, time_tmp, time1)
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
        init_time, time_tmp = juddge(init_time, time_tmp, time1)
        if not flag:  # flag为False说明待排任务已经解决与固定任务的时间冲突，且位置已经固定
            break
    return init_time, time_tmp


def juddge(init_time, time_tmp, time1):
    """
    根据时间表调整待排任务的开始时间和结束时间
    :param init_time: 待排任务的开始时间
    :param time_tmp: 待排任务的结束时间
    :param time1:  待排任务的实验时间
    :return:  调整后的开始时间和结束时间
    """
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


def search_sqp(data_set):

    (
        device_index2string,
        worker_index2string,
        device_index,
        worker_index,
        tasks,
    ) = preprocessing(data_set)

    optimization_problem = {
        "name": "XugongTransmission",
        "n_var": tasks * 2,
        "lower": [1, 0] * tasks,
        "upper": [device_index, worker_index] * tasks,
        "dataSet": [data_set],
        "algoResultType": 0,
    }

    pop_size = 20
    max_fe = 100
    options = {}

    def run_algo():
        evol_algo = pg.algorithms.DE(
            pop_size=pop_size,
            options=options,
            optimization_problem=optimization_problem,
            control_cb=None,
            max_fe=max_fe,
            name="GA-Thread",
            debug=True,
        )
        evol_algo.start()
        evol_algo.join()
        algo_result = evol_algo.get_external_algo_result()

        schedule_result = postprocessing(
            data_set, algo_result, device_index2string, worker_index2string
        )
        print(schedule_result)
        print("Done")
        return schedule_result

    return run_algo()


if __name__ == "__main__":
    data_set = pd.read_json("/home/xie/xugong/xugong-scheduling/传动0804.json")[
        "data"
    ]
    search_sqp(data_set)
    print("done")
