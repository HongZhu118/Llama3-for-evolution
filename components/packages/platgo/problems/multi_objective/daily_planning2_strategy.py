import random
import datetime
import copy
import numpy as np
import pandas as pd
import json
from components.packages.platgo.Problem import Problem
from components.packages.platgo import Population


class Daily_Planning2_Strategy(Problem):
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
        super(Daily_Planning2_Strategy, self).__init__(
            optimization_problem, debug=debug
        )

    def init_pop(self, N: int = None):
        if N is None:
            N = self.pop_size
        data = self.data[0]
        decs = list()
        # list1 = [91, 92, 1, 2, 3, 4, 5, 6, 7, 31, 241, 451, 452, 453, 454, 455, 456, 61, 62, 63, 64, 65, 66, 8, 9, 10, 11, 12, 13, 14, 32, 242, 457, 458, 459, 460, 461, 462, 67, 68, 69, 70, 71, 72, 15, 16, 17, 18, 19, 20, 21, 33, 243, 463, 464, 465, 466, 467, 468, 73, 74, 75, 76, 77, 78, 22, 23, 24, 25, 26, 27, 28, 34, 244, 469, 470, 471, 472, 79, 80, 81, 82, 83, 84, 29, 30, 35, 36, 37, 38, 39, 40, 245, 473, 474, 475, 85, 86, 87, 88, 89, 90, 41, 42, 43, 44, 45, 46, 47, 48, 246, 476, 211, 212, 213, 214, 215, 216, 49, 50, 51, 52, 53, 54, 55, 56, 247, 217, 218, 219, 220, 221, 222, 57, 58, 59, 60, 93, 94, 121, 248, 249, 477, 478, 223, 224, 225, 226, 227, 228, 95, 96, 97, 98, 99, 100, 101, 391, 479, 480, 481, 229, 230, 231, 232, 233, 234, 102, 103, 104, 105, 106, 107, 108, 392, 482, 483, 484, 485, 486, 235, 236, 237, 238, 239, 109, 110, 111, 112, 113, 114, 115, 393, 487, 488, 489, 490, 491, 240, 116, 117, 118, 119, 120, 122, 123, 124, 125, 126, 250, 251, 492, 493, 494, 495, 496, 497, 127, 128, 129, 130, 131, 132, 133, 134, 135, 151, 252, 253, 498, 499, 500, 501, 502, 503, 421, 136, 137, 138, 139, 140, 141, 142, 143, 144, 152, 254, 255, 504, 505, 506, 507, 508, 509, 422, 145, 146, 147, 148, 149, 150, 153, 154, 155, 156, 256, 257, 510, 423, 157, 158, 159, 160, 161, 162, 163, 164, 165, 258, 259, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 511]
        list1 = [91, 92, 1, 2, 3, 4, 5, 6, 7, 31, 241, 451, 452, 453, 454, 455, 456, 61, 62, 63, 64, 65, 66, 8, 9, 10, 11, 12, 13, 14, 32, 242, 457, 458, 459, 460, 461, 462, 67, 68, 69, 70, 71, 72, 15, 16, 17, 18, 19, 20, 21, 33, 243, 463, 464, 465, 466, 467, 468, 73, 74, 75, 76, 77, 78, 22, 23, 24, 25, 26, 27, 28, 34, 244, 469, 470, 471, 472, 79, 80, 81, 82, 83, 84, 29, 30, 35, 36, 37, 38, 39, 40, 245, 473, 474, 475, 85, 86, 87, 88, 89, 90, 41, 42, 43, 44, 45, 46, 47, 48, 246, 476, 211, 212, 213, 214, 215, 216, 49, 50, 51, 52, 53, 54, 55, 56, 247, 217, 218, 219, 220, 221, 222, 57, 58, 59, 60, 93, 94, 121, 248, 249, 477, 478, 223, 224, 225, 226, 227, 228, 95, 96, 97, 98, 99, 100, 101, 151, 250, 251, 479, 480, 481, 229, 230, 231, 232, 233, 234, 102, 103, 104, 105, 106, 107, 108, 152, 252, 253, 482, 483, 484, 485, 486, 235, 236, 237, 238, 239, 109, 110, 111, 112, 113, 114, 115, 153, 254, 255, 487, 488, 489, 490, 491, 492, 240, 116, 117, 118, 119, 120, 122, 123, 124, 125, 126, 256, 257, 493, 494, 495, 496, 497, 498, 127, 128, 129, 130, 131, 132, 133, 134, 135, 154, 258, 259, 499, 500, 501, 502, 503, 504, 421, 136, 137, 138, 139, 140, 141, 142, 143, 144, 155, 260, 261, 505, 506, 507, 508, 509, 510, 422, 145, 146, 147, 148, 149, 150, 156, 157, 158, 159, 262, 263, 423, 160, 161, 162, 163, 164, 165, 166, 167, 168, 181, 264, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 511]
        for i in range(N):
            # task_list = random.sample(range(1, len(data["scheduling_task"]) + 1), len(data["scheduling_task"]))
            # dec = list()
            # for j in task_list:
            #     dec += [k for k in range((j-1)*30+1, j*30+1)]
            # decs += [dec]
            decs += [list1]
        decs = np.array(decs)
        return Population(decs=decs)

    # def fix_decs(self, pop: Population):
    #     # 进行修复, 将开始时间早的往前排
    #     for i in range(len(pop)):
    #         pop.decs[i] = self.bubble_sort(pop.decs[i], self.task_start_time, self.index_task)
    #     return pop
    # def bubble_sort(self, arr, task_start_time: dict, index_task: dict):
    #     arr2 = arr.tolist()  # 转化成列表
    #     split_array = [arr2[i * 15: (i + 1) * 15] for i in range(len(arr2)//15)]
    #     for i in range(len(split_array)):
    #         for j in range(0, len(split_array)-i-1):
    #             if task_start_time[index_task[split_array[j][0]]] > task_start_time[index_task[split_array[j+1][0]]]:
    #                 temp = split_array[j]
    #                 split_array[j] = split_array[j+1]
    #                 split_array[j+1] = temp
    #     return np.array(sum(split_array, []))

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
        task_main_action_list = list(task_main_action.values())
        # 使用filter函数去除值为0的元素
        time_main_action2 = dict()
        for key, value in time_main_action.items():
            time_main_action2[str(key)] = value
        import json
        with open("file_jin.txt", "w") as file:
            json.dump(time_main_action2, file)
        task_main_action_list = list(filter(lambda x: x != 0, task_main_action_list))
        obj1 = self.n_var - sum(task_main_action_list)  # 统计总的主动作[最大化]->[600-(每个任务的主动作之和)]
        obj2 = 1-(sum(task_main_action_list)/30/len(task_main_action_list))  # 任务的总计完成度[最大化]->[1-每个任务的完成度之和的平均值]
        obj3 = next_times*40  # 总的切换时长[最小化]
        # obj2 = 0  # 任务的总计完成度[最大化]->[1-每个任务的完成度之和的平均值]
        # obj3 = 0  # 总的切换时长[最小化]
        return np.array([obj1, obj2, obj3]), f'{time_main_action}'
