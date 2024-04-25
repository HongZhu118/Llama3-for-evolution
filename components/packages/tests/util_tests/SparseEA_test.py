"""开发者可参考本案例开发搜索组件入口程序Python代码
需要注意：

初始化参数
1. 算法参数（options）

启动搜索算法的参数
1. 优化问题（optimization_problem）

搜索算法运行时：
1. 搜索算法输出参数，需要组件send给仿真接口组件的逻辑在回调函数中实现：
    sim_req_cb
2. 仿真接口组件返回仿真结果（即本组件输入），
    仿真结果发给优化算法:search_alg.add_simulation_result(simulation_result)
"""
import sys
import os
from queue import Queue # noqa

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append("./components/packages")

from components.packages import platgo as pg  # noqa
from components.packages.problems.optimization_problems import *  # noqa
from components.packages.common.commons import (AlgoMode,ConvertionProbSim,SimParams)  # noqa


input_prefix = SimParams.InputPrefix
objective_prefix = SimParams.ObjectivePrefix
constraint_prefix = SimParams.ConstraintPrefix


def send(data):
    """模拟组件输出app.send函数

    Args:
        data (dict): 输出参数值
    """
    output_queue.put(data)


def sim_req_cb(simulation_parameter_dict):
    """算法请求执行仿真的回调函数，
    TODO: 开发者需要将参数使用组件app.send方法输出

    Args:
        simulation_parameter_dict (dict): 仿真输入参数
    """
    send(simulation_parameter_dict)
    return


def gen_engineer_opt_prob(
    order: int,
    lower_bound: float,
    upper_bound: float,
    objective_option: str,
) -> dict:

    inputs = [
        {
            "param": input_prefix + str(i + 1),
            "min": lower_bound,
            "max": upper_bound,
        }
        for i in range(order)
    ]
    objectives = [objective_prefix + str(i + 1) for i in range(n_obj)]
    constraints = [constraint_prefix + str(i + 1) for i in range(n_constr)]
    outputs = objectives + constraints

    optimization_problem = {
        "inputs": inputs,
        "outputs": outputs,
        # variables需要定制化
        # "variables": [
        #     {"param": "Surface", "formula": "2*(x1*x2+x2*x3+x1*x3)"}
        # ],
        "objFcn": [
            {"objective": objective, "option": objective_option}
            for objective in objectives
        ],
        "conFcn": [{"function": con} for con in constraints],
    }

    return optimization_problem


def run_algo(pop_size, options, optimization_problem, sim_req_cb, algo_mode):
    global evol_algo
    # 算法线程
    evol_algo = pg.algorithms.SparseEA(
        pop_size,
        options,
        optimization_problem,
        None,
        sim_req_cb=sim_req_cb,
        name="SparseEA-Thread",
        show_bar=True,
        debug=True,
    )
    evol_algo.start()


def run_sim(output_queue, evol_algo, test_func):
    # 主进程: 模拟外部输入

    while True:
        sim_params = output_queue.get()
        if sim_params == {}:
            break

        design_id = sim_params.pop(SimParams.DesignId)
        x = ConvertionProbSim.simparams2x(sim_params)
        prob_res = test_func(x)
        sim_res = ConvertionProbSim.probres2simres(prob_res)

        sim_res[SimParams.DesignId] = design_id
        # 模拟仿真接口数据传输和计算消耗的时间
        # time.sleep(0.001)
        evol_algo.problem.add_simulation_result(sim_res)


if __name__ == "__main__":
    # 仿真输入的队列
    output_queue = Queue()

    # 算法模式：学术或工程
    algo_mode = AlgoMode.ACADEMIC
    # algo_mode = AlgoMode.ENGINEERING

    if algo_mode is AlgoMode.ACADEMIC:
        # 内置问题
        optimization_problem = {"name": "SMOP8", "n_var": 100}
        # 自定义问题
        # optimization_problem = {
        #     "name": "custom_problem",
        #     "encoding": "real",
        #     "n_var": 30,
        #     "lower": 0,
        #     "upper": 1,
        #     "initFcn": [],
        #     "decFcn": [],
        #     "objFcn": [
        #         "x[0]",
        #         "(1+9*mean(x[1:]))*(1-sqrt(x[0]/1+9*mean(x[1:])))"],
        #     "conFcn": []
        # }
    else:  # algo_mode is AlgoMode.ENGINEERING:
        order = 30
        lower_bound = 0
        upper_bound = 1
        n_obj = 2
        # objective_option = "MaximizeValue"
        objective_option = "MinimizeValue"
        n_constr = 1
        optimization_problem = gen_engineer_opt_prob(
            order, lower_bound, upper_bound, objective_option
        )
    print(optimization_problem)

    evol_algo = None
    test_func = SMOP1  # noqa: F405
    pop_size = 100
    options = {}
    run_algo(pop_size, options, optimization_problem, sim_req_cb, algo_mode)
    run_sim(output_queue, evol_algo, test_func)

    print("Done")
