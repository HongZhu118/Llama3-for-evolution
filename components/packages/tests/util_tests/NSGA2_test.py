"""
学术模式，测试内置问题
"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append("./components/packages")

from components.packages import platgo as pg  # noqa


if __name__ == "__main__":
    # 内置问题
    optimization_problem = {
        "name": "Sparse_SR",
        "lenSig": 1027,
        "lenObs": 481,
        "sparsity": 261,
        "sigma": 0.1,
    }
    print(optimization_problem)

    pop_size = 100
    max_fe = 10000
    options = {}

    def run_algo():
        evol_algo = pg.algorithms.NSGA2(
            pop_size=pop_size,
            options=options,
            optimization_problem=optimization_problem,
            control_cb=None,
            max_fe=max_fe,
            name="NSGA2-Thread",
            debug=True,
        )
        evol_algo.start()

    run_algo()
    print("Done")
