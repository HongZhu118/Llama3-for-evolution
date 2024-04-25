import os
import json

from suanpan.app.modules.base import Module
from suanpan.log import logger
from suanpan.app import app
from suanpan.storage import storage
from suanpan import g

from ..platgo.Problem import Problem, EngineeringProblem
from ..common.commons import AlgoMode
from ..common.engineering_mode import sim_req_cb


class ProbFront:
    def __init__(self):
        self._requirements = None
        self._opt_prob_filename = "optimizationProblem.json"
        self.optimization_problem = {}

    def download_opt_prob(self):
        storage.download(storage.getKeyInNodeConfigsStore(
            self._opt_prob_filename), self._opt_prob_filename, quiet=False)
        if os.path.exists(self._opt_prob_filename):
            with open(self._opt_prob_filename, "r") as f:
                self.optimization_problem = json.load(f)

    def upload_opt_prob(self, optimization_problem):
        with open(self._opt_prob_filename, "w") as f:
            json.dump(optimization_problem, f)
        storage.upload(storage.getKeyInNodeConfigsStore(
            self._opt_prob_filename), self._opt_prob_filename, quiet=False)

    def init_static(self):
        app.sio.setStatic("/demos/statics/prob_front")

    def set_requirements(self, requirements):
        self._requirements = requirements

    def _setData(self, context):
        # 前端设置数据
        logger.info(f"data.set:{context.message}")

        optimization_problem = context.message.get("probParams")
        if optimization_problem is not None:
            algo_mode = AlgoMode(optimization_problem.get("mode", 0))
            if algo_mode is AlgoMode.ACADEMIC:
                try:
                    problem = Problem(optimization_problem)
                    problem.validate()
                    optimization_problem.update(
                        {"requirements": self._requirements})
                    self.upload_opt_prob(optimization_problem)
                    app.send(optimization_problem)
                    self.optimization_problem = optimization_problem
                except Exception as err:
                    return str(err)
            else:
                try:
                    EngineeringProblem(
                        optimization_problem, sim_req_cb=sim_req_cb)
                    optimization_problem.update(
                        {"requirements": self._requirements})
                    self.upload_opt_prob(optimization_problem)
                    app.send(optimization_problem)
                    self.optimization_problem = optimization_problem
                except Exception as err:
                    return str(err)

    def _getData(self, context):
        # 前端获取数据
        logger.info(f"data.get:{context.message}")

        refresh = context.message.get("refresh")
        if refresh is not None:
            return {"probStatus": self.optimization_problem}


prob_front = ProbFront()
module = Module()


@module.on("oss.config.get")
def getOssConfig(context):
    return {"ossType": storage.type, "id": g.appId, "userId": g.userId}


@module.on("data.get")
def getData(context):
    return prob_front._getData(context)


@module.on("data.set")
def setData(context):
    return prob_front._setData(context)


app.modules.register("module", module)
