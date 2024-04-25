import time
import psutil

import logger
import app

from .commons import AlgoEvent, AlgoResultType, AlgoState


def control_cb(self):
    self.update_exec_time()

    if self.state is AlgoState.RUNNING:
        self.reset_frame()
        res = self.post_processing()
        data = {"algoResults": res}
        app.sio.emit("node.input", data)
    elif self.state is AlgoState.PAUSING:
        self.pause_algo()
        self.reset_frame()
        res = self.post_processing()
        data = {"algoResults": res}
        app.sio.emit("node.input", data)
    elif self.state is AlgoState.STOPPING:
        self.stop_algo()
    elif self.state is AlgoState.COMPLETED:
        data = {"algoEvent": AlgoEvent.COMPLETE_ALGO.value}
        app.sio.emit("node.input", data)
    self.reset_iter_start_time()
    cpu_usage = psutil.cpu_percent()
    if cpu_usage > 50.0:
        logger.debug(f"{self.name} occupied >50% CPU.")
        time.sleep(1e-1)
    else:
        time.sleep(1e-6)


class AlgoFront:
    def __init__(self):
        self._evol_algo = None

    def init_static(self):
        app.sio.setStatic("/demos/statics/algo_sim_result_post")

    def set_evol_algo(self, evol_algo):
        self._evol_algo = evol_algo

    def _setData(self, context):
        # 前端设置数据
        logger.info(f"data.set:{context.message}")

        event = context.message.get("algoEvent")
        if event is not None:
            if AlgoEvent(event) is AlgoEvent.START_ALGO:
                self._evol_algo.set_start_flag(True)
                if self._evol_algo.state is not AlgoState.WAITING:
                    self._evol_algo.re_init()
                self._evol_algo.state = AlgoState.RUNNING
                logger.info("Algorithm starts.")
            elif AlgoEvent(event) is AlgoEvent.CONTINUE_ALGO:
                self._evol_algo.state = AlgoState.RUNNING
                logger.info("Algorithm continues running.")
            elif AlgoEvent(event) is AlgoEvent.PAUSE_ALGO:
                self._evol_algo.state = AlgoState.PAUSING
                logger.info("Algorithm is pausing.")
            elif AlgoEvent(event) is AlgoEvent.STOP_ALGO:
                self._evol_algo.set_start_flag(False)
                self._evol_algo.state = AlgoState.STOPPING
                logger.info("Algorithm is stopping.")

        data_type = context.message.get("dataType")
        if data_type is not None:
            self._evol_algo.set_data_type(AlgoResultType(data_type))
            if self._evol_algo.is_start():
                res = self._evol_algo.post_processing()
                data = {"algoResults": res}
                app.sio.emit("node.input", data)

    def _getData(self, context):
        # 前端获取数据
        logger.info(f"data.get:{context.message}")
        slidebar_value = context.message.get("slidebarValue")
        if slidebar_value is not None:
            self._evol_algo.set_frame_from_bar_value(slidebar_value)
            res = self._evol_algo.post_processing()
            return res

        export_current_frame_data = context.message.get(
            "exportCurrentFrameData"
        )
        if export_current_frame_data is not None:
            res = self._evol_algo.get_current_frame_data()
            return res

        export_gif_data = context.message.get("exportGifData")
        if export_gif_data is not None:
            res = self._evol_algo.get_gif_data()
            return res

        export_all_frames_data = context.message.get("exportAllFramesData")
        if export_all_frames_data is not None:
            res = self._evol_algo.get_all_frames_data()
            return res

        export_current_frame_best_population_data = context.message.get(
            "exportCurrentFrameBestPopulationData"
        )
        if export_current_frame_best_population_data is not None:
            res = self._evol_algo.get_current_frame_best_population_data()
            return res

        export_current_frame_non_dominant_data = context.message.get(
            "exportCurrentFrameNonDominantData"
        )
        if export_current_frame_non_dominant_data is not None:
            res = self._evol_algo.get_current_frame_non_dominant_data()
            return res

    def add_simulation_result(self, simulation_result):
        self._evol_algo.problem.add_simulation_result(simulation_result)


algo_front = AlgoFront()
# module = Module()


# @module.on("data.get")
# def getData(context):
#     return algo_front._getData(context)


# @module.on("data.set")
# def setData(context):
#     return algo_front._setData(context)


# app.modules.register("module", module)
