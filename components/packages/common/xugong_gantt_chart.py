from suanpan.app.modules.base import Module
from suanpan.log import logger
from suanpan.app import app


class XugongGanttChart:
    def __init__(self):
        self.data = None

    def init_static(self):
        app.sio.setStatic("/demos/statics/xugong_gantt_chart")

    def _setData(self, context):
        # 前端设置数据
        logger.info(f"data.set:{context.message}")

        scheduling_data = context.message

        app.send({"outputData1": scheduling_data})

    def _getData(self, context):
        # 前端获取数据
        logger.info(f"data.get:{context.message}")

        refresh = context.message.get("refresh")
        if refresh is not None:
            return self.data

    def send(self, data):
        self.data = data
        app.sio.emit("node.input", data)


xugong_gantt_chart = XugongGanttChart()
module = Module()


@module.on("data.get")
def getData(context):
    return xugong_gantt_chart._getData(context)


@module.on("data.set")
def setData(context):
    return xugong_gantt_chart._setData(context)


app.modules.register("module", module)
