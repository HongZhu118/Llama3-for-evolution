from suanpan.node import node
from suanpan.app import app


def sim_req_cb(sim_param_dict):
    app.send({"out1": sim_param_dict}, args=node.outargs)
