from .multi_objective.Real_world_MOPs.Sparse_SR import Sparse_SR  # noqa: F401
from .multi_objective.Real_world_MOPs.Sparse_CD import Sparse_CD  # noqa: F401
from .multi_objective.selfDefineProblem1 import selfDefineProblem1
from .multi_objective.selfDefineProblem2 import selfDefineProblem2  # 无角度
from .multi_objective.selfDefineProblem3 import selfDefineProblem3  # 有角度
from .multi_objective.selfDefineProblem4 import selfDefineProblem4  # 折线
from .single_objective.xugong.transmission import (  # noqa: F401
    XugongTransmission,
)  # noqa: F401
from .single_objective.xugong.comprehensive import (  # noqa: F401
    XugongComprehensive,
)  # noqa: F401
from .single_objective.real_world_SOPs.SOP_F1 import SOP_F1
from .single_objective.real_world_SOPs.SOP_F20 import SOP_F20
from .single_objective.satellkite.csp2 import SatellkiteCSP2
from .single_objective.real_world_SOPs.TSP import TSP
from .single_objective.real_world_SOPs.SOP_F12 import SOP_F12
from .single_objective.satellkite.rcpsp import SatellkiteRCPSP
from .single_objective.satellkite.csp import SatellkiteCSP
from .single_objective.satellkite.rcpsp2 import SatellkiteRCPSP2
from .multi_objective.daily_planning2_strategy import Daily_Planning2_Strategy
from .multi_objective.daily_planning import Daily_Planning
from .multi_objective.Real_world_MOPs.SMOP8 import SMOP8