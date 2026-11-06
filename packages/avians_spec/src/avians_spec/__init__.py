"""avians_spec: what the mission must do, and how well a flight did it.

    reg = sar_registry(world)
    spec = MissionSpec.from_dict(mission["spec"], reg, world.regions)
    spec.score(trace)               # finished flight: robustness per clause
    OnlineMonitor(spec, max_speed)  # during the flight: ROSI verdicts
"""
from . import stl
from .monitor import OnlineMonitor
from .signals import Registry, Signal, sar_registry
from .spec import Clause, MissionSpec
from .stl import STLSyntaxError, Trace, parse

__all__ = ["Clause", "MissionSpec", "OnlineMonitor", "Registry", "STLSyntaxError", "Signal",
           "Trace", "parse", "sar_registry", "stl"]
