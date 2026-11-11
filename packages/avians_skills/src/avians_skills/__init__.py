"""avians_skills: the high-level tools the pilot calls.

    from avians_skills import create, tools
    skill = create("search_region", {"region": "r3"})
    skill.start(ctx); sp = skill.tick(ctx, dt)

Tool sets: BENCHMARK_TOOLS is what the single-drone pilot gets.
"""
from . import library  # noqa: F401  (registers the skills)
from .base import (REGISTRY, Context, Event, Limits, Setpoint, Skill, SkillError, Vehicle,
                   VehicleState, create, get, tools)
from .coverage import CoverageMap
from .energy import BatteryModel
from .planning import PathPlanner

SWARM_ONLY = ("handoff", "formation", "assign_tasks", "claim_region")
STUDENT_ONLY = ("escalate",)
BENCHMARK_TOOLS = sorted(n for n in REGISTRY if n not in SWARM_ONLY + STUDENT_ONLY)
SWARM_TOOLS = sorted(n for n in REGISTRY if n not in STUDENT_ONLY)

__all__ = ["BENCHMARK_TOOLS", "SWARM_TOOLS", "BatteryModel", "Context", "CoverageMap", "Event", "Limits",
           "PathPlanner", "REGISTRY", "STUDENT_ONLY", "SWARM_ONLY", "Setpoint", "Skill",
           "SkillError", "Vehicle", "VehicleState", "create", "get", "tools"]
