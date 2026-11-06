"""Multi-drone skills (after the single-drone benchmark).

assign_tasks() is ready: optimal one-to-one assignment by the Hungarian
method (Kuhn 1955; scipy's linear_sum_assignment), the baseline the auction
methods (CBBA, Choi et al. 2009) are compared with. The skills need a swarm
backend (avians_sim with several vehicles and state sharing) and refuse
until there is one.
"""
from __future__ import annotations

import numpy as np

from ..base import Skill, SkillError


def assign_tasks(cost: np.ndarray) -> list[tuple[int, int]]:
    """(drone, task) pairs minimising total cost; cost[i, j] = drone i doing task j."""
    from scipy.optimize import linear_sum_assignment
    rows, cols = linear_sum_assignment(np.asarray(cost, float))
    return list(zip(rows.tolist(), cols.tolist()))


class _SwarmOnly(Skill):
    category = "swarm"

    def start(self, ctx):
        if getattr(ctx.actuators, "swarm", None) is None:
            raise SkillError(f"{self.name} needs other drones; this mission has one")


class HandOff(_SwarmOnly):
    name = "handoff"
    summary = "Hand the current task (e.g. tracking the person) to another drone."
    params = {"to": {"type": "string", "description": "drone id"}}
    required = ["to"]


class Formation(_SwarmOnly):
    name = "formation"
    summary = "Hold a position relative to another drone (e.g. as a radio relay)."
    params = {"leader": {"type": "string"}, "role": {"type": "string", "enum": ["relay", "wing"]}}
    required = ["leader", "role"]


class AssignTasks(_SwarmOnly):
    name = "assign_tasks"
    summary = "Split tasks (search regions, delivery, tracking) between the drones, optimally."
    params = {"tasks": {"type": "array", "items": {"type": "string"}}}
    required = ["tasks"]
