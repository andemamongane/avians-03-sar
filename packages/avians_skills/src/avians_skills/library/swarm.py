"""Multi-drone skills. They need other drones (ctx.actuators.swarm, the
Shared world of avians_sim) and refuse politely without them.

    claim_region   say which region you are searching; the others see it
    assign_tasks   split regions between the drones, optimally: the Hungarian
                   method (Kuhn 1955; scipy linear_sum_assignment) on flight
                   distance; writes the claims and tells each drone by radio
    handoff        ask another drone to take over a task (a radio tasking it
                   decides on)
    formation      hold a position relative to another drone (a relay above
                   and behind it, or a wing beside it) until replaced
"""
from __future__ import annotations

import math

import numpy as np

from ..base import Setpoint, Skill, SkillError

NUM = {"type": "number"}


def assign_tasks(cost: np.ndarray) -> list[tuple[int, int]]:
    """(drone, task) pairs minimising total cost; cost[i, j] = drone i doing task j."""
    from scipy.optimize import linear_sum_assignment
    rows, cols = linear_sum_assignment(np.asarray(cost, float))
    return list(zip(rows.tolist(), cols.tolist()))


def _swarm(ctx):
    sw = getattr(ctx.actuators, "swarm", None)
    if sw is None:
        raise SkillError("this mission has one drone: no other drone to coordinate with")
    return sw


class ClaimRegion(Skill):
    name, category, instant = "claim_region", "swarm", True
    summary = "Claim a search region as yours (the other drones see claims); release with region ''."
    params = {"region": {"type": "string", "description": "a search region id, or '' to release yours"}}
    required = ["region"]

    def start(self, ctx):
        sw = _swarm(ctx)
        me = ctx.drone_id
        if not self.args["region"]:
            for r in [r for r, d in sw.claims.items() if d == me]:
                del sw.claims[r]
            self.succeed(ctx, "claims released")
            return
        r = self.args["region"]
        if r not in ctx.world.regions:
            raise SkillError(f"no region {r!r}")
        owner = sw.claims.get(r)
        if owner and owner != me:
            raise SkillError(f"{r} is claimed by {owner}; pick another or ask them to hand it off")
        for old in [x for x, d in sw.claims.items() if d == me]:
            del sw.claims[old]
        sw.claims[r] = me
        self.succeed(ctx, f"{r} claimed by {me}")


class AssignTasks(Skill):
    name, category, instant = "assign_tasks", "swarm", True
    summary = ("Split search regions between all drones at minimum total flight distance (Hungarian "
               "method); records the claims and radios each drone its region.")
    params = {"regions": {"type": "array", "items": {"type": "string"},
                          "description": "regions to split (default: all search regions not yet searched)"}}

    def start(self, ctx):
        sw = _swarm(ctx)
        regions = self.args.get("regions") or [r.id for r in ctx.world.search_regions()
                                                if ctx.coverage.fraction(r.id) < 0.5]
        regions = [r for r in regions if r in ctx.world.regions]
        if not regions:
            raise SkillError("no regions to assign")
        drones = [ctx.drone_id] + [o["id"] for o in sw.others(ctx.drone_id)]
        pos = {ctx.drone_id: (ctx.state.east, ctx.state.north)}
        pos.update({o["id"]: (o["east"], o["north"]) for o in sw.others(ctx.drone_id)})
        cost = np.array([[math.dist(pos[d], ctx.world.regions[r].centroid) for r in regions] for d in drones])
        pairs = assign_tasks(cost)
        text = []
        for i, j in pairs:
            d, r = drones[i], regions[j]
            for old in [x for x, who in sw.claims.items() if who == d]:
                del sw.claims[old]
            sw.claims[r] = d
            text.append(f"{d}: {r}")
            if d != ctx.drone_id:
                sw.ground.push(ctx.t, ctx.drone_id, "tasking", f"{ctx.drone_id} assigned you {r}", to=d,
                               region=r)
        left = [regions[j] for j in range(len(regions)) if j not in {j for _, j in pairs}]
        self.succeed(ctx, "assigned " + ", ".join(text) + (f"; unassigned: {', '.join(left)}" if left else ""))


class HandOff(Skill):
    name, category, instant = "handoff", "swarm", True
    summary = "Ask another drone, by radio, to take over a task (e.g. tracking the person); it decides."
    params = {"to": {"type": "string", "description": "drone id"},
              "task": {"type": "string", "description": "what to take over, e.g. 'track the person' or 'search r2'"}}
    required = ["to", "task"]

    def start(self, ctx):
        sw = _swarm(ctx)
        ids = [o["id"] for o in sw.others(ctx.drone_id)]
        if self.args["to"] not in ids:
            raise SkillError(f"no drone {self.args['to']!r} (others: {', '.join(ids)})")
        sw.ground.push(ctx.t, ctx.drone_id, "tasking", f"{ctx.drone_id} asks you to {self.args['task']}",
                       to=self.args["to"])
        self.succeed(ctx, f"asked {self.args['to']} to {self.args['task']}")


class Formation(Skill):
    name, category = "formation", "swarm"
    summary = ("Hold a position relative to another drone until replaced: relay (20 m behind and 10 m "
               "above it, a radio relay) or wing (20 m beside it).")
    params = {"leader": {"type": "string", "description": "drone id"},
              "role": {"type": "string", "enum": ["relay", "wing"], "default": "wing"}}
    required = ["leader"]

    def start(self, ctx):
        sw = _swarm(ctx)
        if self.args["leader"] not in [o["id"] for o in sw.others(ctx.drone_id)]:
            raise SkillError(f"no drone {self.args['leader']!r}")
        self.sw = sw

    def tick(self, ctx, dt):
        lead = next((o for o in self.sw.others(ctx.drone_id) if o["id"] == self.args["leader"]), None)
        if lead is None:
            self.fail(ctx, "lost the leader")
            return None
        lv = self.sw.drones[lead["id"]].vehicle.state()
        h = lv.yaw
        if self.args["role"] == "relay":
            off = (-20 * math.cos(h), -20 * math.sin(h), 10.0)
        else:
            off = (20 * math.cos(h + math.pi / 2), 20 * math.sin(h + math.pi / 2), 0.0)
        tgt = (lv.east + off[0], lv.north + off[1], max(ctx.limits.min_alt, lv.alt + off[2]))
        return Setpoint("position", tgt, (lv.ve, lv.vn, 0.0), max_speed=ctx.limits.max_speed)

    def progress(self, ctx):
        return f"{self.args['role']} on {self.args['leader']}"
