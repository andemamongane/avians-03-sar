"""Routes: fly a list of waypoints, scan a line, and plan without flying."""
from __future__ import annotations

import math

from ..base import Setpoint, Skill, SkillError
from ..follow import PathFollower
from ._common import Transit

NUM = {"type": "number"}
WAYPOINT = {"type": "object", "properties": {"east": NUM, "north": NUM, "alt": NUM},
            "required": ["east", "north"], "additionalProperties": False}


class FlyWaypoints(Skill):
    name, category = "fly_waypoints", "movement"
    summary = ("Fly through a list of waypoints in order (each with optional alt), planning around "
               "no-fly zones, closed zones and obstacles between them; stops at the last one.")
    params = {"waypoints": {"type": "array", "items": WAYPOINT, "minItems": 1},
              "speed": {**NUM, "description": "m/s (default: cruise speed)"}}
    required = ["waypoints"]

    def start(self, ctx):
        wps = self.args["waypoints"]
        if not isinstance(wps, list) or not wps:
            raise SkillError("waypoints must be a non-empty list of {east, north[, alt]}")
        lim, st = ctx.limits, ctx.state
        pts = [(st.east, st.north, st.alt)]
        prev = (st.east, st.north)
        self.route = []
        for k, w in enumerate(wps):
            try:
                e, n = float(w["east"]), float(w["north"])
            except (KeyError, TypeError, ValueError):
                raise SkillError(f"waypoint {k} needs numeric east and north") from None
            alt = float(w.get("alt", lim.cruise_alt))
            if not lim.min_alt <= alt <= lim.max_alt:
                raise SkillError(f"waypoint {k}: altitude {alt:g} m is outside {lim.min_alt:g}-{lim.max_alt:g} m")
            why = ctx.planner.blocked_by(e, n, alt, ctx.unsafe_zones())
            if why:
                raise SkillError(f"waypoint {k} ({e:.0f}, {n:.0f}) is inside {why}")
            leg = ctx.planner.plan(prev, (e, n), alt, ctx.unsafe_zones())
            pts += [(x, y, alt) for x, y in leg[1:]]
            self.route.append((e, n, alt))
            prev = (e, n)
        self.speed = self.args.get("speed") or lim.cruise_speed
        self.follower = PathFollower(pts, self.speed, lim.max_accel)
        self.n = len(wps)

    def tick(self, ctx, dt):
        carrot, vel, v = self.follower.step(ctx.state.pos)
        if self.follower.done(ctx.state.pos, ctx.state.speed3):
            e, n, _ = self.route[-1]
            self.succeed(ctx, f"{self.n} waypoint(s) flown; at ({e:.0f}, {n:.0f})")
        return Setpoint("position", carrot, vel, max_speed=v)

    def progress(self, ctx):
        return f"{self.follower.fraction():.0%} of the route, {self.follower.remaining():.0f} m to go"

    def eta_s(self, ctx):
        return self.follower.remaining() / max(self.speed, 0.1) * 1.15


class Scan(Skill):
    name, category = "scan", "search"
    summary = ("Scan a straight line (a road, a creek, a ridge) at low altitude and slow speed with "
               "the camera: from (east, north) to (east2, north2).")
    params = {"east": NUM, "north": NUM, "east2": NUM, "north2": NUM,
              "alt": {**NUM, "description": "m", "default": 15.0},
              "speed": {**NUM, "description": "m/s", "default": 4.0}}
    required = ["east", "north", "east2", "north2"]

    def start(self, ctx):
        a = self.args
        p, q = (a["east"], a["north"]), (a["east2"], a["north2"])
        for name, pt in (("start", p), ("end", q)):
            why = ctx.planner.blocked_by(pt[0], pt[1], a["alt"], ctx.unsafe_zones())
            if why:
                raise SkillError(f"the scan {name} is inside {why}")
        self.q, self.alt = q, a["alt"]
        self.transit = Transit(ctx, p, a["alt"])
        self.follower = None
        self.length = math.dist(p, q)

    def tick(self, ctx, dt):
        if self.follower is None:
            sp = self.transit.tick(ctx)
            if self.transit.done(ctx):
                st = ctx.state
                self.follower = PathFollower([st.pos, (self.q[0], self.q[1], self.alt)], self.args["speed"],
                                             ctx.limits.max_accel)
            return sp
        carrot, vel, v = self.follower.step(ctx.state.pos)
        if self.follower.done(ctx.state.pos, ctx.state.speed3):
            b = ctx.belief() if ctx.belief else None
            self.succeed(ctx, f"scanned {self.length:.0f} m" + (f"; best candidate confidence {b.conf:.2f}"
                                                                 if b and b.source != "prior" else "; nothing found"))
        return Setpoint("position", carrot, vel, max_speed=v)

    def eta_s(self, ctx):
        return (self.transit.eta_s() if self.follower is None else 0.0) + self.length / self.args["speed"]


class PathPlan(Skill):
    name, category, instant = "path_plan", "movement", True
    summary = ("Plan a route to a point or place without flying it: returns its length, time and "
               "battery cost, and whether anything blocks it. Free: use it to compare options.")
    params = {"east": NUM, "north": NUM,
              "place": {"type": "string", "description": "a station, depot, region id, or 'launch'"},
              "alt": {**NUM, "description": "m (default: cruise altitude)"}}

    def start(self, ctx):
        a = self.args
        if a.get("place"):
            try:
                goal = ctx.place(a["place"])
            except KeyError:
                raise SkillError(f"no place named {a['place']!r}") from None
        elif a.get("east") is not None and a.get("north") is not None:
            goal = (float(a["east"]), float(a["north"]))
        else:
            raise SkillError("give east and north, or a place")
        lim, st = ctx.limits, ctx.state
        alt = a.get("alt") or lim.cruise_alt
        path = ctx.planner.plan((st.east, st.north), goal, alt, ctx.unsafe_zones())
        length = ctx.planner.length(path)
        t = length / lim.cruise_speed + abs(alt - st.alt) / lim.max_climb + 4
        cost = ctx.energy.cost(length, lim.cruise_speed, st.payload, climb_m=max(0.0, alt - st.alt)) if ctx.energy else 0.0
        via = "direct" if len(path) == 2 else f"via {len(path) - 2} turn(s)"
        self.succeed(ctx, f"route to ({goal[0]:.0f}, {goal[1]:.0f}): {length:.0f} m {via}, about {t:.0f} s, "
                          f"~{cost:.1f} % battery; {st.battery - cost:.0f} % would remain")
