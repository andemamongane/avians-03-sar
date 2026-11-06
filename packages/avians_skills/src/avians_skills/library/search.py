"""Searching: cover a region with a standard pattern; look closer at a candidate."""
from __future__ import annotations

import math

from shapely.geometry import LineString, Point

from ..base import Setpoint, Skill, SkillError
from ..coverage import expanding_square, lawnmower, sector
from ..follow import PathFollower
from ._common import Transit

NUM = {"type": "number"}


class SearchRegion(Skill):
    name, category = "search_region", "search"
    summary = ("Search a region with a standard pattern: lawnmower (parallel tracks over the whole "
               "region), expanding_square or sector (around the region centre or a datum). Lane "
               "spacing comes from the camera footprint at the altitude. Finishes when the "
               "pattern is flown; a sighting does not stop it (you decide).")
    params = {"region": {"type": "string", "description": "a search region id"},
              "pattern": {"type": "string", "enum": ["lawnmower", "expanding_square", "sector"],
                          "default": "lawnmower"},
              "alt": {**NUM, "description": "m (lower: better detection, narrower lanes)", "default": 30.0},
              "overlap": {**NUM, "description": "lane overlap fraction", "default": 0.2},
              "speed": {**NUM, "description": "m/s (default: cruise speed)"},
              "datum_east": {**NUM, "description": "pattern centre for square/sector"},
              "datum_north": NUM}
    required = ["region"]

    def start(self, ctx):
        a = self.args
        r = ctx.world.regions.get(a["region"])
        if r is None or r.kind != "search":
            raise SkillError(f"{a['region']!r} is not a search region "
                             f"(search regions: {', '.join(x.id for x in ctx.world.search_regions())})")
        self.region, alt = r, a["alt"]
        if not 0 <= a["overlap"] < 0.9:
            raise SkillError("overlap must be in [0, 0.9)")
        fp = ctx.camera.footprint(alt)
        if fp < 3:
            raise SkillError(f"at {alt:g} m the camera sees only {fp:.1f} m around: fly higher")
        self.spacing = 2 * fp * (1 - a["overlap"])
        # 1 m beyond the planner's own margins, so lane ends are valid goals for it
        forbidden = [p.buffer(1.0, quad_segs=2) for _, p in ctx.planner.forbidden(alt, ctx.unsafe_zones())]
        st = ctx.state
        if a["pattern"] == "lawnmower":
            pts = lawnmower(r.shape, self.spacing, (st.east, st.north), forbidden)
        else:
            datum = (a.get("datum_east"), a.get("datum_north"))
            datum = r.centroid if None in datum else datum
            w, h = r.size
            pts = (expanding_square(datum, self.spacing, max(w, h) / 2) if a["pattern"] ==
                   "expanding_square" else sector(datum, max(w, h) / 2))
            pts = [p for p in pts if not any(f.contains(Point(p)) for f in forbidden)]
        if len(pts) < 2:
            raise SkillError(f"nothing of {r.id} is searchable at {alt:g} m right now "
                             f"(closed zones or obstacles cover it)")
        # detour between consecutive pattern points where the straight line is forbidden
        route = [pts[0]]
        for p, q in zip(pts, pts[1:]):
            if any(LineString([p, q]).intersects(f) for f in forbidden):
                route += ctx.planner.plan(p, q, alt, ctx.unsafe_zones())[1:]
            else:
                route.append(q)
        self.route = route
        self.speed = a.get("speed") or ctx.limits.cruise_speed
        self.transit = Transit(ctx, route[0], alt)
        self.follower = None
        self.alt = alt
        self.unsafe = {z.id for z in ctx.unsafe_zones()}
        self.reported = 0
        self.t_start = ctx.t

    def tick(self, ctx, dt):
        now = {z.id for z in ctx.unsafe_zones()}
        if now - self.unsafe:
            closed = [z for z in ctx.unsafe_zones() if z.id in now - self.unsafe]
            if any(z.shape.intersects(self.region.shape) for z in closed):
                self.fail(ctx, f"{', '.join(z.id for z in closed)} closed over part of "
                          f"{self.region.id} ({self.covered(ctx):.0%} covered); search the rest again")
                return None
            self.unsafe = now
        if self.follower is None:
            sp = self.transit.tick(ctx)
            if self.transit.done(ctx) or self.transit.remaining() < 3.0:
                pts = [(e, n, self.alt) for e, n in self.route]
                self.follower = PathFollower(pts, self.speed, ctx.limits.max_accel)
            return sp
        carrot, vel, v = self.follower.step(ctx.state.pos)
        q = int(self.follower.fraction() * 4)
        if q > self.reported and q < 4:
            self.reported = q
            ctx.emit("progress", f"search of {self.region.id}: {q * 25}% of the pattern flown, "
                     f"{self.covered(ctx):.0%} of the region seen", wake=False)
        if self.follower.done(ctx.state.pos, ctx.state.speed3):
            self.succeed(ctx, f"pattern over {self.region.id} flown in {ctx.t - self.t_start:.0f} s; "
                         f"{self.covered(ctx):.0%} of the region seen", region=self.region.id)
        return Setpoint("position", carrot, vel, max_speed=v)

    def covered(self, ctx) -> float:
        return ctx.coverage.fraction(self.region.id) if ctx.coverage else 0.0

    def progress(self, ctx):
        f = self.follower.fraction() if self.follower else 0.0
        return f"{f:.0%} of the pattern flown, {self.covered(ctx):.0%} of {self.region.id} seen"

    def eta_s(self, ctx):
        if self.follower:
            rest = self.follower.remaining()
        else:
            rest = self.transit.remaining() + sum(math.dist(p, q) for p, q in
                                                  zip(self.route, self.route[1:]))
        return rest / max(self.speed, 0.1) * 1.15


class Inspect(Skill):
    name, category = "inspect", "search"
    summary = ("Fly low over the person estimate (or a point) and hover there to look closely: "
               "raises confidence in a real person, lowers it for a decoy.")
    params = {"east": {**NUM, "description": "default: the person estimate"}, "north": NUM,
              "alt": {**NUM, "description": "m", "default": 15.0},
              "seconds": {**NUM, "description": "how long to look", "default": 8.0}}

    def start(self, ctx):
        a = self.args
        if a.get("east") is None or a.get("north") is None:
            if ctx.belief is None:
                raise SkillError("no estimate to inspect: give east and north")
            b = ctx.belief()
            self.at = (b.east, b.north)
            self.what = "the estimate" if b.source != "prior" else "the prior mean"
        else:
            self.at, self.what = (a["east"], a["north"]), "the point"
        self.transit = Transit(ctx, self.at, a["alt"])
        self.t_over = None
        self.conf0 = ctx.belief().conf if ctx.belief else 0.0

    def tick(self, ctx, dt):
        if self.t_over is None:
            sp = self.transit.tick(ctx)
            if self.transit.done(ctx):
                self.t_over = ctx.t
            return sp
        if ctx.t - self.t_over >= self.args["seconds"]:
            b = ctx.belief() if ctx.belief else None
            msg = (f"looked at {self.what} for {self.args['seconds']:.0f} s: confidence "
                   f"{self.conf0:.2f} -> {b.conf:.2f}" if b else "looked")
            self.succeed(ctx, msg)
        return Setpoint("position", (self.at[0], self.at[1], self.args["alt"]), max_speed=2.0)
