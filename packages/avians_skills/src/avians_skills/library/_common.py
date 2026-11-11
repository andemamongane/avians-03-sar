"""Building blocks the skills share: transit (plan + climb + follow + replan),
landing on a pad, taking off, holding."""
from __future__ import annotations

import math

from shapely.geometry import LineString

from ..base import Context, Setpoint, SkillError
from ..follow import PathFollower


def hold(st) -> Setpoint:
    return Setpoint("position", (st.east, st.north, max(st.alt, 0.0)), max_speed=2.0)


class Transit:
    """Fly to (east, north) at `alt`: climb in place first if the target is
    higher (over obstacles before crossing them), then follow the planned
    path; a descent happens along the way. Replans when a zone closes across the rest of the
    path; raises SkillError if no path exists any more."""

    def __init__(self, ctx: Context, goal, alt: float, speed: float | None = None,
                 arrive_speed_tol: float = 0.8, tolerance: float = 1.5):
        self.goal, self.alt = (float(goal[0]), float(goal[1])), float(alt)
        self.speed = min(speed or ctx.limits.cruise_speed, ctx.limits.max_speed)
        self.tol = tolerance
        if not (ctx.limits.min_alt <= alt <= ctx.limits.max_alt):
            raise SkillError(f"altitude {alt:g} m is outside {ctx.limits.min_alt:g}-"
                             f"{ctx.limits.max_alt:g} m")
        self._plan(ctx)

    def _plan(self, ctx: Context) -> None:
        st = ctx.state
        self.unsafe = {z.id for z in ctx.unsafe_zones()}
        # planned at the altitude the leg is flown at (the altitude change happens in place first)
        path = ctx.planner.plan((st.east, st.north), self.goal, self.alt, ctx.unsafe_zones())
        pts = [(st.east, st.north, st.alt)]
        if self.alt - st.alt > 2.0:
            # climb in place first (an obstacle between the two altitudes could lie on the path);
            # a descent happens en route: the path is clear at the target altitude and the drone
            # stays above it (and a deconfliction override, which only raises, cannot stall it)
            pts.append((st.east, st.north, self.alt))
        pts += [(e, n, self.alt) for e, n in path[1:]]
        self.path2d = path
        self.follower = PathFollower(pts, self.speed, ctx.limits.max_accel, tolerance=self.tol)

    @property
    def length(self) -> float:
        return self.follower.length

    def remaining(self) -> float:
        return self.follower.remaining()

    def tick(self, ctx: Context) -> Setpoint:
        now = {z.id for z in ctx.unsafe_zones()}
        if now - self.unsafe:
            closed = [z for z in ctx.unsafe_zones() if z.id in now - self.unsafe]
            rest = LineString(self.path2d) if len(self.path2d) > 1 else None
            if rest is not None and any(rest.intersects(z.shape.buffer(ctx.planner.margin))
                                        for z in closed):
                self._plan(ctx)                  # SkillError propagates: no way through
                ctx.emit("replanned", f"replanned around {', '.join(z.id for z in closed)}",
                         wake=False)
            self.unsafe = now
        carrot, vel, v = self.follower.step(ctx.state.pos)
        return Setpoint("position", carrot, vel, max_speed=v)

    def done(self, ctx: Context) -> bool:
        st = ctx.state
        return self.follower.done(st.pos, st.speed3)

    def eta_s(self) -> float:
        return self.remaining() / max(self.speed, 0.1) + 2.0


class TakeOff:
    def __init__(self, ctx: Context, alt: float):
        st = ctx.state
        self.target = (st.east, st.north, float(alt))

    def tick(self, ctx: Context) -> Setpoint:
        return Setpoint("position", self.target, max_speed=ctx.limits.max_climb)

    def done(self, ctx: Context) -> bool:
        st = ctx.state
        return abs(st.alt - self.target[2]) < 0.6 and st.speed3 < 0.6


class LandAt:
    """Transit to a pad at approach altitude, descend over it, land."""

    def __init__(self, ctx: Context, pad, approach_alt: float | None = None):
        self.pad = (float(pad[0]), float(pad[1]))
        alt = approach_alt or ctx.limits.cruise_alt
        st = ctx.state
        self.transit = None if (st.landed and math.dist((st.east, st.north), self.pad) < 2.0) \
            else Transit(ctx, self.pad, max(alt, ctx.limits.min_alt), tolerance=1.0)
        self.phase = "transit" if self.transit else "landed"

    def tick(self, ctx: Context) -> Setpoint:
        if self.phase == "transit":
            sp = self.transit.tick(ctx)
            if self.transit.done(ctx):
                self.phase = "land"
            return sp
        if self.phase == "land":
            if ctx.state.landed:
                self.phase = "landed"
            return Setpoint("land", (self.pad[0], self.pad[1], 0.0))
        return Setpoint("idle")

    def done(self, ctx: Context) -> bool:
        return self.phase == "landed"
