"""The medical kit: pick it up at a depot, deliver it to the person."""
from __future__ import annotations

import math


from ..ballistics import fall_time, release_point
from ..base import Setpoint, Skill, SkillError
from ._common import LandAt, TakeOff, Transit

NUM = {"type": "number"}


class PickUp(Skill):
    name, category = "pick_up", "payload"
    summary = "Fly to a depot, land, load a medical kit (takes the depot's load time), take off again."
    params = {"depot": {"type": "string", "description": "depot id", "default": "base"}}

    def start(self, ctx):
        d = ctx.world.depots.get(self.args["depot"])
        if d is None:
            raise SkillError(f"no depot {self.args['depot']!r} (depots: {', '.join(ctx.world.depots)})")
        if ctx.state.payload:
            raise SkillError("already carrying a kit")
        self.depot = d
        self.land = LandAt(ctx, d.position)
        self.phase, self.t_loaded = "land", None

    def tick(self, ctx, dt):
        if self.phase == "land":
            sp = self.land.tick(ctx)
            if self.land.done(ctx):
                self.phase, self.t_on = "load", ctx.t
            return sp
        if self.phase == "load":
            if ctx.t - self.t_on >= self.depot.load_s:
                err = ctx.actuators.load(self.depot.id)
                if err:
                    self.fail(ctx, err)
                    return Setpoint("idle")
                self.phase, self.up = "takeoff", TakeOff(ctx, ctx.limits.cruise_alt)
            return Setpoint("idle")
        sp = self.up.tick(ctx)
        if self.up.done(ctx):
            self.succeed(ctx, f"kit loaded at {self.depot.id}, back at {ctx.state.alt:.0f} m")
        return sp


class Deliver(Skill):
    name, category = "deliver", "payload"
    summary = ("Drop the kit next to the person: fly over the estimate (or a point), descend to "
               "the drop altitude, hold at the computed release point (ballistic drop with drag "
               "and the estimated wind), release, climb back. Refuses when the estimate is too "
               "uncertain: inspect first.")
    params = {"target": {"type": "string", "enum": ["person", "point"], "default": "person"},
              "east": NUM, "north": NUM,
              "drop_alt": {**NUM, "description": "release altitude, m", "default": 8.0},
              "max_sigma": {**NUM, "description": "refuse above this estimate uncertainty, m",
                            "default": 8.0}}

    def start(self, ctx):
        a = self.args
        if not ctx.state.payload:
            raise SkillError("not carrying a kit: pick_up first")
        if a["target"] == "person":
            b = ctx.belief() if ctx.belief else None
            if b is None or b.source == "prior":
                raise SkillError("no person track: find the person, or deliver to a point")
            if b.sigma > a["max_sigma"]:
                raise SkillError(f"estimate uncertain by {b.sigma:.1f} m (> {a['max_sigma']:g}): inspect first")
            self.target = (b.east, b.north)
            self.src = b.source                   # follow this track only (a moving person: lead)
        else:
            self.src = None
            if a.get("east") is None or a.get("north") is None:
                raise SkillError("deliver to a point needs east and north")
            self.target = (a["east"], a["north"])
        if not ctx.limits.min_alt <= a["drop_alt"] <= 20:
            raise SkillError(f"drop altitude must be {ctx.limits.min_alt:g}-20 m")
        self.phase = "transit"
        self.transit = Transit(ctx, self._aim(ctx), ctx.limits.cruise_alt)

    def _aim(self, ctx):
        """Release point for a hovering drop: inverts the ballistic model with the wind estimate."""
        wind = ctx.actuators.wind_estimate()
        return release_point(self.target, self.args["drop_alt"], (0.0, 0.0, 0.0), wind,
                             **ctx.world.kit.params())

    def tick(self, ctx, dt):
        st = ctx.state
        if self.src is not None and self.phase in ("transit", "descend", "settle"):
            b = ctx.belief()
            # the person's track (its id changes when they are re-acquired); never a jump to a stray
            if b.source != "prior" and (b.found or b.source == self.src) and \
                    math.dist((b.east, b.north), self.target) < 15.0:
                # lead a walking person: where they will be when the kit lands
                # only the settle and the fall: a walker's heading changes within ~20 s,
                # so predicting further than the drop itself aims worse than not leading
                lead = 2.0 + fall_time(self.args["drop_alt"], **ctx.world.kit.params())
                self.target = (b.east + b.ve * lead, b.north + b.vn * lead)
        if self.phase == "transit":
            sp = self.transit.tick(ctx)
            if self.transit.done(ctx):
                self.phase = "descend"
            return sp
        aim = self._aim(ctx)
        if self.phase == "descend":
            if abs(st.alt - self.args["drop_alt"]) < 0.5 and math.dist((st.east, st.north), aim) < 1.0:
                self.phase, self.t_settle = "settle", ctx.t
            return Setpoint("position", (aim[0], aim[1], self.args["drop_alt"]), max_speed=2.0)
        if self.phase == "settle":
            if ctx.t - self.t_settle >= 2.0:
                where = ctx.actuators.release()
                self.phase = "climb"
                ctx.emit("released", f"kit released at ({st.east:.0f}, {st.north:.0f}, "
                         f"{st.alt:.0f} m) {where}", wake=False)
            return Setpoint("position", (aim[0], aim[1], self.args["drop_alt"]), max_speed=1.0)
        tgt = (st.east, st.north, ctx.limits.cruise_alt)
        if abs(st.alt - tgt[2]) < 1.0:
            self.succeed(ctx, "kit delivered; climbing done")
        return Setpoint("position", tgt, max_speed=ctx.limits.max_climb)
