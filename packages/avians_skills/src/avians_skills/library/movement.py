"""Moving the drone: take off, go somewhere, hold, wait, orbit, land, return."""
from __future__ import annotations

import math

from ..base import Context, Setpoint, Skill, SkillError
from ._common import LandAt, TakeOff, Transit, hold

NUM = {"type": "number"}


def _goal(ctx: Context, args: dict) -> tuple[float, float]:
    if args.get("place"):
        try:
            return ctx.place(args["place"])
        except KeyError:
            raise SkillError(f"no place named {args['place']!r}") from None
    if args.get("east") is None or args.get("north") is None:
        raise SkillError("give east and north, or a place")
    return (float(args["east"]), float(args["north"]))


class Takeoff(Skill):
    name, category = "takeoff", "movement"
    summary = "Arm and climb straight up to the given altitude."
    params = {"alt": {**NUM, "description": "altitude, m", "default": 30.0}}

    def start(self, ctx):
        if not ctx.state.landed:
            self.succeed(ctx, "already flying")
            return
        self.op = TakeOff(ctx, self.args["alt"])

    def tick(self, ctx, dt):
        sp = self.op.tick(ctx)
        if self.op.done(ctx):
            self.succeed(ctx, f"at {ctx.state.alt:.0f} m")
        return sp


class Goto(Skill):
    name, category = "goto", "movement"
    summary = ("Fly to a point or a named place (station, depot, region centre, 'launch') "
               "on the shortest path around no-fly zones, closed zones and obstacles.")
    params = {"east": {**NUM, "description": "m east"}, "north": {**NUM, "description": "m north"},
              "place": {"type": "string", "description": "a station, depot or region id, or 'launch'"},
              "alt": {**NUM, "description": "altitude, m (default: cruise altitude)"},
              "speed": {**NUM, "description": "m/s (default: cruise speed)"}}

    def start(self, ctx):
        self.goal = _goal(ctx, self.args)
        alt = self.args.get("alt") or ctx.limits.cruise_alt
        self.transit = Transit(ctx, self.goal, alt, self.args.get("speed"))

    def tick(self, ctx, dt):
        sp = self.transit.tick(ctx)
        if self.transit.done(ctx):
            self.succeed(ctx, f"arrived at ({self.goal[0]:.0f}, {self.goal[1]:.0f})")
        return sp

    def progress(self, ctx):
        return f"{self.transit.remaining():.0f} m to go"

    def eta_s(self, ctx):
        return self.transit.eta_s()


class Hover(Skill):
    name, category = "hover", "movement"
    summary = "Hold position for some seconds (0: until the next command)."
    params = {"seconds": {**NUM, "description": "how long; 0 holds until replaced", "default": 10.0}}

    def start(self, ctx):
        st = ctx.state
        if st.landed:
            raise SkillError("on the ground: take off first")
        self.at, self.t0 = (st.east, st.north, st.alt), ctx.t

    def tick(self, ctx, dt):
        if self.args["seconds"] > 0 and ctx.t - self.t0 >= self.args["seconds"]:
            self.succeed(ctx, f"held for {self.args['seconds']:.0f} s")
        return Setpoint("position", self.at, max_speed=2.0)

    def eta_s(self, ctx):
        s = self.args["seconds"]
        return None if s <= 0 else max(0.0, s - (ctx.t - self.t0))


class WaitUntil(Skill):
    name, category = "wait_until", "movement"
    summary = ("Hold position (or stay on the ground) until something happens: help arrives, "
               "the person is found, a time, or the battery drops below a value.")
    params = {"condition": {"type": "string", "enum": ["help_arrived", "found", "time", "battery_below"]},
              "value": {**NUM, "description": "the time (s) or battery (%) for time/battery_below",
                        "default": 0.0},
              "timeout_s": {**NUM, "description": "give up after this long", "default": 900.0}}
    required = ["condition"]

    def start(self, ctx):
        st = ctx.state
        self.at, self.t0, self.ground = (st.east, st.north, st.alt), ctx.t, st.landed

    def _met(self, ctx) -> bool:
        c, v = self.args["condition"], self.args["value"]
        if c == "help_arrived":
            return ctx.world_state.help_arrived(ctx.t)
        if c == "found":
            return bool(ctx.belief and ctx.belief().found)
        if c == "time":
            return ctx.t >= v
        return ctx.state.battery < v

    def tick(self, ctx, dt):
        if self._met(ctx):
            self.succeed(ctx, f"{self.args['condition']} after {ctx.t - self.t0:.0f} s")
        elif ctx.t - self.t0 > self.args["timeout_s"]:
            self.fail(ctx, f"timed out after {self.args['timeout_s']:.0f} s")
        return Setpoint("idle") if self.ground else Setpoint("position", self.at, max_speed=2.0)


class Orbit(Skill):
    name, category = "orbit", "movement"
    summary = ("Circle a point (or the person's current estimate) at a radius, altitude and "
               "speed; laps 0 circles until replaced.")
    params = {"around": {"type": "string", "enum": ["point", "person"], "default": "point"},
              "east": NUM, "north": NUM,
              "radius": {**NUM, "description": "m", "default": 20.0},
              "alt": {**NUM, "description": "m (default: cruise altitude)"},
              "speed": {**NUM, "description": "m/s along the circle", "default": 5.0},
              "laps": {**NUM, "description": "0: until replaced", "default": 0.0}}

    def start(self, ctx):
        a = self.args
        if a["around"] == "point" and (a.get("east") is None or a.get("north") is None):
            raise SkillError("orbit around a point needs east and north")
        if a["around"] == "person" and (ctx.belief is None or ctx.belief().source == "prior"):
            raise SkillError("no person track to orbit (only the prior): search or inspect first")
        if a["radius"] < 5:
            raise SkillError("radius must be at least 5 m")
        self.alt = a.get("alt") or ctx.limits.cruise_alt
        self.swept, self.last_angle, self.approach = 0.0, None, None

    def centre(self, ctx):
        if self.args["around"] == "person":
            b = ctx.belief()
            return (b.east, b.north)
        return (self.args["east"], self.args["north"])

    def tick(self, ctx, dt):
        st, (ce, cn) = ctx.state, self.centre(ctx)
        r, v = self.args["radius"], min(self.args["speed"], ctx.limits.max_speed)
        d = math.hypot(st.east - ce, st.north - cn)
        ang = math.atan2(st.north - cn, st.east - ce)
        if abs(d - r) > 8.0 and self.last_angle is None:      # get onto the circle first
            if self.approach is None:
                entry = (ce + r * math.cos(ang), cn + r * math.sin(ang))
                self.approach = Transit(ctx, entry, self.alt)
            sp = self.approach.tick(ctx)
            if not self.approach.done(ctx):
                return sp
        if self.last_angle is not None:
            self.swept += abs(math.remainder(ang - self.last_angle, 2 * math.pi))
        self.last_angle = ang
        if self.args["laps"] > 0 and self.swept >= 2 * math.pi * self.args["laps"]:
            self.succeed(ctx, f"{self.args['laps']:.0f} lap(s) done")
        nxt = ang + max(4.0, v) / r                           # carrot ahead on the circle
        tgt = (ce + r * math.cos(nxt), cn + r * math.sin(nxt), self.alt)
        vel = (-v * math.sin(nxt), v * math.cos(nxt), 0.0)
        return Setpoint("position", tgt, vel, max_speed=v + 1.0,
                        yaw=math.atan2(cn - st.north, ce - st.east))

    def progress(self, ctx):
        return f"{self.swept / (2 * math.pi):.1f} laps"


class Land(Skill):
    name, category = "land", "movement"
    summary = "Land where you are (or at a named pad with place)."
    params = {"place": {"type": "string", "description": "a station or depot id, or 'launch'"}}

    def start(self, ctx):
        st = ctx.state
        pad = ctx.place(self.args["place"]) if self.args.get("place") else (st.east, st.north)
        self.op = LandAt(ctx, pad, approach_alt=max(10.0, min(st.alt, ctx.limits.cruise_alt)))

    def tick(self, ctx, dt):
        sp = self.op.tick(ctx)
        if self.op.done(ctx):
            self.succeed(ctx, f"landed at ({ctx.state.east:.0f}, {ctx.state.north:.0f})")
        return sp


class PrecisionLand(Land):
    name = "precision_land"
    summary = "Land exactly on a named pad (station, depot or 'launch'), aligning over it first."
    params = {"place": {"type": "string", "description": "a station or depot id, or 'launch'"}}
    required = ["place"]


class ReturnToLaunch(Skill):
    name, category = "rtl", "movement"
    summary = "Return to the launch point at cruise altitude and land."
    params = {}

    def start(self, ctx):
        self.op = LandAt(ctx, ctx.place("launch"))

    def tick(self, ctx, dt):
        sp = self.op.tick(ctx)
        if self.op.done(ctx):
            self.succeed(ctx, "landed at launch")
        return sp


class SetSpeed(Skill):
    name, category, instant = "set_speed", "movement", True
    summary = "Change the cruise speed (and optionally cruise altitude) used by later commands."
    params = {"speed": {**NUM, "description": "m/s"}, "alt": {**NUM, "description": "cruise altitude, m"}}

    def start(self, ctx):
        lim = ctx.limits
        if self.args.get("speed") is not None:
            s = float(self.args["speed"])
            if not 1.0 <= s <= lim.max_speed:
                raise SkillError(f"speed must be 1-{lim.max_speed:g} m/s")
            lim.cruise_speed = s
        if self.args.get("alt") is not None:
            a = float(self.args["alt"])
            if not lim.min_alt <= a <= lim.max_alt:
                raise SkillError(f"altitude must be {lim.min_alt:g}-{lim.max_alt:g} m")
            lim.cruise_alt = a
        self.succeed(ctx, f"cruise {lim.cruise_speed:g} m/s at {lim.cruise_alt:g} m")


class EmergencyStop(Skill):
    name, category = "emergency_stop", "movement"
    summary = "Brake now and hover in place (stops whatever was running)."
    params = {}

    def start(self, ctx):
        st = ctx.state
        if st.landed:
            self.succeed(ctx, "on the ground")
            return
        v, d = st.speed, ctx.limits.stopping_distance(st.speed)
        k = d / v if v > 0.1 else 0.0
        self.at = (st.east + st.ve * k, st.north + st.vn * k, max(st.alt, ctx.limits.min_alt))

    def tick(self, ctx, dt):
        if ctx.state.speed3 < 0.3:
            self.succeed(ctx, f"stopped at ({ctx.state.east:.0f}, {ctx.state.north:.0f})")
        return Setpoint("position", self.at, max_speed=ctx.limits.max_speed)


__all__ = ["EmergencyStop", "Goto", "Hover", "Land", "Orbit", "PrecisionLand", "ReturnToLaunch",
           "SetSpeed", "Takeoff", "WaitUntil", "hold"]
