"""Targets: track the person, intercept a moving one, capture an image, mark a spot."""
from __future__ import annotations

import math

from ..base import Setpoint, Skill, SkillError
from ._common import Transit

NUM = {"type": "number"}


def _track(ctx):
    if ctx.belief is None:
        raise SkillError("no perception")
    b = ctx.belief()
    if b.source == "prior":
        raise SkillError("no person track yet (only the prior): search or inspect first")
    return b


class TrackPerson(Skill):
    name, category = "track_person", "targets"
    summary = ("Stay with the person: orbit them at a standoff radius (mode orbit) or hover "
               "over them (mode follow), following the estimate as it updates. With "
               "until=help_arrived it finishes when help arrives; otherwise it runs until replaced.")
    params = {"mode": {"type": "string", "enum": ["orbit", "follow"], "default": "orbit"},
              "standoff": {**NUM, "description": "orbit radius, m (keep it inside the camera view)",
                           "default": 10.0},
              "alt": {**NUM, "description": "m", "default": 25.0},
              "speed": {**NUM, "description": "m/s along the orbit", "default": 4.0},
              "until": {"type": "string", "enum": ["none", "help_arrived"], "default": "none"}}

    def start(self, ctx):
        _track(ctx)
        if self.args["standoff"] < 5:
            raise SkillError("standoff must be at least 5 m")
        fp = ctx.camera.footprint(self.args["alt"])
        if self.args["mode"] == "orbit" and self.args["standoff"] > 0.8 * fp:
            raise SkillError(f"at {self.args['alt']:g} m the camera sees {fp:.0f} m around: a "
                             f"{self.args['standoff']:g} m standoff loses sight of the person "
                             f"(use <= {0.8 * fp:.0f} m or fly higher)")
        self.approach, self.on_station = None, False
        self.t0 = ctx.t

    def tick(self, ctx, dt):
        a, st = self.args, ctx.state
        b = ctx.belief()
        if a["until"] == "help_arrived" and ctx.world_state.help_arrived(ctx.t):
            self.succeed(ctx, f"help arrived; tracked for {ctx.t - self.t0:.0f} s")
            return Setpoint("position", st.pos, max_speed=2.0)
        if b.source == "prior":
            self.fail(ctx, "lost the person track")
            return Setpoint("position", st.pos, max_speed=2.0)
        r = a["standoff"] if a["mode"] == "orbit" else 0.0
        d = math.hypot(st.east - b.east, st.north - b.north)
        if not self.on_station:
            if self.approach is None:
                self.approach = Transit(ctx, (b.east, b.north), a["alt"])
            sp = self.approach.tick(ctx)
            if d <= r + 10 or self.approach.done(ctx):
                self.on_station = True
                ctx.emit("progress", f"on station over the person ({d:.0f} m)", wake=False)
            return sp
        if a["mode"] == "follow":
            return Setpoint("position", (b.east, b.north, a["alt"]), (b.ve, b.vn, 0.0),
                            max_speed=ctx.limits.max_speed)
        ang = math.atan2(st.north - b.north, st.east - b.east) + max(4.0, a["speed"]) / r
        v = a["speed"]
        tgt = (b.east + r * math.cos(ang), b.north + r * math.sin(ang), a["alt"])
        vel = (-v * math.sin(ang) + b.ve, v * math.cos(ang) + b.vn, 0.0)
        return Setpoint("position", tgt, vel, max_speed=v + 2.0,
                        yaw=math.atan2(b.north - st.north, b.east - st.east))

    def progress(self, ctx):
        return f"tracking for {ctx.t - self.t0:.0f} s"


def intercept_point(p, target, vt, speed):
    """Where a pursuer at p flying at `speed` meets a target at `target` moving
    at constant velocity vt (the classic closest-time solution of
    |target + vt*t - p| = speed*t). Returns (point, time) or None."""
    dx, dy = target[0] - p[0], target[1] - p[1]
    a = vt[0] ** 2 + vt[1] ** 2 - speed ** 2
    b = 2 * (dx * vt[0] + dy * vt[1])
    c = dx * dx + dy * dy
    if abs(a) < 1e-9:
        t = -c / b if b < 0 else None
    else:
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        roots = [(-b - math.sqrt(disc)) / (2 * a), (-b + math.sqrt(disc)) / (2 * a)]
        pos = [r for r in roots if r > 0]
        t = min(pos) if pos else None
    if t is None:
        return None
    return (target[0] + vt[0] * t, target[1] + vt[1] * t), t


class Intercept(Skill):
    name, category = "intercept", "targets"
    summary = ("Fly to where a moving person will be (constant-velocity intercept from the "
               "tracker's estimate), re-aiming every few seconds; finishes within 5 m of them.")
    params = {"alt": {**NUM, "description": "m", "default": 20.0},
              "speed": {**NUM, "description": "m/s (default: cruise speed)"}}

    def start(self, ctx):
        _track(ctx)
        self.speed = self.args.get("speed") or ctx.limits.cruise_speed
        self._aim(ctx)

    def _aim(self, ctx):
        b, st = ctx.belief(), ctx.state
        hit = intercept_point((st.east, st.north), (b.east, b.north), (b.ve, b.vn), self.speed)
        self.aim = hit[0] if hit else (b.east, b.north)
        self.transit = Transit(ctx, self.aim, self.args["alt"], self.speed)
        self.t_aim = ctx.t

    def tick(self, ctx, dt):
        b, st = ctx.belief(), ctx.state
        if math.hypot(st.east - b.east, st.north - b.north) < 5.0:
            self.succeed(ctx, "with the person")
        elif ctx.t - self.t_aim > 3.0:
            self._aim(ctx)
        return self.transit.tick(ctx)


class CaptureImage(Skill):
    name, category, instant = "capture_image", "targets", True
    summary = "Take a photo with the camera and get what the detector sees in it."
    params = {"note": {"type": "string", "description": "why (kept with the photo)", "default": ""}}

    def start(self, ctx):
        self.succeed(ctx, ctx.actuators.capture(self.args["note"]))


class DropMarker(Skill):
    name, category, instant = "drop_marker", "targets", True
    summary = "Mark a spot on the shared map (here, or at a point) with a label, for later or for the crew."
    params = {"label": {"type": "string"}, "east": NUM, "north": NUM}
    required = ["label"]

    def start(self, ctx):
        st = ctx.state
        e = self.args.get("east", st.east)
        n = self.args.get("north", st.north)
        ctx.markers.append({"t": ctx.t, "label": self.args["label"], "east": e, "north": n})
        self.succeed(ctx, f"marked '{self.args['label']}' at ({e:.0f}, {n:.0f})")
