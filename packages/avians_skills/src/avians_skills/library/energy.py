"""Battery service: recharge on a charging pad, or swap at a swap station."""
from __future__ import annotations

from ..base import Setpoint, Skill, SkillError
from ._common import LandAt, TakeOff

NUM = {"type": "number"}


class _Service(Skill):
    kind = ""

    def start(self, ctx):
        sid = self.args["station"]
        s = ctx.world.stations.get(sid)
        if s is None or s.kind != self.kind:
            ok = [x.id for x in ctx.world.stations.values() if x.kind == self.kind]
            raise SkillError(f"{sid!r} is not a {self.kind} station ({self.kind} stations: {', '.join(ok)})")
        self.station = s
        self.land = LandAt(ctx, s.position)
        self.phase = "land"

    def tick(self, ctx, dt):
        if self.phase == "land":
            sp = self.land.tick(ctx)
            if self.land.done(ctx):
                self.phase, self.t_on, self.b0 = "service", ctx.t, ctx.state.battery
            return sp
        if self.phase == "service":
            if self.serve(ctx, dt):
                self.phase, self.up = "takeoff", TakeOff(ctx, ctx.limits.cruise_alt)
            return Setpoint("idle")
        sp = self.up.tick(ctx)
        if self.up.done(ctx):
            self.succeed(ctx, f"battery {self.b0:.0f} -> {ctx.state.battery:.0f} % in "
                         f"{ctx.t - self.t_on:.0f} s on {self.station.id}; airborne")
        return sp

    def serve(self, ctx, dt) -> bool:
        raise NotImplementedError


class Recharge(_Service):
    name, category, kind = "recharge", "energy", "charge"
    summary = "Fly to a charging pad, land, charge to a target level, take off again."
    params = {"station": {"type": "string", "description": "charging station id"},
              "until_pct": {**NUM, "description": "charge to this %", "default": 95.0}}
    required = ["station"]

    def serve(self, ctx, dt):
        ctx.actuators.charge(self.station.id, dt)
        return ctx.state.battery >= min(self.args["until_pct"], 100.0) - 1e-6

    def eta_s(self, ctx):
        need = max(0.0, self.args["until_pct"] - ctx.state.battery)
        return need / self.station.charge_rate_pct_s


class SwapBattery(_Service):
    name, category, kind = "swap_battery", "energy", "swap"
    summary = "Fly to a battery swap station, land, get a full battery (fixed time), take off again."
    params = {"station": {"type": "string", "description": "swap station id"}}
    required = ["station"]

    def serve(self, ctx, dt):
        if ctx.t - self.t_on >= self.station.swap_s:
            ctx.actuators.swap(self.station.id)
            return True
        return False
