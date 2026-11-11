"""Runs one skill at a time on a simulator, through the reflex layer.

    ex = Executor(sim)
    ok, text = ex.start("search_region", {"region": "r3"})   # pre-check + start
    ex.tick()                                               # one control step

A new start() replaces the running skill (it is interrupted). With no skill
running the drone holds (or stays idle on the ground). The reflex layer sees
every setpoint; when it intervenes, the skill is interrupted and an event
wakes the pilot. Events accumulate in sim.ctx.events; the pilot loop drains
them.
"""
from __future__ import annotations

from avians_safety import precheck
from avians_skills import Setpoint, SkillError, create
from avians_skills.library.movement import hold


class Executor:
    def __init__(self, sim):
        self.sim = sim
        self.ctx = sim.ctx
        self.skill = None
        self.history: list[dict] = []
        sim.executor = self

    def start(self, name: str, args: dict) -> tuple[bool, str]:
        """(accepted, what to tell the pilot)."""
        ctx = self.ctx
        try:
            skill = create(name, args)
            warnings = precheck(ctx, name, skill.args)
            skill.started_t = ctx.t
            skill.start(ctx)
        except SkillError as e:
            self.history.append({"t": ctx.t, "skill": name, "args": args, "accepted": False,
                                 "reason": str(e)})
            return False, f"rejected: {e}"
        running = self.skill if self.skill is not None and not self.skill.done else None
        self.history.append({"t": ctx.t, "skill": name, "args": skill.args, "accepted": True,
                             "warnings": warnings})
        warn = "".join(f"\nwarning: {w}" for w in warnings)
        if skill.done:
            # finished inside start(): its answer is the reply; don't wake the pilot twice
            ctx.events = [e for e in ctx.events
                          if not (e.kind == "skill_done" and e.data.get("skill") == name)]
            text = f"{name}: {skill.message}"
            if skill.instant:                    # report, set_speed, ...: the running skill goes on
                if running is not None:
                    text += f" (still running: {running.describe()})"
                return True, text + warn
        if running is not None:
            running.interrupt(ctx, f"replaced by {name}")
        self.skill = skill
        if skill.done:
            return True, f"{name}: {skill.message}" + warn
        text = f"started {skill.describe()}"
        eta = skill.eta_s(ctx)
        if eta:
            text += f", about {eta:.0f} s"
        return True, text + warn

    def tick(self) -> None:
        """One control step: the skill's setpoint through the reflex, then the simulator."""
        self.sim.step(self.setpoint())

    def setpoint(self):
        """The setpoint to send this tick (reflex applied), without stepping:
        MultiSim collects one per drone and steps them together."""
        ctx, sim = self.ctx, self.sim
        sp = None
        if self.skill is not None and not self.skill.done:
            try:
                sp = self.skill.tick(ctx, sim.dt)
            except SkillError as e:
                self.skill.fail(ctx, str(e))
        if sp is None:
            st = ctx.state
            sp = Setpoint("idle") if st.landed else hold(st)
        sp, why = sim.reflex.check(ctx, sp)
        if why:
            if self.skill is not None and not self.skill.done:
                self.skill.interrupt(ctx, why)
            ctx.emit("reflex", f"safety reflex: {why}")
        return sp

    def status(self) -> str:
        s = self.skill
        if s is None:
            return "no skill has run yet"
        if s.done:
            return f"last skill {s.describe()} {s.status}: {s.message}"
        p = s.progress(self.ctx)
        eta = s.eta_s(self.ctx)
        return (f"running {s.describe()} for {self.ctx.t - (s.started_t or 0):.0f} s"
                + (f": {p}" if p else "") + (f", about {eta:.0f} s left" if eta else ""))
