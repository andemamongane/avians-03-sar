"""The pilot's prompts: one fixed system prompt per mission (cacheable), one
self-contained decision prompt per decision point."""
from __future__ import annotations

import math

from avians_skills import tools as skill_tools

KEEP_TOOL = {
    "name": "keep",
    "description": "Let the running skill continue (or stay idle). Use when nothing needs to change.",
    "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
}
COMMON = {
    "reason": {"type": "string", "description": "one sentence: why this, now"},
    "notes": {"type": "string",
              "description": "optional: replaces your notes (your only memory across decisions); keep them short"},
}


def tool_definitions(names: list[str]) -> list[dict]:
    """Skills + keep, each with the common reason/notes fields."""
    out = []
    for t in skill_tools(names) + [KEEP_TOOL]:
        t = {**t, "input_schema": {**t["input_schema"],
                                   "properties": {**t["input_schema"]["properties"], **COMMON}}}
        out.append(t)
    return out


SYSTEM = """You are the pilot of a search-and-rescue drone.

You fly by calling skills (the tools). A skill runs on its own until it
finishes, fails or you replace it: it plans paths around no-fly zones, closed
zones and obstacles, holds altitude and speed, and reports back. Starting a new
skill replaces the running one (instant skills such as report or set_speed do
not). A safety layer under you brakes before no-fly zones and lands on a
critical battery; avoid ever needing it.

You are asked to decide at decision points: when something happens (a skill
finished or failed, the person was found, a zone changed, the battery crossed
a mark, the safety layer acted, help arrived) and at least every {wake_s:g} s.
At each decision call exactly ONE tool. Call keep to let the running skill
continue. Every tool also takes `reason` (one sentence) and `notes`.

Your decision takes effect only after your thinking time, and the drone keeps
flying meanwhile: prefer decisions that stay right for a while. Each decision
prompt is complete on its own: you will not see earlier prompts. Use `notes`
for anything you want to remember (plans, suspicions, what is left to do).

MISSION
{brief}

The mission is scored by signal temporal logic: every clause below must hold
over the whole flight, and robustness (how much margin you keep) is the
reward. The spec's signals are listed after it.

{spec}

{signals}

{world}

Units: metres east/north of the launch point, altitude above ground, seconds
since the start, battery in percent."""


def system_prompt(mission, wake_s: float, world_text: str | None = None) -> str:
    used = set()
    for c in mission.spec.clauses:
        used |= c.node.signals()
    reg = mission.spec.registry
    sig = "Signals used by the spec:\n" + "\n".join(
        f"- {n} ({reg[n].unit}): {reg[n].description}" for n in reg.names() if n in used)
    return SYSTEM.format(wake_s=wake_s, brief=mission.brief, spec=mission.spec.describe(),
                         signals=sig, world=world_text or mission.world.describe())


def _battery_to(sim, goal) -> tuple[float, float]:
    v = sim.vehicle.s
    d = math.dist((v.east, v.north), goal)
    return d, sim.energy.cost(d, sim.ctx.limits.cruise_speed, v.payload)


def decision_prompt(sim, executor, memory, monitor, events, last_result: str, horizon: float) -> str:
    v, t, lim = sim.vehicle.s, sim.t, sim.ctx.limits
    b = sim.tracker.belief()
    endurance = sim.energy.endurance_s(max(0.0, v.battery - lim.reserve_pct), lim.cruise_speed, v.payload)
    lines = [f"TIME t={t:.1f} s of {horizon:g} s",
             f"DRONE at ({v.east:.0f}, {v.north:.0f}) alt {v.alt:.0f} m, speed {v.speed:.1f} m/s, "
             f"{'ON THE GROUND' if v.landed else 'flying'}; battery {v.battery:.1f} % "
             f"(~{endurance / 60:.1f} min of cruise above the {lim.reserve_pct:g} % reserve); "
             f"carrying {'the medical kit' if v.payload else 'nothing'}; cruise {lim.cruise_speed:g} m/s "
             f"at {lim.cruise_alt:g} m",
             f"SKILL: {executor.status()}",
             f"LAST RESULT: {last_result or '(none)'}",
             "EVENTS since the last decision:"]
    lines += [f"- t={e.t:.0f} {e.kind}: {e.text}" for e in events] or ["- (none)"]
    if b.found:
        lines.append(f"PERSON: FOUND (t={sim.tracker.found_t:.0f} s). Estimate ({b.east:.1f}, {b.north:.1f}) "
                     f"+/- {b.sigma:.1f} m, confidence {b.conf:.2f}; you are "
                     f"{math.hypot(v.east - b.east, v.north - b.north):.0f} m away")
    elif b.source != "prior":
        lines.append(f"PERSON: candidate at ({b.east:.0f}, {b.north:.0f}) +/- {b.sigma:.1f} m, "
                     f"confidence {b.conf:.2f} (found at {sim.tracker.found_threshold:.2f}); "
                     f"{b.tracks} candidate track(s)")
    else:
        lines.append(f"PERSON: not found; no credible candidate ({b.tracks} weak track(s))")
    cov = sim.coverage.summary(t)
    lines.append("SEARCH COVERAGE: " + "; ".join(
        f"{rid} {c['covered']:.0%} seen" + (f", last {c['last_seen_s_ago']:.0f} s ago"
                                            if c['last_seen_s_ago'] is not None else ", never")
        for rid, c in cov.items()))
    lines.append("WORLD NOW:\n" + sim.world_state.status(t))
    near = []
    for s in sim.world.stations.values():
        d, c = _battery_to(sim, s.position)
        near.append(f"{s.id} ({s.kind}) {d:.0f} m, ~{c:.0f} %")
    for dp in sim.world.depots.values():
        d, c = _battery_to(sim, dp.position)
        near.append(f"depot {dp.id} {d:.0f} m, ~{c:.0f} % ({sim.supplies.get(dp.id, 0)} kit(s) left)")
    lines.append("DISTANCES (battery to get there): " + "; ".join(near))
    decided = monitor.status_lines() if monitor else []
    if decided:
        lines.append("SPEC, decided so far:\n" + "\n".join(decided))
    lines.append("MISSION MEMORY:\n" + memory.render_facts())
    lines.append("YOUR NOTES:\n" + (memory.notes or "(empty)"))
    lines.append("RECENT DECISIONS:\n" + memory.render_decisions())
    lines.append("Decide: call one tool.")
    return "\n".join(lines)
