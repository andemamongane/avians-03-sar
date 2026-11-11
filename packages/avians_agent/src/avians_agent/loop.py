"""One episode: the simulator runs continuously; the pilot decides at decision
points; each decision takes effect after a simulated latency.

    t_obs     a decision point: the prompt is built from the state at t_obs
    t_apply   t_obs + latency (sampled from the pilot's latency model); until
              then the drone keeps flying whatever it was flying
    events    everything that happens in between is shown at the next decision

Decision points: the first step; any waking event (skill finished or failed,
person found, zone change, battery mark, reflex, help arrived, a rejected
call); and at least every wake_s. The model's real wall-clock time does not
matter in the fast tier (the simulator waits for it), so rollouts stay fast
while the timing the drone experiences is the modelled one.

With a router (student pilots), each decision point first asks the router:
commit (keep, no model), local (the student) or escalate (the teacher); a
student calling `escalate` also hands over to the teacher. Latencies add up.

Several drones (mission `drones: n`): one Agent per drone, each with its own
pilot, memory, decision timer, latency and ADP trajectory, all flying in one
shared world (avians_sim.multi). A Fleet adapter makes one drone and n drones
the same loop.
"""
from __future__ import annotations

import copy
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from avians_adp import EpisodeRecorder
from avians_sim import Executor, FastSim
from avians_sim.multi import MultiSim
from avians_skills import BENCHMARK_TOOLS, SWARM_TOOLS
from avians_spec import OnlineMonitor

from .latency import LatencyModel
from .memory import Memory
from .pilots import Decision, DecisionRequest, snapshot
from .prompts import decision_prompt, system_prompt, tool_definitions
from .router import features

FACT_KINDS = {"found", "kit_landed", "zone", "help", "reflex", "released", "news"}
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0)}


def _cost(model: str, usage: dict) -> float:
    pin, pout = PRICES.get(model, (0.0, 0.0))
    return (pin * (usage.get("input", 0) + 1.25 * usage.get("cache_write", 0) + 0.1 * usage.get("cache_read", 0))
            + pout * usage.get("output", 0)) / 1e6


def _decide(state, prompt, events, system, tools, pilots, router, lat, lat_router, rng):
    """Route, ask the pilot(s): (Decision, simulated latency, who, route)."""
    route, latency, chosen = None, 0.0, "local"
    if router is not None:
        route = router.route(features(state, events))
        latency += lat_router.sample(rng)
        chosen = route.choice
    if chosen == "commit":
        return Decision("keep", reason="router: commit", model="router"), latency, chosen, route
    d = pilots[chosen].decide(DecisionRequest(system, prompt, tools, state))
    latency += lat[chosen].sample(rng)
    if d.tool == "escalate" and chosen == "local":
        chosen = "escalate"
        d = pilots["escalate"].decide(DecisionRequest(system, prompt, tools, state))
        latency += lat["escalate"].sample(rng)
    return d, latency, chosen, route


def _log_decision(rec, d, latency, chosen, route, state, t, n_dec, usage_tot, quiet) -> float:
    for k in usage_tot:
        usage_tot[k] += int(d.usage.get(k, 0) or 0)
    details = {"t_obs": round(t, 2), "t_apply": round(t + latency, 2), "latency_s": round(latency, 3),
               "wall_s": round(d.wall_s, 3), "model": d.model, "usage": d.usage,
               "route": asdict(route) if route else None, "by": chosen, "state": state, "error": d.error}
    if d.text or d.thinking:
        rec.message(d.text, t, thinking=d.thinking, decision=n_dec)
    rec.action(d.tool, d.args, d.reason, t, notes=d.notes, **details)
    if not quiet:
        who = f"{rec.agent} " if rec.agent else ""
        print(f"[{t:7.1f}] {who}-> {d.tool}({json.dumps(d.args)}) after {latency:.1f} s: {d.reason}"
              + (f"  ERROR {d.error}" if d.error else ""))
    return _cost(d.model, d.usage)



class Fleet:
    """One drone (a FastSim) or several (a MultiSim) behind the same calls."""

    def __init__(self, sim):
        self.sim = sim
        self.multi = isinstance(sim, MultiSim)
        self.sims = sim.sims if self.multi else {sim.drone_id: sim}

    @property
    def t(self):
        return self.sim.t

    def step(self, setpoints: dict) -> None:
        if self.multi:
            self.sim.step(setpoints)
        else:
            self.sim.step(next(iter(setpoints.values())))

    def signals(self) -> dict:
        return self.sim.signals()

    def all_landed(self) -> bool:
        return all(s.vehicle.state().landed for s in self.sims.values())

    def any_dead(self) -> bool:
        return any(s.vehicle.state().battery <= 0 and not s.vehicle.state().landed for s in self.sims.values())


class Agent:
    """The per-drone decision state of the loop."""

    def __init__(self, did, sim, pilot, memory, rec, system, tools, lat):
        self.id, self.sim, self.pilot, self.memory, self.rec = did, sim, pilot, memory, rec
        self.system, self.tools, self.lat = system, tools, lat
        self.ex = Executor(sim)
        self.pending = None
        self.since: list = []
        self.need, self.last_t, self.last_result, self.last_rejected = True, -math.inf, "", None
        self.n_dec, self.cost = 0, 0.0
        self.usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}


def _console_state(fleet, agents) -> str:
    lines = []
    for did, sim in fleet.sims.items():
        v = sim.vehicle.state()
        lines.append(f"{did}: t={sim.t:.0f} s at ({v.east:.0f}, {v.north:.0f}) alt {v.alt:.0f} m battery "
                     f"{v.battery:.0f} % {'kit aboard' if v.payload else 'no kit'}; {agents[did].ex.status()}")
    b = fleet.sim.tracker.belief()
    lines.append(f"person: {'FOUND' if b.found else 'not found'} estimate ({b.east:.0f}, {b.north:.0f}) +/- {b.sigma:.0f} m")
    lines.append(fleet.sim.world_state.status(fleet.t))
    return "\n".join(lines)


def run_episode(mission, pilot, seed: int, out_dir: str | Path, agent_cfg: dict, latency_cfg: dict,
                remote_pilot=None, router=None, quiet: bool = True, make_sim=None,
                realtime: bool = False, ground_live: int | None = None, make_pilot=None) -> dict:
    """make_sim(mission, seed, limits, dt, horizon) -> a sim (FastSim, MultiSim or a Gazebo sim);
    make_pilot(drone_id) -> a pilot per drone (default: `pilot` for every drone; scripted pilots
    keep state, so give each drone its own). A sim with realtime = True runs the pilots in threads
    while the drones fly, and the latency is the real one."""
    dc = agent_cfg["decisions"]
    wake_s, min_gap, horizon = float(dc["wake_s"]), float(dc["min_gap_s"]), mission.horizon_s
    rng = np.random.default_rng(seed + 7)
    limits, dt = copy.deepcopy(mission.limits), float(agent_cfg.get("control_dt", 0.1))
    ground_cfg = dict(mission.raw.get("ground") or {})
    n = len(mission.drones)
    if make_sim:
        sim = make_sim(mission, seed, limits, dt, horizon + 60)
    elif n > 1:
        sim = MultiSim(mission.world, seed, limits, mission.extras["critical_pct"], n=n, dt=dt,
                       horizon_s=horizon + 60, ground=ground_cfg, realtime=realtime, ids=mission.drones)
    else:
        sim = FastSim(mission.world, seed, limits, mission.extras["critical_pct"], dt=dt, horizon_s=horizon + 60,
                      ground=ground_cfg, realtime=realtime, drone_id=mission.drones[0])
    fleet = Fleet(sim)
    realtime = bool(getattr(sim, "realtime", False))
    pool = ThreadPoolExecutor(n) if realtime else None
    mon = OnlineMonitor(mission.spec, mission.extras["monitor_max_speed"],
                        float(agent_cfg["monitor"]["every_s"]), str(agent_cfg["monitor"]["policy"]))
    tool_names = list(SWARM_TOOLS if n > 1 else BENCHMARK_TOOLS) + (["escalate"] if router is not None else [])
    tools = tool_definitions(tool_names)
    lat_router = LatencyModel.for_model(latency_cfg, "onboard-router")
    ep_id = f"{mission.name}_{getattr(pilot, 'name', 'pilot')}_s{seed:04d}"
    out = Path(out_dir)
    meta = {"episode": ep_id, "mission": mission.name, "world": mission.world.name, "seed": seed,
            "pilot": getattr(pilot, "name", "?"), "remote_pilot": getattr(remote_pilot, "name", None),
            "router": type(router).__name__ if router else None,
            "tier": "gazebo" if realtime and make_sim else "fast", "drones": mission.drones,
            "envelope": mission.extras["envelope"], "horizon_s": horizon, "wake_s": wake_s,
            "person": sim.scenario.person_position(0), "help_delay_s": sim.scenario.help_delay_s,
            "scenario": sim.scenario.summary()}
    agents: dict[str, Agent] = {}
    rec0 = None
    for did, dsim in fleet.sims.items():
        p = make_pilot(did) if make_pilot else pilot
        pilots = {"local": p, "escalate": remote_pilot or p}
        lat = {k: LatencyModel.for_model(latency_cfg, getattr(x, "name", "scripted")) for k, x in pilots.items()}
        meta.setdefault("latency", {})[did] = {k: asdict(v) for k, v in lat.items()}
        system = system_prompt(mission, wake_s, dsim.world_state.describe(), did, mission.drones)
        rec = EpisodeRecorder(out, ep_id, meta, float(agent_cfg["recording"]["signals_hz"]),
                              agent=did if n > 1 else None, shared=rec0)
        rec0 = rec0 or rec
        rec.brief(system, tools)
        (out / (f"system_prompt_{did}.txt" if n > 1 else "system_prompt.txt")).write_text(system)
        a = Agent(did, dsim, pilots, Memory(int(dc["history"]), int(dc["max_notes_chars"])), rec, system,
                  tools, lat)
        a.router = router
        agents[did] = a
    console = None
    if ground_live:
        from avians_sim.ground_live import Console
        console = Console(sim.ground, lambda: _console_state(fleet, agents), lambda: fleet.t, port=ground_live)
        print(f"ground station console: {console.url}")
    end_reason, wall0 = "horizon", time.monotonic()
    stl_log = open(out / "stl_online.jsonl", "w")

    while fleet.t < horizon - 1e-9:
        t = fleet.t
        for a in agents.values():
            # ------------------------------------------------ decide
            if a.pending is None and t - a.last_t >= min_gap:
                wake = a.need or any(e.wake for e in a.since) or (t - a.last_t >= wake_s)
                if wake:
                    state = snapshot(a.sim, a.ex)
                    prompt = decision_prompt(a.sim, a.ex, a.memory, mon, a.since, a.last_result, horizon)
                    a.n_dec += 1
                    a.rec.observation(prompt, t, decision=a.n_dec)
                    job = (lambda ag=a, st=state, pr=prompt, ev=list(a.since):  # noqa: E731
                           _decide(st, pr, ev, ag.system, ag.tools, ag.pilot, ag.router, ag.lat, lat_router, rng))
                    if realtime:
                        a.pending = {"t_obs": t, "future": pool.submit(job), "state": state}
                    else:
                        d, latency, chosen, route = job()
                        a.pending = {"t_obs": t, "t_apply": t + latency, "result": (d, latency, chosen, route),
                                     "state": state}
                        a.cost += _log_decision(a.rec, d, latency, chosen, route, state, t, a.n_dec, a.usage, quiet)
                    a.since, a.need, a.last_t = [], False, t
            # ------------------------------------------------ apply
            if a.pending is not None and realtime and "result" not in a.pending and a.pending["future"].done():
                d, _, chosen, route = a.pending["future"].result()
                latency = t - a.pending["t_obs"]                       # the real one
                a.pending.update(result=(d, latency, chosen, route), t_apply=t)
                a.cost += _log_decision(a.rec, d, latency, chosen, route, a.pending["state"], a.pending["t_obs"],
                                        a.n_dec, a.usage, quiet)
            if a.pending is not None and "t_apply" in a.pending and t >= a.pending["t_apply"] - 1e-9:
                d, t_obs = a.pending["result"][0], a.pending["t_obs"]
                if d.tool == "keep":
                    ok, text = True, "kept: " + a.ex.status()
                else:
                    ok, text = a.ex.start(d.tool, d.args)
                a.memory.set_notes(d.notes)
                call = f"{d.tool}({', '.join(f'{k}={v}' for k, v in d.args.items())})"
                a.memory.add_decision(t_obs, call, text.split("\n")[0][:200], d.reason)
                a.last_result = f"(decided at t={t_obs:.0f}, applied at t={t:.0f}) {call}: {text}"
                a.rec.tool_result(text, t, accepted=ok, decision=a.n_dec)
                if ok and d.tool in ("report", "drop_marker", "request_help", "assign_tasks", "handoff", "claim_region"):
                    a.memory.add_fact(t, text)
                if not ok:
                    sig = (d.tool, json.dumps(d.args, sort_keys=True))
                    a.need = sig != a.last_rejected        # the same rejected call again: wait for wake_s
                    a.last_rejected = sig
                else:
                    a.last_rejected = None
                a.pending = None
        # ---------------------------------------------------- step
        fleet.step({did: a.ex.setpoint() for did, a in agents.items()})
        for a in agents.values():
            for e in a.sim.ctx.events:
                a.rec.event(e.t, e.kind, e.text)
                a.since.append(e)
                if e.kind in FACT_KINDS or (e.kind == "skill_done" and e.data.get("skill") in
                                             ("search_region", "recharge", "swap_battery", "pick_up", "deliver")):
                    a.memory.add_fact(e.t, e.text)
            a.sim.ctx.events.clear()
        sig = fleet.signals()
        mon.add(fleet.t, sig)
        rec0.signals(fleet.t, sig)
        rep = mon.maybe_eval(fleet.t)
        if rep is not None:
            stl_log.write(json.dumps(rep) + "\n")
        if mon.should_stop():
            end_reason = f"violated: {', '.join(mon.violated())}"
            break
        if dc.get("end_when") == "help_arrived_and_landed" and sim.world_state.help_arrived(fleet.t) \
                and fleet.all_landed() and fleet.t > 5:
            end_reason = "help arrived and landed"
            break
        if fleet.any_dead():
            end_reason = "battery depleted in flight"
            break
    stl_log.close()
    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=True)
    if console is not None:
        console.close()
    with open(out / "messages.jsonl", "w") as f:
        for m in sim.ground.summary():
            f.write(json.dumps(m, default=str) + "\n")
    tr = mon.trace()
    score = mission.spec.score(tr)
    first = next(iter(agents.values()))
    usage_tot = {k: sum(a.usage[k] for a in agents.values()) for k in first.usage}
    summary = {"episode": ep_id, "seed": seed, "end": end_reason, "t_end": round(fleet.t, 1),
               "robustness": score["robustness"], "satisfied": score["satisfied"],
               "violated": [c["clause"] for c in score["clauses"] if not c["satisfied"]],
               "decisions": sum(a.n_dec for a in agents.values()), "usage": usage_tot,
               "cost_usd": round(sum(a.cost for a in agents.values()), 4),
               "wall_s": round(time.monotonic() - wall0, 2),
               "found_t": sim.tracker.found_t, "help_dispatched_t": sim.world_state.help.dispatched_t,
               "help_arrival_t": sim.world_state.help.arrival_t,
               "drop_err": min((k["err"] for k in sim.kits if k["landed"]), default=None),
               "battery_end": {did: round(s.vehicle.state().battery, 1) for did, s in fleet.sims.items()}
               if n > 1 else round(sim.vehicle.state().battery, 1),
               "reports": sim.reports, "markers": sim.ctx.markers,
               "skills": {did: a.ex.history for did, a in agents.items()} if n > 1 else first.ex.history}
    for a in agents.values():
        if a.rec is not rec0:
            a.rec.finish(score, {"decisions": a.n_dec, "cost_usd": round(a.cost, 4)})
    rec0.finish(score, summary)
    return summary
