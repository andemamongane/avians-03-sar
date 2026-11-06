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
from avians_skills import BENCHMARK_TOOLS
from avians_spec import OnlineMonitor

from .latency import LatencyModel
from .memory import Memory
from .pilots import Decision, DecisionRequest, snapshot
from .prompts import decision_prompt, system_prompt, tool_definitions
from .router import features

FACT_KINDS = {"found", "kit_landed", "zone", "help", "reflex", "released"}
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
        print(f"[{t:7.1f}] -> {d.tool}({json.dumps(d.args)}) after {latency:.1f} s: {d.reason}"
              + (f"  ERROR {d.error}" if d.error else ""))
    return _cost(d.model, d.usage)


def run_episode(mission, pilot, seed: int, out_dir: str | Path, agent_cfg: dict, latency_cfg: dict,
                remote_pilot=None, router=None, quiet: bool = True, make_sim=None) -> dict:
    """make_sim(mission, seed, limits, dt, horizon) -> a sim; default FastSim.
    A sim with realtime = True (Gazebo) runs the pilot in a thread while the
    drone flies, and the latency is the real one."""
    dc = agent_cfg["decisions"]
    wake_s, min_gap, horizon = float(dc["wake_s"]), float(dc["min_gap_s"]), mission.horizon_s
    rng = np.random.default_rng(seed + 7)
    limits, dt = copy.deepcopy(mission.limits), float(agent_cfg.get("control_dt", 0.1))
    sim = (make_sim(mission, seed, limits, dt, horizon + 60) if make_sim else
           FastSim(mission.world, seed, limits, mission.extras["critical_pct"], dt=dt, horizon_s=horizon + 60))
    realtime = bool(getattr(sim, "realtime", False))
    pool = ThreadPoolExecutor(1) if realtime else None
    ex = Executor(sim)
    mon = OnlineMonitor(mission.spec, mission.extras["monitor_max_speed"],
                        float(agent_cfg["monitor"]["every_s"]), str(agent_cfg["monitor"]["policy"]))
    memory = Memory(int(dc["history"]), int(dc["max_notes_chars"]))
    tool_names = list(BENCHMARK_TOOLS) + (["escalate"] if router is not None else [])
    tools = tool_definitions(tool_names)
    system = system_prompt(mission, wake_s)
    pilots = {"local": pilot, "escalate": remote_pilot or pilot}
    lat = {k: LatencyModel.for_model(latency_cfg, getattr(p, "name", "scripted")) for k, p in pilots.items()}
    lat_router = LatencyModel.for_model(latency_cfg, "onboard-router")
    ep_id = f"{mission.name}_{getattr(pilot, 'name', 'pilot')}_s{seed:04d}"
    out = Path(out_dir)
    meta = {"episode": ep_id, "mission": mission.name, "world": mission.world.name, "seed": seed,
            "pilot": getattr(pilot, "name", "?"), "remote_pilot": getattr(remote_pilot, "name", None),
            "router": type(router).__name__ if router else None,
            "tier": "gazebo" if realtime else "fast",
            "envelope": mission.extras["envelope"], "horizon_s": horizon, "wake_s": wake_s,
            "latency": {k: asdict(v) for k, v in lat.items()},
            "person": sim.scenario.person_position(0), "help_delay_s": sim.scenario.help_delay_s}
    rec = EpisodeRecorder(out, ep_id, meta, float(agent_cfg["recording"]["signals_hz"]))
    rec.brief(system, tools)
    with open(out / "system_prompt.txt", "w") as f:
        f.write(system)

    pending = None                       # (t_apply, Decision, t_obs)
    since: list = []                     # events since the last decision prompt
    need, last_t, last_result, last_rejected = True, -math.inf, "", None
    n_dec, cost, wall0 = 0, 0.0, time.monotonic()
    usage_tot = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    end_reason = "horizon"
    stl_log = open(out / "stl_online.jsonl", "w")

    while sim.t < horizon - 1e-9:
        t = sim.t
        # ---------------------------------------------------- decide
        if pending is None and t - last_t >= min_gap:
            wake = need or any(e.wake for e in since) or (t - last_t >= wake_s)
            if wake:
                state = snapshot(sim, ex)
                prompt = decision_prompt(sim, ex, memory, mon, since, last_result, horizon)
                n_dec += 1
                rec.observation(prompt, t, decision=n_dec)
                job = (lambda st=state, pr=prompt, ev=list(since):  # noqa: E731
                       _decide(st, pr, ev, system, tools, pilots, router, lat, lat_router, rng))
                if realtime:
                    pending = {"t_obs": t, "future": pool.submit(job), "state": state}
                else:
                    d, latency, chosen, route = job()
                    pending = {"t_obs": t, "t_apply": t + latency, "result": (d, latency, chosen, route),
                               "state": state}
                    cost += _log_decision(rec, d, latency, chosen, route, state, t, n_dec, usage_tot, quiet)
                since, need, last_t = [], False, t
        # ---------------------------------------------------- apply
        if pending is not None and realtime and "result" not in pending and pending["future"].done():
            d, _, chosen, route = pending["future"].result()
            latency = t - pending["t_obs"]                       # the real one
            pending.update(result=(d, latency, chosen, route), t_apply=t)
            cost += _log_decision(rec, d, latency, chosen, route, pending["state"], pending["t_obs"],
                                  n_dec, usage_tot, quiet)
        if pending is not None and "t_apply" in pending and t >= pending["t_apply"] - 1e-9:
            d, t_obs = pending["result"][0], pending["t_obs"]
            if d.tool == "keep":
                ok, text = True, "kept: " + ex.status()
            else:
                ok, text = ex.start(d.tool, d.args)
            memory.set_notes(d.notes)
            call = f"{d.tool}({', '.join(f'{k}={v}' for k, v in d.args.items())})"
            memory.add_decision(t_obs, call, text.split("\n")[0][:200], d.reason)
            last_result = f"(decided at t={t_obs:.0f}, applied at t={t:.0f}) {call}: {text}"
            rec.tool_result(text, t, accepted=ok, decision=n_dec)
            if ok and d.tool in ("report", "drop_marker", "request_help"):
                memory.add_fact(t, text)
            if not ok:
                sig = (d.tool, json.dumps(d.args, sort_keys=True))
                need = sig != last_rejected        # the same rejected call again: wait for wake_s
                last_rejected = sig
            else:
                last_rejected = None
            pending = None
        # ---------------------------------------------------- step
        ex.tick()
        for e in sim.ctx.events:
            rec.event(e.t, e.kind, e.text)
            since.append(e)
            if e.kind in FACT_KINDS or (e.kind == "skill_done" and e.data.get("skill") in
                                         ("search_region", "recharge", "swap_battery", "pick_up", "deliver")):
                memory.add_fact(e.t, e.text)
        sim.ctx.events.clear()
        sig = sim.signals()
        mon.add(sim.t, sig)
        rec.signals(sim.t, sig)
        rep = mon.maybe_eval(sim.t)
        if rep is not None:
            stl_log.write(json.dumps(rep) + "\n")
        if mon.should_stop():
            end_reason = f"violated: {', '.join(mon.violated())}"
            break
        v = sim.vehicle.s
        if dc.get("end_when") == "help_arrived_and_landed" and sim.world_state.help_arrived(sim.t) \
                and v.landed and sim.t > 5:
            end_reason = "help arrived and landed"
            break
        if v.battery <= 0 and not v.landed:
            end_reason = "battery depleted in flight"
            break
    stl_log.close()
    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=True)
    tr = mon.trace()
    score = mission.spec.score(tr)
    summary = {"episode": ep_id, "seed": seed, "end": end_reason, "t_end": round(sim.t, 1),
               "robustness": score["robustness"], "satisfied": score["satisfied"],
               "violated": [c["clause"] for c in score["clauses"] if not c["satisfied"]],
               "decisions": n_dec, "usage": usage_tot, "cost_usd": round(cost, 4),
               "wall_s": round(time.monotonic() - wall0, 2),
               "found_t": sim.tracker.found_t, "help_dispatched_t": sim.world_state.help.dispatched_t,
               "help_arrival_t": sim.world_state.help.arrival_t,
               "drop_err": min((k["err"] for k in sim.kits if k["landed"]), default=None),
               "battery_end": round(sim.vehicle.s.battery, 1), "reports": sim.reports,
               "markers": sim.ctx.markers, "skills": ex.history}
    rec.finish(score, summary)
    return summary
