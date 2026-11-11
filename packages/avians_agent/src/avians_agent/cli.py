"""avians: fly, sweep, score and replay missions.

    avians check    MISSION [--seed N]              the prompts and tools the pilot gets; nothing flies
    avians fly      MISSION [--pilot P] [--seed N] [--tier fast|gazebo] [--video] [--qgc]
                            [--ground live --port 8765] [--realtime]    a live operator console
    avians rollouts MISSION [--pilot P] --seeds 0-99 [--workers 8]
    avians score    EPISODE                          re-score from signals.jsonl
    avians video    EPISODE [--speed 20]             the 2D sped-up replay (replay.mp4)
    avians qgc      EPISODE                          world.kml, flight.kml, flight.tlog for QGroundControl
    avians fly ... --realtime --qgc-live 127.0.0.1:14550   watch it live in QGroundControl
    avians gazebo-replay EPISODE                     re-fly it visually in Gazebo, recording gazebo*.mp4
    avians latency-fit DIR                           fit configs/latency.yaml from measured wall times
    avians doctor                                    what is installed and reachable
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .mission import Mission, load_config, repo_root


def _seeds(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def _out(args, mission_name: str, pilot: str) -> Path:
    base = Path(args.out) if args.out else repo_root() / "data" / "episodes"
    return base / mission_name / pilot


def _load_env() -> None:
    """Read KEY=VALUE lines from .env at the repo root into the environment (not overriding)."""
    env = repo_root() / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            k, eq, v = line.strip().partition("=")
            if eq and k and not k.startswith("#") and v and k not in os.environ:
                os.environ[k] = v


def cmd_check(a) -> int:
    from avians_sim import Executor, FastSim

    from .memory import Memory
    from .prompts import decision_prompt, system_prompt, tool_definitions
    from avians_skills import BENCHMARK_TOOLS
    m = Mission.load(a.mission, a.envelope)
    cfg = load_config("agent.yaml")
    if len(m.drones) > 1:
        from avians_sim.multi import MultiSim
        ms = MultiSim(m.world, a.seed, m.limits, m.extras["critical_pct"], n=len(m.drones), ids=m.drones)
        sim = ms.sims[m.drones[0]]
        for s_ in ms.sims.values():
            Executor(s_)
    else:
        sim = FastSim(m.world, a.seed, m.limits, m.extras["critical_pct"])
        Executor(sim)
    print("=" * 30, "SYSTEM PROMPT", "=" * 30)
    print(system_prompt(m, cfg["decisions"]["wake_s"], sim.world_state.describe(), m.drones[0], m.drones))
    print("=" * 30, "FIRST DECISION PROMPT", "=" * 30)
    print(decision_prompt(sim, sim.executor, Memory(), None, [], "", m.horizon_s))
    from avians_skills import SWARM_TOOLS
    tools = tool_definitions(SWARM_TOOLS if len(m.drones) > 1 else BENCHMARK_TOOLS)
    print("=" * 30, f"TOOLS ({len(tools)})", "=" * 30)
    for t in tools:
        print(f"- {t['name']}: {t['description'][:90]}")
    sc = sim.scenario
    print(f"\nscenario (seed {a.seed}): person at {tuple(round(x) for x in sc.person_position(0))} in "
          f"{sc.person_region} ({sc.motion}); help delay {sc.help_delay_s:.0f} s; wind "
          f"({sc.wind.east:g}, {sc.wind.north:g}); zone timeline {[(e.t, e.zone, e.safe) for e in sc.timeline]}; "
          f"{len(sc.decoys)} decoy(s); {len(sc.news)} news item(s); spec horizon {m.spec.horizon:g} s")
    return 0


def _gazebo_factory(out: Path):
    from avians_sim.aas import GazeboSim
    from avians_sim.gazebo import GazeboControl
    aas = load_config("aas.yaml")
    holder = {}

    def make(mission, seed, limits, dt, horizon):
        ctl = GazeboControl(aas.get("sim", {}), out)
        scen = mission.world.sample(seed, horizon)
        ctl.prepare(mission.world, limits, scen.person_position(0))
        sim = GazeboSim(mission.world, seed, limits, mission.extras["critical_pct"], aas, ctl, dt=dt,
                        horizon_s=horizon)
        ctl.start(horizon, seed=seed)
        holder["ctl"] = ctl
        return sim
    return make, holder


def cmd_fly(a) -> int:
    _load_env()
    from .loop import run_episode
    from .pilots import make_pilot
    m = Mission.load(a.mission, a.envelope)
    pcfg = load_config("pilots.yaml")
    name = a.pilot or pcfg.get("default", "scripted")
    pilot = make_pilot(name, pcfg)
    out = _out(a, m.name, name) / f"s{a.seed:04d}"
    make_sim, holder = (None, {})
    if a.tier == "gazebo":
        make_sim, holder = _gazebo_factory(out)
    try:
        s = run_episode(m, pilot, a.seed, out, load_config("agent.yaml"), load_config("latency.yaml"),
                        quiet=a.quiet, make_sim=make_sim, realtime=a.realtime,
                        ground_live=a.port if a.ground == "live" else None,
                        make_pilot=lambda did: make_pilot(name, pcfg), qgc_live=a.qgc_live)
    finally:
        if "ctl" in holder:
            holder["ctl"].stop()
            (out / "sim_summary.json").write_text(json.dumps(holder["ctl"].summary(), indent=2))
    s.pop("skills", None)
    print(json.dumps({k: s[k] for k in ("episode", "end", "t_end", "robustness", "satisfied", "violated",
                                        "decisions", "cost_usd", "wall_s", "drop_err")}, indent=1, default=str))
    print(f"episode: {out}")
    if a.video:
        from avians_viz import render
        print("video:", render(out, m.world, speed=a.speed))
    if a.qgc:
        _qgc(out, m.world)
    return 0


def cmd_rollouts(a) -> int:
    _load_env()
    from .rollouts import run_many, table
    m = Mission.load(a.mission, a.envelope)
    name = a.pilot or load_config("pilots.yaml").get("default", "scripted")
    res = run_many(m.path, name, _seeds(a.seeds), _out(a, m.name, name), a.workers, a.envelope)
    print(table(res))
    return 0


def _world_of(ep: Path):
    meta = json.loads((ep / "meta.json").read_text())
    from avians_env import WorldSpec
    return WorldSpec.load(repo_root() / "worlds" / meta["world"]), meta


def cmd_score(a) -> int:
    from avians_spec import Trace
    ep = Path(a.episode)
    world, meta = _world_of(ep)
    m = Mission.load(Path("missions") / f"{meta['mission']}.yaml")
    rows = [json.loads(line) for line in open(ep / "signals.jsonl")]
    tr = Trace.from_samples([r["t"] for r in rows], rows, m.spec.dt, m.spec.registry.defaults())
    print(json.dumps(m.spec.score(tr), indent=1))
    return 0


def cmd_video(a) -> int:
    from avians_viz import render, summary_png
    ep = Path(a.episode)
    world, _ = _world_of(ep)
    print(summary_png(ep, world))
    print(render(ep, world, speed=a.speed))
    return 0


def _qgc(ep: Path, world) -> None:
    from avians_viz import Episode, flight_kml, tlog, world_kml
    (ep / "world.kml").write_text(world_kml(world))
    (ep / "flight.kml").write_text(flight_kml(Episode.load(ep), world))
    tlog(Episode.load(ep), world, ep / "flight.tlog")
    print(f"QGroundControl: {ep / 'flight.tlog'} (Analyze Tools > Log Replay); {ep / 'world.kml'} "
          f"(Plan view > import KML)")


def cmd_qgc(a) -> int:
    ep = Path(a.episode)
    world, _ = _world_of(ep)
    _qgc(ep, world)
    return 0


def cmd_gazebo_replay(a) -> int:
    from avians_sim.gazebo import GazeboControl
    ep = Path(a.episode)
    world, _ = _world_of(ep)
    ctl = GazeboControl(load_config("aas.yaml").get("sim", {}), ep)
    print(json.dumps(ctl.replay(ep, world, a.speed), indent=1))
    return 0


def cmd_latency_fit(a) -> int:
    from .latency import fit
    by_model: dict[str, list[float]] = {}
    for tj in Path(a.dir).rglob("trajectory.json"):
        for step in json.loads(tj.read_text())["content"]:
            d = step.get("details", {})
            if step.get("class_") == "api_action" and d.get("wall_s") and d.get("model") not in ("scripted", "router"):
                by_model.setdefault(d["model"], []).append(d["wall_s"])
    for model, xs in by_model.items():
        f = fit(xs, floor_s=0.0)
        print(f"{model}: {{median_s: {f.median_s:.2f}, sigma: {f.sigma:.2f}, floor_s: 0.0}}   # n={len(xs)}")
    return 0


def cmd_doctor(a) -> int:
    import importlib
    import shutil
    ok = True
    for mod in ("avians_env", "avians_spec", "avians_perception", "avians_skills", "avians_safety",
                "avians_sim", "avians_adp", "avians_viz", "avians_agent"):
        try:
            importlib.import_module(mod)
            print(f"ok   {mod}")
        except Exception as e:
            ok = False
            print(f"FAIL {mod}: {e}")
    for mod, why in (("anthropic", "the Haiku teacher"), ("rclpy", "the Gazebo tier (inside the container)"),
                     ("pymavlink", "QGC .tlog export")):
        try:
            importlib.import_module(mod)
            print(f"ok   {mod}")
        except Exception:
            print(f"--   {mod} not installed ({why})")
    print("ok   ffmpeg" if shutil.which("ffmpeg") else "--   ffmpeg not on PATH (videos use imageio-ffmpeg)")
    _load_env()
    print("ok   ANTHROPIC_API_KEY set" if os.environ.get("ANTHROPIC_API_KEY") else
          "--   ANTHROPIC_API_KEY not set (.env): only --pilot scripted works")
    print(f"repo root: {repo_root()}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="avians", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check")
    p.add_argument("mission")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--envelope")
    p = sub.add_parser("fly")
    p.add_argument("mission")
    p.add_argument("--pilot")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tier", choices=["fast", "gazebo"], default="fast")
    p.add_argument("--envelope")
    p.add_argument("--out")
    p.add_argument("--video", action="store_true")
    p.add_argument("--speed", type=float, default=20.0)
    p.add_argument("--qgc", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--realtime", action="store_true", help="fast tier slowed to real time (for a live operator)")
    p.add_argument("--ground", choices=["scripted", "live"], default="scripted",
                   help="live: open the operator console (http://localhost:PORT)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--qgc-live", metavar="HOST:PORT", help="stream MAVLink to QGroundControl while flying "
                   "(QGC listens on 14550); use with --realtime")
    p = sub.add_parser("rollouts")
    p.add_argument("mission")
    p.add_argument("--pilot")
    p.add_argument("--seeds", default="0-9")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--envelope")
    p.add_argument("--out")
    for name in ("score", "video", "qgc", "gazebo-replay"):
        p = sub.add_parser(name)
        p.add_argument("episode")
        p.add_argument("--speed", type=float, default=20.0 if name == "video" else 1.0)
    p = sub.add_parser("latency-fit")
    p.add_argument("dir")
    sub.add_parser("doctor")
    a = ap.parse_args(argv)
    fn = {"check": cmd_check, "fly": cmd_fly, "rollouts": cmd_rollouts, "score": cmd_score,
          "video": cmd_video, "qgc": cmd_qgc, "gazebo-replay": cmd_gazebo_replay,
          "latency-fit": cmd_latency_fit, "doctor": cmd_doctor}[a.cmd]
    return fn(a)


if __name__ == "__main__":
    sys.exit(main())
