"""Many episodes in parallel (one process per episode; the fast tier is
single-threaded numpy, so processes scale with cores)."""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def _one(job):
    mission_path, pilot_name, seed, out, envelope = job
    from .loop import run_episode
    from .mission import Mission, load_config
    from .pilots import make_pilot
    m = Mission.load(mission_path, envelope)
    pcfg = load_config("pilots.yaml")
    try:
        return run_episode(m, make_pilot(pilot_name, pcfg), seed, Path(out) / f"s{seed:04d}",
                           load_config("agent.yaml"), load_config("latency.yaml"),
                           make_pilot=lambda did: make_pilot(pilot_name, pcfg))
    except Exception as e:                          # one bad episode must not stop a sweep
        return {"seed": seed, "error": f"{type(e).__name__}: {e}"}


def run_many(mission_path, pilot_name: str, seeds, out_dir, workers: int = 4, envelope=None) -> list[dict]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(str(mission_path), pilot_name, s, str(out), envelope) for s in seeds]
    if workers <= 1:
        results = [_one(j) for j in jobs]
    else:
        with ProcessPoolExecutor(workers) as pool:
            results = list(pool.map(_one, jobs))
    (out / "results.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in results))
    return results


def table(results: list[dict]) -> str:
    ok = [r for r in results if "error" not in r]
    rows = ["| seed | end | robustness | violated | decisions | found at | drop err | $ |",
            "|---|---|---|---|---|---|---|---|"]
    for r in ok:
        rows.append(f"| {r['seed']} | {r['end']} | {r['robustness']} | {', '.join(r['violated']) or '-'} | "
                    f"{r['decisions']} | {r['found_t'] if r['found_t'] is None else round(r['found_t'])} | "
                    f"{'-' if r['drop_err'] is None else round(r['drop_err'], 1)} | {r['cost_usd']} |")
    sat = sum(1 for r in ok if r["satisfied"])
    rows.append(f"\n{sat}/{len(ok)} satisfied; {len(results) - len(ok)} errored")
    for r in results:
        if "error" in r:
            rows.append(f"- seed {r['seed']}: {r['error']}")
    return "\n".join(rows)
