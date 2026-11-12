"""Run a student offline over a teacher's episode: what would it have called?

    calls = run_student(traj, decide)
    write_calls(episode_dir, calls)        -> <episode_dir>/student_calls.jsonl
    read_calls(episode_dir)                -> the same list, or None

`decide(system, prompt, tools, state)` is any function returning either an
object with `.tool` and `.args` (avians_agent's Decision; `.error` and
`.reason` are read too) or a dict {"tool", "args"} (optionally "error",
"no_tool_call"). It gets what the teacher got at that decision: the system
prompt, the decision prompt, the tool definitions as recorded (Anthropic
layout) plus `extra_tools` (the CLI adds `escalate`, which a student always
has in flight), and the state snapshot. `reason` and `notes` are dropped from
the arguments (routing ignores them).

Use a fresh decide per episode: a pilot may remember earlier decisions
(ScriptedPilot remembers the regions it searched), and memory from another
episode would make it disagree for the wrong reason.

Only the decisions whose label depends on the student are run: the
teacher's real, non-keep calls. A teacher keep is `commit` whatever the
student says, and the other decisions are not labelled at all, so asking
the student there would only cost money.

Each row:
    {"episode", "decision", "tool", "args"}                 a real call
    + "no_tool_call": true, tool None                         it replied without a tool (labelled escalate)
    + "error": "...", tool None                               the call failed (API error, exception):
                                                              says nothing about the student, so skipped

    python -m avians_adp.student_calls <episodes dir> --pilot scripted [--force]

runs a pilot from configs/pilots.yaml (needs avians_agent). A pilot that
calls an API costs money per decision: check with Andema before large runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

from .decisions import decisions
from .schema import Trajectory
from .sft import decision_kind

FILE = "student_calls.jsonl"


def _normalise(result) -> dict:
    get = result.get if isinstance(result, dict) else (lambda k, d=None: getattr(result, k, d))
    tool, args = get("tool"), dict(get("args") or {})
    row: dict = {"tool": tool, "args": {k: v for k, v in args.items() if k not in ("reason", "notes")}}
    if get("error"):                                     # AnthropicPilot returns keep + error on API failure
        row.update(tool=None, args={}, error=str(get("error")))
    elif get("no_tool_call") or (tool == "keep" and get("reason") == "no tool call"):
        row.update(tool=None, args={}, no_tool_call=True)
    return row


def needs_student(traj: Trajectory) -> list:
    """The decisions whose label depends on the student: real teacher calls other than keep."""
    routed = bool(traj.details.get("meta", {}).get("router"))
    out = []
    for dec in decisions(traj):
        a = dec.action
        if a is None or a.function == "keep":
            continue
        if decision_kind(a.function, a.description or "", a.details, routed) != "call":
            continue
        out.append(dec)
    return out


def run_student(traj: Trajectory, decide: Callable[[str, str, list, dict], object], *,
                model: str | None = None, extra_tools: list[dict] | None = None) -> list[dict]:
    d = traj.details
    system, tools = d.get("system", ""), list(d.get("tools", []))
    names = {t.get("name") for t in tools}
    tools += [t for t in extra_tools or [] if t.get("name") not in names]
    out = []
    for dec in needs_student(traj):
        n = dec.n if dec.n is not None else dec.index + 1
        state = dec.action.details.get("state") or {}
        row = {"episode": traj.id, "decision": n}
        try:
            row.update(_normalise(decide(system, dec.prompt.content, tools, state)))
        except Exception as e:  # noqa: BLE001  (one bad decision must not lose the episode's other calls)
            row.update(tool=None, args={}, error=f"{type(e).__name__}: {e}")
        if model:
            row["model"] = model
        out.append(row)
    return out


def write_calls(episode_dir: str | Path, calls: list[dict]) -> Path:
    p = Path(episode_dir) / FILE
    p.write_text("".join(json.dumps(c, default=str) + "\n" for c in calls))
    return p


def read_calls(episode_dir: str | Path) -> list[dict] | None:
    p = Path(episode_dir) / FILE
    if not p.exists():
        return None
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def pilot_decide(name: str) -> tuple[Callable, str]:
    """A decide function for a new instance of a pilot named in
    configs/pilots.yaml. Call it once per episode."""
    from avians_agent.mission import load_config
    from avians_agent.pilots import DecisionRequest, make_pilot
    pilot = make_pilot(name, load_config("pilots.yaml"))

    def decide(system, prompt, tools, state):
        return pilot.decide(DecisionRequest(system, prompt, tools, state))
    return decide, name


def escalate_tool() -> list[dict]:
    """The `escalate` tool exactly as the loop gives it to a student in flight."""
    from avians_agent.prompts import tool_definitions
    return [t for t in tool_definitions(["escalate"]) if t["name"] == "escalate"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m avians_adp.student_calls",
                                 description="Run a student pilot offline over teacher episodes.")
    ap.add_argument("episodes", help="folder searched recursively for trajectory.json")
    ap.add_argument("--pilot", required=True, help="pilot name from configs/pilots.yaml")
    ap.add_argument("--force", action="store_true", help=f"replace an existing {FILE}")
    a = ap.parse_args(argv)
    try:
        pilot_decide(a.pilot)                  # fail early on an unknown pilot
        extra = escalate_tool()
    except (ImportError, KeyError, ValueError) as e:
        print(f"error: cannot make pilot {a.pilot!r}: {e}", file=sys.stderr)
        return 1
    done, failed = 0, 0
    for p in sorted(Path(a.episodes).rglob("trajectory.json")):
        ep = p.parent
        if (ep / FILE).exists() and not a.force:
            print(f"skipped {ep}: {FILE} exists (--force replaces it)", file=sys.stderr)
            continue
        traj = Trajectory.model_validate_json(p.read_text())
        decide, model = pilot_decide(a.pilot)  # a fresh pilot per episode: no memory carried over
        calls = run_student(traj, decide, model=model, extra_tools=extra)
        write_calls(ep, calls)
        errors = sum(1 for c in calls if c.get("error"))
        silent = sum(1 for c in calls if c.get("no_tool_call"))
        failed += errors
        done += 1
        print(f"{ep}: {len(calls)} calls" + (f", {errors} FAILED (skipped when labelling)" if errors else "")
              + (f", {silent} without a tool call" if silent else ""))
        if calls and errors == len(calls):
            print(f"error: every call failed in {ep} (first: {calls[0]['error']}); stopping", file=sys.stderr)
            return 1
    if failed:
        print(f"warning: {failed} student calls failed; rerun with --force once the pilot works", file=sys.stderr)
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
