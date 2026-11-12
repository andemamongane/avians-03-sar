"""Router examples (System One): at each decision point, should the drone
    commit    keep its current skill without asking a model,
    local     ask the onboard student, or
    escalate  ask the remote teacher?

Labels come from what the teacher did and what a student would have done on
the same prompt (student_calls.py runs a student offline over a teacher's
episode):

    commit     the teacher called `keep`
    local      otherwise, and the student's call matches the teacher's
    escalate   otherwise: the student's call differs, the student asked to
               escalate, it replied without calling a tool, or there is no
               student call for that decision

A student call that failed for an infrastructure reason (an API error) says
nothing about the student: that decision is skipped, not labelled.

Only the teacher's decisions are labelled. In an episode flown with a router
the student made the `local` ones (decision_kind "student"); they are
skipped, as are router commits and the teacher's API-error and no-tool-call
keeps.

A match is the same tool and the same arguments, ignoring `reason` and
`notes`. Strings must be equal; numbers may differ by the tolerance for that
argument name in TOLERANCES (0 for names not listed). An argument present on
one side only is a mismatch.

Each example also carries the router's input: the state snapshot and the
event kinds since the previous decision, and `text` = features(state,
events) when a features function is given (avians_agent.router.features is
the one the live router reads). The event kinds come from the prompt's
`events` detail, which the recorder writes; records made before it existed
fall back to a time window over events.jsonl and are flagged
events_exact=False (a time window cannot place an event stamped at the
decision time on the right side).
"""
from __future__ import annotations

import math
from collections.abc import Callable
from types import SimpleNamespace

from .decisions import decisions
from .schema import Trajectory
from .sft import decision_kind

CHOICES = ("commit", "local", "escalate")
IGNORED = ("reason", "notes")

# Absolute tolerance per numeric argument name, in the argument's own unit.
# Copied into the dataset manifest (Phase 5); change it there, not per call.
TOLERANCES: dict[str, float] = {
    "east": 10.0, "north": 10.0, "datum_east": 10.0, "datum_north": 10.0,   # m
    "alt": 5.0, "drop_alt": 5.0, "radius": 5.0, "standoff": 5.0,           # m
    "max_sigma": 5.0,                                                       # m
    "speed": 1.0,                                                           # m/s
    "seconds": 10.0, "timeout_s": 10.0,                                     # s
    "until_pct": 5.0,                                                       # % battery
    "overlap": 0.05,                                                        # fraction
}


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def args_match(teacher: dict, student: dict, tolerances: dict[str, float] | None = None) -> tuple[bool, str]:
    """(match, why) for two argument dicts, ignoring reason and notes."""
    tol = TOLERANCES if tolerances is None else tolerances
    a = {k: v for k, v in (teacher or {}).items() if k not in IGNORED}
    b = {k: v for k, v in (student or {}).items() if k not in IGNORED}
    if set(a) != set(b):
        return False, f"arguments differ: {sorted(set(a) ^ set(b))}"
    for k in sorted(a):
        x, y = a[k], b[k]
        if _is_number(x) and _is_number(y):
            if not (math.isfinite(x) and math.isfinite(y)) or abs(x - y) > tol.get(k, 0.0) + 1e-9:
                return False, f"{k}: {x} vs {y}"
        elif isinstance(x, bool) != isinstance(y, bool) or x != y:     # True == 1 in Python: not a match here
            return False, f"{k}: {x!r} vs {y!r}"
    return True, "match"


def label(teacher_call: dict, student_call: dict | None,
          tolerances: dict[str, float] | None = None) -> tuple[str, str]:
    """(label, why). Calls are {"tool": name, "args": {...}}."""
    if teacher_call["tool"] == "keep":
        return "commit", "teacher kept going"
    if student_call is not None and student_call.get("no_tool_call"):
        return "escalate", "student made no tool call"
    if student_call is None or student_call.get("tool") is None:
        return "escalate", "no student call"
    if student_call["tool"] == "escalate":
        return "escalate", "student escalated"
    if student_call["tool"] != teacher_call["tool"]:
        return "escalate", f"tool: {teacher_call['tool']} vs {student_call['tool']}"
    ok, why = args_match(teacher_call.get("args", {}), student_call.get("args", {}), tolerances)
    return ("local", why) if ok else ("escalate", why)


def event_kinds(traj: Trajectory, events: list[dict] | None = None) -> dict[int, tuple[list[str], bool]]:
    """{decision index: (event kinds since the previous prompt, exact?)}.

    Exact from the prompt's `events` detail; otherwise from `events` (the
    rows of events.jsonl) in the window (previous prompt t, this prompt t].
    """
    out: dict[int, tuple[list[str], bool]] = {}
    prev_t = float("-inf")
    for d in decisions(traj):
        det = d.prompt.details
        t = float(det.get("t", 0.0))
        if isinstance(det.get("events"), list):
            out[d.index] = (list(det["events"]), True)
        else:
            kinds = [e["kind"] for e in (events or []) if prev_t < float(e.get("t", 0.0)) <= t]
            out[d.index] = (kinds, False)
        prev_t = t
    return out


def router_examples(teacher: Trajectory, student_calls: list[dict], *,
                    features: Callable[[dict, list], str] | None = None,
                    tolerances: dict[str, float] | None = None,
                    events: list[dict] | None = None,
                    models: tuple[str, ...] = ()) -> tuple[list[dict], dict[str, int]]:
    """One router example per teacher decision; returns (examples, skipped).

    student_calls: [{"decision": n, "tool": name or None, "args": {...}}],
    matched to the teacher's decisions by number. Skipped, and counted by
    reason: decisions without an action, keeps the teacher never chose
    (decision_kind != "call": API error, no tool call, router commit), and,
    when `models` is given, decisions made by another model (a teacher
    episode flown with a router holds the student's decisions too). Keeps
    are never thinned: every one is a `commit` example.
    """
    meta_ep = teacher.details.get("meta", {})
    routed = bool(meta_ep.get("router"))
    by_n = {c["decision"]: c for c in student_calls if c.get("decision") is not None}
    kinds = event_kinds(teacher, events)
    out, skipped = [], {}
    for d in decisions(teacher):
        a = d.action
        if a is None:
            skipped["no action"] = skipped.get("no action", 0) + 1
            continue
        det = a.details
        kind = decision_kind(a.function, a.description or "", det, routed)
        if kind != "call":
            skipped[kind] = skipped.get(kind, 0) + 1
            continue
        if models and det.get("model") not in models:
            skipped["model"] = skipped.get("model", 0) + 1
            continue
        n = d.n if d.n is not None else d.index + 1
        teacher_call = {"tool": a.function, "args": dict(a.kwargs)}
        student_call = by_n.get(n)
        if student_call is not None and student_call.get("error") and a.function != "keep":
            skipped["student error"] = skipped.get("student error", 0) + 1
            continue
        lab, why = label(teacher_call, student_call, tolerances)
        state = det.get("state") or {}
        ev, exact = kinds.get(d.index, ([], False))
        text = features(state, [SimpleNamespace(kind=k) for k in ev]) if features and state else None
        out.append({
            "id": f"{teacher.id}#d{n}",
            "label": lab,
            "options": list(CHOICES),
            "text": text,
            "state": state,
            "events": ev,
            "meta": {"episode": teacher.id, "seed": meta_ep.get("seed"), "pilot": meta_ep.get("pilot"),
                     "decision": n, "t": det.get("t_obs", det.get("t")), "label": lab, "why": why,
                     "teacher": teacher_call,
                     "student": None if student_call is None else
                     {"tool": student_call.get("tool"), "args": student_call.get("args", {})},
                     "events_exact": exact},
        })
    return out, skipped
