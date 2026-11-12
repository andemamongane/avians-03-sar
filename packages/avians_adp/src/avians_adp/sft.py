"""ADP -> training examples.

Two datasets come out of the same trajectories:

1. Pilot SFT (behaviour cloning the teacher): `pilot_examples`. One example
   per decision that has an action:
       input  = system prompt + tools + the decision prompt at that point
                (self-contained: window + structured memory, so no earlier
                turns are needed)
       target = the teacher's tool call, with `reason` (and `notes`) put
                back into its arguments: the student's tool schema has them
                as arguments, while the record keeps them apart
                (description, details.notes)
   The example is layout-neutral; formats.py writes the TRL and
   LLaMA-Factory layouts. Nothing is filtered here: every decision with an
   action gives one example, and `meta` carries what the filters
   (Phase 3) need: error, model, accepted, robustness, ...

2. Router examples (System One: commit / local / escalate):
   `router_examples`. TODO (Phase 4).
"""
from __future__ import annotations

import re
from typing import Any

from .decisions import decisions
from .schema import Trajectory

_EXTRA_CALLS = re.compile(r"\s*\(ignored (\d+) more call\(s\)\)\s*$")


def openai_tools(tools: list[dict]) -> list[dict]:
    """Anthropic tool definitions (name, description, input_schema) -> the
    OpenAI function layout chat templates and TRL expect. Tools already in
    the OpenAI layout pass through."""
    out = []
    for t in tools:
        if t.get("type") == "function" and "function" in t:
            out.append(t)
            continue
        out.append({"type": "function", "function": {
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t.get("input_schema") or t.get("parameters") or {"type": "object", "properties": {}},
        }})
    return out


def clean_reason(reason: str | None) -> tuple[str, int]:
    """Strip the loop's ' (ignored N more call(s))' suffix: (reason, N)."""
    reason = reason or ""
    m = _EXTRA_CALLS.search(reason)
    if not m:
        return reason, 0
    return reason[:m.start()], int(m.group(1))


def target_arguments(kwargs: dict, reason: str, notes: str | None) -> dict[str, Any]:
    """The arguments the teacher actually sent: skill args + reason + notes.

    `reason` is optional in the tool schema and split_common records a
    missing one as "", so an empty reason is left out (the teacher never
    sent it). `notes` is recorded as None when absent; "" is a real value
    (it clears the notes) and is kept."""
    args = dict(kwargs)
    if reason:
        args["reason"] = reason
    if notes is not None:
        args["notes"] = notes
    return args


def decision_kind(function: str, reason: str, details: dict) -> str:
    """Whether the recorded call is one the pilot really made.

        call            a real tool call (including a real `keep`)
        api_error       the API failed; the loop recorded keep
        no_tool_call    the model replied without calling a tool; the loop recorded keep
        router_commit   the router chose commit; no model was asked
    Only "call" is the teacher's behaviour; Phase 3 filters on this.
    (Matches avians_agent.pilots.AnthropicPilot.decide and loop._decide.)"""
    if details.get("error"):
        return "api_error"
    if function == "keep" and (details.get("model") == "router" or details.get("by") == "commit"):
        return "router_commit"
    if function == "keep" and reason == "no tool call":
        return "no_tool_call"
    return "call"


def pilot_examples(traj: Trajectory) -> list[dict]:
    """One layout-neutral example per decision that has an action.

    Keys: id, system, tools (OpenAI layout), prompt, call {name, arguments},
    text (the pilot's visible text, or ""), thinking (its summary, or ""),
    state (the structured snapshot at t_obs, used by filters.thin_keeps),
    meta {episode, seed, pilot, kind, model, t, decision, by, error, stop,
    accepted, robustness, satisfied, extra_calls}. `kind` (decision_kind)
    says whether the call is one the pilot really made.
    """
    d = traj.details
    system = d.get("system", "")
    tools = openai_tools(d.get("tools", []))
    meta_ep = d.get("meta", {})
    summary = d.get("summary", {})
    score = d.get("score", {})
    robustness = summary.get("robustness", score.get("robustness"))
    satisfied = summary.get("satisfied", score.get("satisfied"))
    out = []
    for dec in decisions(traj):
        a = dec.action
        if a is None:
            continue
        n = dec.n if dec.n is not None else dec.index + 1
        reason, extra = clean_reason(a.description)
        thinking = a.reasoning_content or (dec.message.reasoning_content if dec.message else None) or ""
        det = a.details
        out.append({
            "id": f"{traj.id}#d{n}",
            "system": system,
            "tools": tools,
            "prompt": dec.prompt.content,
            "call": {"name": a.function, "arguments": target_arguments(a.kwargs, reason, det.get("notes"))},
            "text": dec.message.content if dec.message else "",
            "thinking": thinking,
            "state": det.get("state"),          # for keep-thinning; never written to a file
            "meta": {
                "episode": traj.id, "seed": meta_ep.get("seed"), "pilot": meta_ep.get("pilot"),
                "kind": decision_kind(a.function, reason, det),
                "model": det.get("model"), "t": det.get("t_obs", det.get("t")), "decision": n,
                "by": det.get("by"), "error": det.get("error") or "",
                "stop": (det.get("usage") or {}).get("stop"), "accepted": dec.accepted,
                "robustness": robustness, "satisfied": satisfied, "extra_calls": extra,
            },
        })
    return out


def router_examples(teacher: Trajectory, student_calls: list[dict]) -> list[dict]:
    raise NotImplementedError("McGlashan: router labels from teacher/student comparisons (Phase 4)")
