"""Checks every saved trajectory must pass.

    validate(traj)        -> list of problems; empty means it passes
    validate_dir(root)    -> {episode_dir: problems} for every trajectory.json under root
    to_upstream(traj)     -> the record as a plain dict in upstream's strict
                             shape (no per-step details), for a check against
                             neulab/agent-data-protocol's own models

Two layers of rules:
  upstream (ADP 1.3.1)    enforced by the pydantic models in schema.py:
                          source values, tool-call links, supported version.
                          Re-running the model on its own dump reports them here.
  avians                  the decision pattern (decisions.py), decision numbers
                          increasing, t_obs <= t_apply, available_apis covering
                          every function called, the trajectory details every
                          episode has.

`python -m avians_adp validate <dir>` prints the problems per episode.
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from .decisions import decisions
from .schema import ApiAction, Trajectory

REQUIRED_DETAILS = ("system", "tools", "meta", "score", "summary")


def validate(traj: Trajectory) -> list[str]:
    problems: list[str] = []
    try:                                       # the upstream rules, on the record as it will be saved
        Trajectory.model_validate(traj.model_dump())
    except ValidationError as e:
        problems += [err["msg"].removeprefix("Value error, ") for err in e.errors()]
    for k in REQUIRED_DETAILS:
        if k not in traj.details:
            problems.append(f"details has no {k!r}")
    if traj.available_apis is not None:
        apis = set(traj.available_apis)
        missing = sorted({s.function for s in traj.content if isinstance(s, ApiAction)} - apis)
        if missing:
            problems.append(f"available_apis does not list called functions {missing}")
    try:
        decs = decisions(traj, strict=True)
    except ValueError as e:
        problems.append(str(e))
        decs = decisions(traj)
    last_n = 0
    for d in decs:
        tag = f"decision {d.n if d.n is not None else '#' + str(d.index)}"
        if d.n is not None:
            if d.n <= last_n:
                problems.append(f"{tag}: decision number does not increase (after {last_n})")
            last_n = d.n
        if not d.finished and d.index != len(decs) - 1:
            problems.append(f"{tag}: unfinished (no {'action' if d.action is None else 'result'}) before the last one")
        a = d.action
        if a is not None:
            det = a.details
            if "t_obs" in det and "t_apply" in det and det["t_obs"] > det["t_apply"] + 1e-9:
                problems.append(f"{tag}: t_obs {det['t_obs']} > t_apply {det['t_apply']}")
            if d.n is not None and det.get("decision") not in (None, d.n):
                problems.append(f"{tag}: action carries decision {det.get('decision')}")
            if d.result is not None and "accepted" not in d.result.details:
                problems.append(f"{tag}: result has no 'accepted'")
    return problems


def validate_dir(root: str | Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for p in sorted(Path(root).rglob("trajectory.json")):
        try:
            out[str(p.parent)] = validate(Trajectory.model_validate_json(p.read_text()))
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            out[str(p.parent)] = [f"does not load: {e}"]
    return out


def to_upstream(traj: Trajectory) -> dict:
    """The record without the avians extension, as upstream's strict models accept it."""
    d = traj.model_dump()
    for step in d["content"]:
        step.pop("details", None)
    return d

