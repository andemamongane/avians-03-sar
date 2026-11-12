"""Which episodes and decisions go into a training set.

Three stages, applied by build.py in this order:

    EpisodeFilter     whole episodes, from the score and meta: satisfied,
                      robustness, named clauses, pilot, tier
    DecisionFilter    single decisions, from each example's meta: drop the
                      keeps the teacher never chose (meta.kind), rejected
                      calls, calls with no result; optionally only some models
    thin_keeps        train split only: runs of near-identical `keep`
                      decisions are thinned to one in k

Every stage reports what it removed and why, so the build can count it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .schema import Trajectory

DROPPABLE = ("api_error", "no_tool_call", "router_commit", "rejected", "unanswered")


def number(v) -> float | None:
    """A robustness value as a float. The scorer writes infinities as the
    strings "inf"/"-inf" (JSON has none); anything unreadable is None."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


@dataclass
class EpisodeFilter:
    satisfied: bool = False                     # keep only episodes that satisfied the whole spec
    min_robustness: float | None = None         # keep only episodes with robustness >= this
    require_clauses: tuple[str, ...] = ()       # these clauses satisfied, with robustness >= clause_margin
    clause_margin: float = 0.0
    pilots: tuple[str, ...] = ()                # meta.pilot in this set (empty: any)
    tiers: tuple[str, ...] = ()                 # meta.tier in this set (empty: any)

    def reject_reason(self, traj: Trajectory) -> str | None:
        """None if the episode passes, else the first failed test."""
        d = traj.details
        summary, score, meta = d.get("summary", {}), d.get("score", {}), d.get("meta", {})
        if self.pilots and meta.get("pilot") not in self.pilots:
            return "pilot"
        if self.tiers and meta.get("tier") not in self.tiers:
            return "tier"
        if self.satisfied and not summary.get("satisfied", score.get("satisfied")):
            return "not satisfied"
        if self.min_robustness is not None:
            r = number(summary.get("robustness", score.get("robustness")))
            if r is None or r < self.min_robustness:
                return "robustness"
        if self.require_clauses:
            clauses = {c.get("clause"): c for c in score.get("clauses", [])}
            for name in self.require_clauses:
                c = clauses.get(name)
                r = None if c is None else number(c.get("robustness"))
                if c is None or not c.get("satisfied") or r is None or r < self.clause_margin:
                    return f"clause {name}"
        return None

    def check_clause_names(self, trajectories: list[Trajectory]) -> None:
        """A required clause that no episode has is a typo, not a filter."""
        if not self.require_clauses or not trajectories:
            return
        known = {c.get("clause") for t in trajectories for c in t.details.get("score", {}).get("clauses", [])}
        unknown = [n for n in self.require_clauses if n not in known]
        if unknown:
            raise ValueError(f"unknown clause(s) {unknown}; the episodes have {sorted(k for k in known if k)}")


@dataclass
class DecisionFilter:
    allow: tuple[str, ...] = ()                 # DROPPABLE kinds to keep anyway
    models: tuple[str, ...] = ()                # meta.model in this set (empty: any)

    def __post_init__(self):
        bad = [a for a in self.allow if a not in DROPPABLE]
        if bad:
            raise ValueError(f"unknown allow value(s) {bad}; choose from {DROPPABLE}")

    def reject_reason(self, ex: dict) -> str | None:
        m = ex["meta"]
        kind = m.get("kind", "call")
        if kind != "call" and kind not in self.allow:
            return kind
        if m.get("accepted") is False and "rejected" not in self.allow:
            return "rejected"
        if m.get("accepted") is None and "unanswered" not in self.allow:
            return "unanswered"
        if self.models and m.get("model") not in self.models:
            return "model"
        return None


BATTERY_BAND = 25.0     # percent; see keep_signature


def keep_signature(state: dict | None, battery_band: float = BATTERY_BAND) -> tuple | None:
    """What makes two keep decisions near-identical: the same skill in the
    same status, the same mission facts, the same closed regions and the
    battery in the same band (battery_band percent; 0 leaves battery out).
    None (never thinned) without a state.

    The band is 25 %, not 10 %: the battery drains through every long keep
    run, and on scripted seeds 0-5 10 % bands cut runs into pieces of 2-3,
    so keep_every barely mattered (k=3 and k=10 removed 28 and 29 of 91
    keeps). With 25 % bands they remove 38 and 43."""
    if not state:
        return None
    belief = state.get("belief") or {}
    battery = number(state.get("battery"))
    return (
        state.get("skill"), state.get("skill_status"), bool(belief.get("found")),
        bool(state.get("payload")), bool(state.get("released")), bool(state.get("help_arrived")),
        tuple(sorted(str(r.get("id")) for r in state.get("regions") or [] if r.get("closed"))),
        None if battery is None or battery_band <= 0 else int(battery // battery_band),
    )


def thin_keeps(examples: list[dict], every: int, battery_band: float = BATTERY_BAND) -> tuple[list[dict], int]:
    """Within each episode, in decision order, a run of consecutive `keep`
    examples with one signature keeps its 1st, (1+every)th, (1+2*every)th, ...
    Any other call among `examples`, or a change of signature, ends the run.
    The build thins after the decision filters, so a dropped decision (an
    API-error keep, a rejected call) does not end a run. every=1 keeps all.
    Returns (kept examples, number removed). Order is preserved."""
    if every < 1:
        raise ValueError("keep_every must be >= 1")
    out, removed = [], 0
    run_key, run_pos = None, 0
    for ex in examples:
        episode = ex["meta"]["episode"]
        sig = keep_signature(ex.get("state"), battery_band) if ex["call"]["name"] == "keep" else None
        key = (episode, sig) if sig is not None else None
        if key is not None and key == run_key:
            run_pos += 1
        else:
            run_key, run_pos = key, 0
        if key is not None and run_pos % every != 0:
            removed += 1
            continue
        out.append(ex)
    return out, removed


@dataclass
class Counts:
    """What each stage removed, by reason."""
    episodes: dict[str, int] = field(default_factory=dict)
    decisions: dict[str, int] = field(default_factory=dict)
    thinned: int = 0

    def add(self, table: dict[str, int], reason: str) -> None:
        table[reason] = table.get(reason, 0) + 1
