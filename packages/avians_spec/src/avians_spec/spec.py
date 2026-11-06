"""A mission's specification: named STL clauses, all of which must hold.

In a mission file:

    spec:
      dt: 0.5
      clauses:
        no_fly:   {formula: "G[0,1500] !in(nofly_power)",  says: "never enter the power line corridor"}
        battery:  {formula: "G[0,1500] battery >= 10",     says: "keep at least 10 % battery"}

The pilot sees every clause (formula and plain words) before takeoff. The
score is the minimum robustness over clauses (their conjunction), reported per
clause so a failure says which promise was broken.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import stl
from .signals import Registry


@dataclass
class Clause:
    name: str
    formula: str
    says: str = ""
    node: stl.Node | None = None


@dataclass
class MissionSpec:
    clauses: list[Clause]
    registry: Registry
    dt: float = 0.5
    regions: dict = field(default_factory=dict)       # id -> object with .robustness(e, n, a)

    @classmethod
    def from_dict(cls, d: dict, registry: Registry, regions: dict) -> "MissionSpec":
        clauses = []
        for name, c in d["clauses"].items():
            c = {"formula": c} if isinstance(c, str) else c
            node = stl.parse(c["formula"], registry)
            missing = node.regions() - set(regions)
            if missing:
                raise stl.STLSyntaxError(f"clause {name}: unknown region(s) {sorted(missing)}")
            clauses.append(Clause(name, c["formula"], c.get("says", ""), node))
        return cls(clauses, registry, float(d.get("dt", 0.5)), regions)

    @property
    def horizon(self) -> float:
        return max(c.node.horizon() for c in self.clauses)

    def describe(self) -> str:
        """For the pilot's prompt."""
        lines = [f"SPEC (signal temporal logic; all clauses must hold; times in s from takeoff "
                 f"command; horizon {self.horizon:g} s):"]
        for c in self.clauses:
            lines.append(f"- {c.name}: {c.formula}" + (f"   -- {c.says}" if c.says else ""))
        return "\n".join(lines)

    def score(self, trace: stl.Trace) -> dict:
        """Robustness of a finished flight."""
        rows = []
        for c in self.clauses:
            r = stl.robustness(c.node, trace, self.regions)
            rows.append({"clause": c.name, "formula": c.formula, "robustness": stl.fmt(r),
                         "satisfied": r > 0})
        overall = min(_num(r["robustness"]) for r in rows)
        return {"robustness": stl.fmt(overall), "satisfied": overall > 0, "clauses": rows,
                "horizon_s": self.horizon, "flight_end_s": round((trace.n - 1) * trace.dt, 2),
                "dt_s": trace.dt}

    def score_online(self, trace: stl.Trace, max_speed: float, only=None) -> dict:
        """Robust satisfaction intervals for the flight so far."""
        rows = []
        for c in self.clauses:
            if only is not None and c.name not in only:
                continue
            lo, hi = stl.interval(c.node, trace, self.regions, self.registry, max_speed)
            rows.append({"clause": c.name, "lo": stl.fmt(lo), "hi": stl.fmt(hi),
                         "verdict": stl.verdict(lo, hi)})
        lo = min((_num(r["lo"]) for r in rows), default=float("inf"))
        hi = min((_num(r["hi"]) for r in rows), default=float("inf"))
        return {"t": round((trace.n - 1) * trace.dt, 2), "lo": stl.fmt(lo), "hi": stl.fmt(hi),
                "verdict": stl.verdict(lo, hi), "clauses": rows}


def _num(v) -> float:
    return float(v) if not isinstance(v, str) else (float("inf") if v == "inf" else float("-inf"))
