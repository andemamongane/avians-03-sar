"""Online monitoring during a flight: ROSI verdicts per clause, early stop.

The rollout loop calls add(t, signals) every simulation step and maybe_eval(t)
after it. Evaluations run every `every_s` of simulated time; once a clause is
decided (satisfied or violated) it is not evaluated again, which keeps the
cost down as the trace grows. With policy "violated" the episode ends as soon
as any clause can no longer be satisfied: the rest of the rollout would only
cost compute (and API calls) without changing the score's sign.
"""
from __future__ import annotations

import numpy as np

from .spec import MissionSpec
from .stl import Trace


class OnlineMonitor:
    def __init__(self, spec: MissionSpec, max_speed: float, every_s: float = 5.0,
                 policy: str = "off"):
        if policy not in ("off", "violated"):
            raise ValueError("policy must be off or violated")
        self.spec, self.max_speed, self.every_s, self.policy = spec, max_speed, every_s, policy
        self.cols: dict[str, list[float]] = {k: [] for k in spec.registry.names()}
        self.n = 0
        self.decided: dict[str, dict] = {}
        self.history: list[dict] = []
        self._next = 0.0

    def add(self, t: float, signals: dict) -> None:
        """Record the signals at every grid time t_k = k * dt up to t (zero-order hold)."""
        dt = self.spec.dt
        while self.n * dt <= t + 1e-9:
            for k, col in self.cols.items():
                col.append(float(signals.get(k, self.spec.registry[k].default)))
            self.n += 1

    def trace(self) -> Trace:
        return Trace(self.spec.dt, {k: np.asarray(v) for k, v in self.cols.items()})

    def maybe_eval(self, t: float) -> dict | None:
        if t + 1e-9 < self._next or self.n == 0:
            return None
        self._next = t + self.every_s
        open_ = [c.name for c in self.spec.clauses if c.name not in self.decided]
        rep = self.spec.score_online(self.trace(), self.max_speed, only=open_) if open_ else \
            {"t": round(t, 2), "clauses": []}
        for r in rep["clauses"]:
            if r["verdict"] != "undecided":
                self.decided[r["clause"]] = {**r, "t": rep["t"]}
        rep["decided"] = dict(self.decided)
        self.history.append(rep)
        return rep

    def violated(self) -> list[str]:
        return [k for k, v in self.decided.items() if v["verdict"] == "violated"]

    def should_stop(self) -> bool:
        return self.policy == "violated" and bool(self.violated())

    def status_lines(self) -> list[str]:
        """For the pilot's prompt: what is already decided."""
        out = []
        for c in self.spec.clauses:
            d = self.decided.get(c.name)
            if d:
                out.append(f"- {c.name}: {d['verdict'].upper()} (decided at t={d['t']:.0f} s)")
        return out
