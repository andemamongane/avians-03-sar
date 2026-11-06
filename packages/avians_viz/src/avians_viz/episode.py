"""Read an episode directory back (any tier)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Episode:
    dir: Path
    meta: dict
    summary: dict
    score: dict
    t: np.ndarray
    sig: dict            # name -> array, aligned with t
    events: list[dict]
    decisions: list[dict]   # t_obs, t_apply, tool, args, reason

    @classmethod
    def load(cls, d: str | Path) -> "Episode":
        d = Path(d)
        rows = [json.loads(l) for l in open(d / "signals.jsonl")]
        t = np.array([r["t"] for r in rows])
        names = set().union(*rows) - {"t"}
        sig = {k: np.array([r.get(k, np.nan) for r in rows], float) for k in names}
        events = [json.loads(l) for l in open(d / "events.jsonl")]
        decisions = []
        tj = d / "trajectory.json"
        if tj.exists():
            for step in json.loads(tj.read_text())["content"]:
                if step.get("class_") == "api_action":
                    dd = step["details"]
                    decisions.append({"t_obs": dd.get("t_obs", dd.get("t")), "t_apply": dd.get("t_apply"),
                                      "tool": step["function"], "args": step["kwargs"],
                                      "reason": step.get("description") or "", "by": dd.get("by")})
        load = lambda n: json.loads((d / n).read_text()) if (d / n).exists() else {}  # noqa: E731
        return cls(d, load("meta.json"), load("summary.json"), load("score.json"), t, sig, events, decisions)
