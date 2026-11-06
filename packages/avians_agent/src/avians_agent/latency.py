"""Simulated decision latency: the time between the observation a decision is
based on and the moment it takes effect. The simulation keeps running in
between (it is not lockstep): the drone keeps flying its current skill, and
events that happen meanwhile are shown at the next decision.

Each model's latency is lognormal (median, sigma) plus a floor, from
configs/latency.yaml; fit them to the wall-clock times logged per decision.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class LatencyModel:
    median_s: float = 3.0
    sigma: float = 0.5
    floor_s: float = 0.3

    def sample(self, rng: np.random.Generator) -> float:
        if self.median_s <= 0:
            return self.floor_s
        return self.floor_s + float(rng.lognormal(math.log(self.median_s), self.sigma))

    @classmethod
    def for_model(cls, cfg: dict, model: str) -> "LatencyModel":
        d = cfg.get("models", {}).get(model, cfg.get("default", {}))
        return cls(float(d.get("median_s", 3.0)), float(d.get("sigma", 0.5)), float(d.get("floor_s", 0.3)))


def fit(samples: list[float], floor_s: float = 0.0) -> LatencyModel:
    x = np.log(np.maximum(np.asarray(samples) - floor_s, 1e-3))
    return LatencyModel(float(math.exp(np.median(x))), float(x.std()), floor_s)
