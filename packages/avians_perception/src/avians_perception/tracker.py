"""Person tracker: from detections to a belief that always exists.

Each candidate is a track: a Kalman filter on its ground position and velocity
(filterpy), with the velocity mean-reverting to zero (an Ornstein-Uhlenbeck /
Singer model, tau ~10 s): a person mostly sits or walks slowly, so between
sightings the estimate stays put and its sigma grows like sqrt(t), not t plus a log-odds that it is the person. A detection updates
the nearest track inside the gate (Mahalanobis), or starts a new one. Its
score moves the log-odds by gain * logit(score), so repeated low-score
detections (decoys) lower it; a track the camera should have seen this frame
but did not loses miss_penalty. conf = sigmoid(log-odds).

The belief the pilot and the spec see (est_east, est_north, est_sigma, conf)
is the track that reached `found` (while it lives), else the most confident
credible track, else the prior: the
mean of the search regions weighted by where the person probably is, with
sigma their spread. `found` latches once conf reaches found_threshold.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from filterpy.kalman import KalmanFilter


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


@dataclass
class Track:
    id: int
    kf: KalmanFilter
    logodds: float
    t: float
    hits: int = 1
    truths: dict = field(default_factory=dict)     # truth label -> count (metrics only)

    @property
    def conf(self) -> float:
        return 1.0 / (1.0 + math.exp(-self.logodds))

    @property
    def pos(self) -> tuple[float, float]:
        return float(self.kf.x[0, 0]), float(self.kf.x[1, 0])

    @property
    def sigma(self) -> float:
        return float(math.sqrt(max(self.kf.P[0, 0], self.kf.P[1, 1])))


@dataclass
class Belief:
    east: float
    north: float
    sigma: float
    conf: float
    found: bool
    source: str             # prior | track <id>
    tracks: int
    ve: float = 0.0         # estimated velocity, m/s (a walking person)
    vn: float = 0.0

    def signals(self) -> dict[str, float]:
        return {"est_east": self.east, "est_north": self.north, "est_sigma": self.sigma,
                "conf": self.conf, "found": 1.0 if self.found else 0.0}


class PersonTracker:
    def __init__(self, prior: tuple[float, float, float], prior_conf: float = 0.2,
                 gain: float = 0.6, miss_penalty: float = 0.3, gate: float = 3.0,
                 found_threshold: float = 0.8, credible: float = 0.35, tau: float = 10.0,
                 sigma_v: float = 0.25, q_pos: float = 0.01, gate_m: float = 12.0):
        self.prior = prior                  # east, north, sigma
        self.l0 = _logit(prior_conf)
        self.gain, self.miss, self.gate = gain, miss_penalty, gate
        self.found_threshold, self.credible = found_threshold, credible
        self.tau, self.sigma_v, self.q_pos, self.gate_m = tau, sigma_v, q_pos, gate_m
        self.found_id: int | None = None
        self.tracks: list[Track] = []
        self.found_t: float | None = None
        self._next_id = 1
        self._t = 0.0

    @classmethod
    def from_world(cls, world, **kw) -> "PersonTracker":
        weights = world.person.region_weights or {r.id: 1.0 for r in world.search_regions()}
        tot = sum(weights.values())
        pts = [(world.regions[k].centroid, v / tot) for k, v in weights.items()]
        me = sum(c[0] * w for c, w in pts)
        mn = sum(c[1] * w for c, w in pts)
        var = sum(w * ((c[0] - me) ** 2 + (c[1] - mn) ** 2) for c, w in pts)
        return cls((me, mn, math.sqrt(var / 2) + 30.0), **kw)

    def _new(self, d) -> Track:
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.x = np.array([[d.east], [d.north], [0.0], [0.0]])
        kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        kf.P = np.diag([d.sigma ** 2, d.sigma ** 2, self.sigma_v ** 2, self.sigma_v ** 2])
        kf.R = np.eye(2) * d.sigma ** 2
        tr = Track(self._next_id, kf, self.l0 + self.gain * _logit(d.score), d.t)
        tr.truths[d.truth] = 1
        self._next_id += 1
        return tr

    def _predict(self, t: float) -> None:
        for tr in self.tracks:
            dt = t - tr.t
            if dt <= 0:
                continue
            a = np.exp(-dt / self.tau)
            k = self.tau * (1 - a)
            tr.kf.F = np.array([[1, 0, k, 0], [0, 1, 0, k], [0, 0, a, 0], [0, 0, 0, a]], float)
            qv = self.sigma_v ** 2 * (1 - a * a)
            tr.kf.Q = np.diag([self.q_pos * dt, self.q_pos * dt, qv, qv])
            tr.kf.predict()
            tr.t = t

    def update(self, t: float, detections, seen: tuple[float, float, float] | None = None) -> Belief:
        """One camera frame: its detections, and the circle it saw (east,
        north, radius) for misses."""
        self._predict(t)
        used = set()
        for d in detections:
            best, best_m = None, self.gate ** 2
            z = np.array([[d.east], [d.north]])
            for tr in self.tracks:
                if math.dist(tr.pos, (d.east, d.north)) > self.gate_m:   # physical gate too
                    continue
                if d.score < 0.5 and tr.conf >= self.credible:
                    continue          # a weak box never moves a credible track (false positives)
                S = tr.kf.H @ tr.kf.P @ tr.kf.H.T + np.eye(2) * d.sigma ** 2
                y = z - tr.kf.H @ tr.kf.x
                m = (y.T @ np.linalg.solve(S, y)).item()
                if m < best_m:
                    best, best_m = tr, m
            if best is None:
                tr = self._new(d)
                self.tracks.append(tr)
                used.add(tr.id)
                continue
            best.kf.R = np.eye(2) * d.sigma ** 2
            best.kf.update(z)
            best.logodds = min(8.0, best.logodds + self.gain * _logit(d.score))
            best.hits += 1
            best.truths[d.truth] = best.truths.get(d.truth, 0) + 1
            used.add(best.id)
        if seen is not None:
            se, sn, r = seen
            for tr in self.tracks:
                if tr.id not in used and math.dist(tr.pos, (se, sn)) < 0.8 * r:
                    tr.logodds -= self.miss
        self.tracks = [tr for tr in self.tracks
                       if tr.logodds > self.l0 - 3.0 or tr.id == self.found_id]
        self._t = t
        if self.found_id is None:
            tr = self.best()
            if tr is not None and tr.conf >= self.found_threshold:
                self.found_id, self.found_t = tr.id, t
        return self.belief()

    def best(self) -> Track | None:
        return max(self.tracks, key=lambda tr: (tr.logodds, tr.t), default=None)

    def belief(self) -> Belief:
        found = self.found_id is not None
        tr = self.best()
        if found:              # the found track unless another has more evidence (ties: found)
            ft = next((x for x in self.tracks if x.id == self.found_id), None)
            if ft is not None and (tr is None or ft.logodds >= tr.logodds - 1e-9):
                tr = ft
        if tr is not None and tr.conf < self.credible and not found:
            tr = None
        if tr is not None:
            e, n = tr.pos
            return Belief(e, n, tr.sigma, tr.conf, found, f"track {tr.id}", len(self.tracks),
                          float(tr.kf.x[2, 0]), float(tr.kf.x[3, 0]))
        e, n, s = self.prior
        best = self.best()
        return Belief(e, n, s, best.conf if best else 0.0, found, "prior", len(self.tracks))
