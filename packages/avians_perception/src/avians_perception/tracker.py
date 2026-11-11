"""Person tracker: from detections to a belief that always exists.

Each candidate is a track: a Kalman filter on ground position and velocity,
and a log-likelihood ratio (LLR) that the track is the person rather than
clutter (the track-existence test of multi-target tracking, Bar-Shalom; the
score-likelihood form of IPDA).

Motion: the velocity is an Ornstein-Uhlenbeck process (Singer model) with
standard deviation `sigma_v` and correlation time `tau`. A person mostly
sits still, but an injured hiker may crawl or walk, so between sightings the
position uncertainty grows honestly: about sigma_v * sqrt(2 tau t) for long
gaps (47 m after 5 minutes with the defaults), and collapses again at the
next sighting. `deliver` refuses a loose estimate, which is the point.

Two log-likelihood ratios per track, because two different questions:
    existence  is there a persistent object here, or clutter?
               hit:  += log( P_D / P_FA )      miss: += log( (1 - P_D) / (1 - P_FA) )
    identity   is that object the person, or a decoy (tent, boulder)?
               hit:  += log( p(score | person) / p(score | decoy or clutter) )
with P_D the camera's detection probability at that altitude and distance
(the same model the simulated detector draws from, calibrated to YOLO in the
Gazebo tier) and the score densities from the camera model. A decoy is
detected as persistently as a person, so only its scores give it away; that
is why the two are kept apart. conf = P(object) x P(person | object).
A track whose existence falls far below the prior is dropped.

Association: nearest track inside the Mahalanobis gate and a physical gate;
a low-score box never moves a credible track.

The belief the pilot and the spec see (est_east, est_north, est_sigma, conf,
found) is the found track while it lives, else the most credible track, else
the prior: the mean of the search regions weighted by the person prior, with
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


def _gauss(x: float, mu: float, sd: float) -> float:
    return math.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


@dataclass
class Track:
    id: int
    kf: KalmanFilter
    llr: float                  # existence: object vs clutter
    t: float
    idr: float = 0.0            # identity: person vs decoy, from scores (includes the prior)
    hits: int = 1
    truths: dict = field(default_factory=dict)     # truth label -> count (metrics only)

    @property
    def conf(self) -> float:
        return _sig(self.llr) * _sig(self.idr)

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
    def __init__(self, prior: tuple[float, float, float], camera=None, prior_conf: float = 0.2,
                 gate: float = 3.0, gate_m: float = 12.0, found_threshold: float = 0.8,
                 credible: float = 0.35, tau: float = 60.0, sigma_v: float = 0.15,
                 q_pos: float = 0.01, p_fa: float = 0.02, clutter_weight: float = 0.5):
        self.prior = prior                  # east, north, sigma
        self.cam = camera
        self.l0 = _logit(prior_conf)
        self.gate, self.gate_m = gate, gate_m
        self.found_threshold, self.credible = found_threshold, credible
        self.tau, self.sigma_v, self.q_pos = tau, sigma_v, q_pos
        self.p_fa, self.clutter_weight = p_fa, clutter_weight
        self.found_id: int | None = None
        self.tracks: list[Track] = []
        self.found_t: float | None = None
        self._next_id = 1
        self._t = 0.0

    @classmethod
    def from_world(cls, world, camera=None, weights: dict | None = None, **kw) -> "PersonTracker":
        weights = weights or world.person.region_weights or {r.id: 1.0 for r in world.search_regions()}
        tot = sum(weights.values())
        pts = [(world.regions[k].centroid, v / tot) for k, v in weights.items()]
        me = sum(c[0] * w for c, w in pts)
        mn = sum(c[1] * w for c, w in pts)
        var = sum(w * ((c[0] - me) ** 2 + (c[1] - mn) ** 2) for c, w in pts)
        return cls((me, mn, math.sqrt(var / 2) + 30.0), camera, **kw)

    # ------------------------------------------------------------ likelihoods
    def _p_detect(self, alt: float, dist: float) -> float:
        if self.cam is None:
            return 0.7
        return max(0.05, min(0.95, self.cam.p_detect(alt, dist)))

    def _score_llr(self, score: float) -> float:
        """log p(score | person) - log p(score | clutter)."""
        if self.cam is None:
            return _logit(score)
        mu_p, sd_p = self.cam.person_score
        mu_d, sd_d = self.cam.decoy_score
        lo, hi = self.cam.false_score
        p_person = _gauss(score, mu_p, sd_p)
        p_clutter = (self.clutter_weight * _gauss(score, mu_d, sd_d)
                     + (1 - self.clutter_weight) * (1.0 / (hi - lo) if lo <= score <= hi else 1e-3))
        return math.log(max(p_person, 1e-6) / max(p_clutter, 1e-6))

    def _hit_llr(self, p_d: float) -> float:
        return math.log(p_d / self.p_fa)

    def _miss_llr(self, p_d: float) -> float:
        return math.log((1 - p_d) / (1 - self.p_fa))

    # ------------------------------------------------------------ tracks
    def _new(self, d, p_d: float) -> Track:
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.x = np.array([[d.east], [d.north], [0.0], [0.0]])
        kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        kf.P = np.diag([d.sigma ** 2, d.sigma ** 2, self.sigma_v ** 2, self.sigma_v ** 2])
        kf.R = np.eye(2) * d.sigma ** 2
        tr = Track(self._next_id, kf, self._hit_llr(p_d), d.t, idr=self.l0 + self._score_llr(d.score))
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
            # position noise from the OU velocity over the step: Var(int v) = 2 s_v^2 tau (dt - k)
            qp = self.q_pos * dt + 2 * self.sigma_v ** 2 * self.tau * max(0.0, dt - k)
            tr.kf.Q = np.array([[qp, 0, 0, 0], [0, qp, 0, 0], [0, 0, qv, 0], [0, 0, 0, qv]], float)
            tr.kf.predict()
            tr.t = t

    def update(self, t: float, detections, seen: tuple[float, float, float] | None = None,
               alt: float | None = None) -> Belief:
        """One camera frame: its detections, the circle it saw (east, north,
        radius) for misses, and the camera altitude (for P_D)."""
        self._predict(t)
        used = set()
        alt = alt if alt is not None else (seen[2] / math.tan(self.cam.hfov / 2) if seen and self.cam else 30.0)
        for d in detections:
            best, best_m = None, self.gate ** 2
            z = np.array([[d.east], [d.north]])
            dist = math.dist((d.east, d.north), (seen[0], seen[1])) if seen else 0.0
            p_d = self._p_detect(alt, dist)
            for tr in self.tracks:
                if math.dist(tr.pos, (d.east, d.north)) > self.gate_m + 2 * tr.sigma:
                    continue
                if d.score < 0.5 and tr.conf >= self.credible:
                    continue          # a weak box never moves a credible track (false positives)
                S = tr.kf.H @ tr.kf.P @ tr.kf.H.T + np.eye(2) * d.sigma ** 2
                y = z - tr.kf.H @ tr.kf.x
                m = (y.T @ np.linalg.solve(S, y)).item()
                if m < best_m:
                    best, best_m = tr, m
            if best is None:
                tr = self._new(d, p_d)
                self.tracks.append(tr)
                used.add(tr.id)
                continue
            best.kf.R = np.eye(2) * d.sigma ** 2
            best.kf.update(z)
            best.llr = min(12.0, best.llr + self._hit_llr(p_d))
            best.idr = max(-12.0, min(12.0, best.idr + self._score_llr(d.score)))
            best.hits += 1
            best.truths[d.truth] = best.truths.get(d.truth, 0) + 1
            used.add(best.id)
        if seen is not None:
            se, sn, r = seen
            for tr in self.tracks:
                dist = math.dist(tr.pos, (se, sn))
                if tr.id not in used and dist + tr.sigma < 0.9 * r:     # surely in view, not seen
                    tr.llr += self._miss_llr(self._p_detect(alt, dist))
        self.tracks = [tr for tr in self.tracks if tr.llr > -4.0 or tr.id == self.found_id]
        self._t = t
        self._readopt()
        if self.found_id is None:
            tr = self.best()
            if tr is not None and tr.conf >= self.found_threshold:
                self.found_id, self.found_t = tr.id, t
        return self.belief()

    def _readopt(self) -> None:
        """The person found, then re-seen as a new track (their estimate had drifted out
        of the gate): a confident track near where the found track says they are takes
        over the found role and inherits its identity evidence."""
        if self.found_id is None:
            return
        ft = next((x for x in self.tracks if x.id == self.found_id), None)
        if ft is None:
            return
        for tr in self.tracks:
            if tr.id == ft.id or tr.conf < self.credible or tr.hits < 3:
                continue
            if math.dist(tr.pos, ft.pos) <= 3 * ft.sigma + 10.0 and tr.idr > -2.0:
                tr.idr = max(tr.idr, ft.idr)
                self.found_id = tr.id
                self.tracks = [x for x in self.tracks if x.id != ft.id]
                return

    def best(self) -> Track | None:
        return max(self.tracks, key=lambda tr: (tr.conf, tr.t), default=None)

    def belief(self) -> Belief:
        found = self.found_id is not None
        tr = self.best()
        if found:
            # the found track (its last estimate, with a growing sigma, when out of sight) unless
            # another track is itself confident enough to count as found: never a weak stray
            ft = next((x for x in self.tracks if x.id == self.found_id), None)
            if ft is not None and (tr is None or tr.conf < self.found_threshold or ft.conf >= tr.conf - 1e-9):
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
