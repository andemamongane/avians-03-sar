"""Simulated person detector for the fast tier: what a downward camera with a
YOLO-class detector would report, without rendering anything.

Per camera frame (rate_hz), each candidate inside the camera's ground
footprint is detected with a probability that falls with altitude and with
distance from the image centre. Candidates are the person, the world's decoys
(detected less often and with lower scores) and random false positives. A
detection is a ground position with noise growing with altitude, plus a score.

In the Gazebo tier the same Detection comes from AAS's YOLO through
projection.ground_point(), so the tracker downstream does not know which tier
it is in. The numbers below are placeholders to be fitted to YOLO's behaviour
on Gazebo renders (see README.md, "Calibration").
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    t: float
    east: float
    north: float
    score: float            # 0..1, the detector's confidence for this box
    sigma: float            # m, 1-sigma position error
    truth: str = ""         # person | decoy:<id> | false (for logging and metrics only)


@dataclass
class CameraModel:
    hfov: float = 1.2               # rad, nadir camera (AAS's x500 camera is ~1.2 rad)
    rate_hz: float = 4.0
    p_max: float = 0.9              # detection probability at low altitude, image centre
    alt_scale: float = 40.0         # m: p falls by e above 20 m every alt_scale m
    edge_loss: float = 0.5          # p at the footprint edge is (1 - edge_loss) of the centre
    sigma0: float = 0.5             # m position noise at 0 m...
    sigma_per_m: float = 0.03       # ...plus this per metre of altitude
    false_rate: float = 0.02        # false positives per frame
    person_score: tuple = (0.8, 0.08)
    decoy_score: tuple = (0.45, 0.1)
    false_score: tuple = (0.3, 0.5)  # uniform
    min_alt: float = 2.0

    def footprint(self, alt: float) -> float:
        """Radius of the ground circle the camera sees, m."""
        return max(0.0, alt) * math.tan(self.hfov / 2)

    def p_detect(self, alt: float, d: float) -> float:
        r = self.footprint(alt)
        if alt < self.min_alt or d > r:
            return 0.0
        p = self.p_max * math.exp(-max(0.0, alt - 20.0) / self.alt_scale)
        return p * (1.0 - self.edge_loss * (d / r) ** 2)


def occluded(camera, target, obstacles) -> bool:
    """Is the ground point `target` hidden from `camera` by any obstacle prism?"""
    from shapely.geometry import LineString, Point
    e0, n0, alt = camera
    seg = LineString([(e0, n0), (target[0], target[1])])
    pt = Point(target[0], target[1])
    for o in obstacles:
        if o.shape.contains(pt):
            return True                      # under the canopy / inside the structure
        if not seg.intersects(o.shape):
            continue
        # altitude of the sight line where it crosses the obstacle: below its top = blocked
        cross = seg.intersection(o.shape)
        pts = list(cross.coords) if cross.geom_type == "LineString" else \
            [c for g in getattr(cross, "geoms", []) for c in g.coords]
        total = seg.length
        for (x, y) in pts:
            frac = math.dist((e0, n0), (x, y)) / total if total > 0 else 1.0
            if alt * (1 - frac) < o.height:
                return True
    return False


class SimulatedDetector:
    def __init__(self, camera: CameraModel | None = None, seed: int = 0):
        self.cam = camera or CameraModel()
        self.rng = np.random.default_rng(seed)

    def frame(self, t: float, drone: tuple[float, float, float], person: tuple[float, float],
              decoys=(), obstacles=()) -> list[Detection]:
        """One frame from `drone` (east, north, alt). `obstacles` (avians_env
        Obstacle) occlude: a target is hidden when the line of sight from the
        camera passes through one, or the target stands inside its footprint
        (under a canopy)."""
        e0, n0, alt = drone
        cam, rng, out = self.cam, self.rng, []
        sigma = cam.sigma0 + cam.sigma_per_m * max(0.0, alt)
        cands = [("person", person, 1.0, cam.person_score)]
        cands += [(f"decoy:{d.id}", d.position, d.strength, cam.decoy_score) for d in decoys]
        for truth, (pe, pn), strength, (mu, sd) in cands:
            if obstacles and occluded((e0, n0, alt), (pe, pn), obstacles):
                continue
            p = cam.p_detect(alt, math.hypot(pe - e0, pn - n0)) * strength
            if rng.random() < p:
                out.append(Detection(t, pe + rng.normal(0, sigma), pn + rng.normal(0, sigma),
                                     float(np.clip(rng.normal(mu, sd), 0.01, 0.99)), sigma, truth))
        r = cam.footprint(alt)
        if alt >= cam.min_alt and rng.random() < cam.false_rate:
            a, rr = rng.uniform(0, 2 * math.pi), r * math.sqrt(rng.random())
            out.append(Detection(t, e0 + rr * math.cos(a), n0 + rr * math.sin(a),
                                 float(rng.uniform(*cam.false_score)), sigma, "false"))
        return out
