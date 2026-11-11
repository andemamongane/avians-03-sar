"""Fit the simulated camera to what the real detector does.

Gazebo-tier episodes log, per camera frame, the drone's altitude, the true
distance of the person from the image centre on the ground, and whether YOLO
reported them (events of kind "detector_sample"). `fit_camera(rows)` fits the
simulated detector's P_D(alt, dist) to those rows by maximum likelihood over
(p_max, alt_scale, edge_loss), and the score distributions from the reported
scores, so the fast tier's detector and the tracker's likelihoods match the
real one.

    avians calibrate-detector data/episodes/sar_supply/haiku   # prints a CameraModel block

Rows: dicts with alt, dist, detected (0/1), score (when detected), truth
("person" | "decoy:<id>" | "false").
"""
from __future__ import annotations

import math

import numpy as np

from .detector import CameraModel


def _p(alt, dist, cam: CameraModel):
    return np.array([cam.p_detect(a, d) for a, d in zip(alt, dist)])


def fit_camera(rows: list[dict], base: CameraModel | None = None) -> CameraModel:
    base = base or CameraModel()
    person = [r for r in rows if r.get("truth", "person") == "person"]
    if not person:
        return base
    alt = np.array([float(r["alt"]) for r in person])
    dist = np.array([float(r["dist"]) for r in person])
    y = np.array([1.0 if r["detected"] else 0.0 for r in person])
    best, best_ll = base, -math.inf
    for p_max in np.linspace(0.5, 0.98, 13):
        for alt_scale in (15, 20, 30, 40, 60, 90):
            for edge_loss in (0.2, 0.35, 0.5, 0.65, 0.8):
                cam = CameraModel(hfov=base.hfov, rate_hz=base.rate_hz, p_max=p_max, alt_scale=alt_scale,
                                  edge_loss=edge_loss, min_alt=base.min_alt)
                p = np.clip(_p(alt, dist, cam), 1e-3, 1 - 1e-3)
                ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
                if ll > best_ll:
                    best, best_ll = cam, ll
    scores = [float(r["score"]) for r in person if r.get("detected") and r.get("score") is not None]
    if len(scores) >= 10:
        best.person_score = (float(np.mean(scores)), float(np.std(scores) + 1e-3))
    false = [float(r["score"]) for r in rows if r.get("truth") == "false" and r.get("score") is not None]
    if len(false) >= 10:
        best.false_score = (float(np.percentile(false, 5)), float(np.percentile(false, 95)))
    decoy = [float(r["score"]) for r in rows if str(r.get("truth", "")).startswith("decoy") and r.get("score") is not None]
    if len(decoy) >= 10:
        best.decoy_score = (float(np.mean(decoy)), float(np.std(decoy) + 1e-3))
    n_frames = {r.get("frame") for r in rows}
    if n_frames:
        best.false_rate = len([r for r in rows if r.get("truth") == "false"]) / max(1, len(n_frames))
    return best


def camera_yaml(cam: CameraModel) -> str:
    return ("camera:\n" + "".join(f"  {k}: {v!r}\n" for k, v in vars(cam).items()))
