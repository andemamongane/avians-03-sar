"""Following a path: the carrot (pure pursuit) on a polyline, with a speed
profile that brakes for corners and for the end.

Each tick: project the drone onto the path, put the carrot `lookahead` metres
further along, and command the carrot with the path's tangent velocity. The
commanded speed is limited so that the drone can still slow to the corner speed
(v_c = v * max(0.25, cos(turn / 2))) and to zero at the end with the vehicle's
acceleration: v <= sqrt(v_c^2 + 2 a d). The vehicle's own controller (Ruckig in
the fast tier, PX4's in Gazebo) handles jerk and attitude.
"""
from __future__ import annotations

import math

import numpy as np


class PathFollower:
    def __init__(self, points, speed: float, accel: float, lookahead: float | None = None,
                 tolerance: float = 1.5):
        pts = [tuple(map(float, p)) for p in points]
        dedup = [pts[0]]
        for p in pts[1:]:
            if math.dist(p, dedup[-1]) > 0.05:
                dedup.append(p)
        self.pts = np.array(dedup if len(dedup) > 1 else dedup * 2)
        seg = np.diff(self.pts, axis=0)
        self.seglen = np.linalg.norm(seg, axis=1)
        self.cum = np.concatenate([[0.0], np.cumsum(self.seglen)])
        self.length = float(self.cum[-1])
        self.speed, self.accel, self.tol = speed, accel, tolerance
        self.lookahead = lookahead if lookahead is not None else max(4.0, 0.8 * speed)
        self.s = 0.0                          # progress along the path, m
        self.vc = np.full(len(self.pts), speed)
        for i in range(1, len(self.pts) - 1):
            a, b = seg[i - 1], seg[i]
            na, nb = np.linalg.norm(a[:2]), np.linalg.norm(b[:2])
            if na > 0 and nb > 0:
                turn = math.acos(max(-1.0, min(1.0, float(a[:2] @ b[:2]) / (na * nb))))
                self.vc[i] = speed * max(0.25, math.cos(turn / 2))
        self.vc[-1] = 0.0

    def point(self, s: float) -> np.ndarray:
        s = min(max(s, 0.0), self.length)
        i = min(int(np.searchsorted(self.cum, s, side="right")) - 1, len(self.seglen) - 1)
        f = 0.0 if self.seglen[i] == 0 else (s - self.cum[i]) / self.seglen[i]
        return self.pts[i] + f * (self.pts[i + 1] - self.pts[i])

    def tangent(self, s: float) -> np.ndarray:
        i = min(int(np.searchsorted(self.cum, min(s, self.length - 1e-6), side="right")) - 1,
                len(self.seglen) - 1)
        d = self.pts[i + 1] - self.pts[i]
        n = np.linalg.norm(d)
        return d / n if n > 0 else np.zeros(3)

    def _project(self, p: np.ndarray) -> float:
        best, best_s = math.inf, self.s
        # search forward from the current progress only (no jumping back on crossings)
        for i in range(len(self.seglen)):
            if self.cum[i + 1] < self.s - 1.0:
                continue
            a, b = self.pts[i], self.pts[i + 1]
            ab = b - a
            L2 = float(ab @ ab)
            f = 0.0 if L2 == 0 else max(0.0, min(1.0, float((p - a) @ ab) / L2))
            q = a + f * ab
            d = float(np.linalg.norm(p - q))
            if d < best - 1e-9:
                best, best_s = d, self.cum[i] + f * self.seglen[i]
            if self.cum[i] > self.s + 3 * self.lookahead:
                break
        return max(self.s, best_s)

    def step(self, pos) -> tuple[tuple, tuple, float]:
        """(carrot position, feed-forward velocity, commanded speed)."""
        p = np.asarray(pos, float)
        self.s = self._project(p)
        s_c = min(self.length, self.s + self.lookahead)
        v = self.speed
        for i in range(len(self.pts)):
            d = self.cum[i] - self.s
            if d >= 0:
                v = min(v, math.sqrt(self.vc[i] ** 2 + 2 * self.accel * 0.8 * d))
        carrot = self.point(s_c)
        vel = self.tangent(s_c) * v if s_c < self.length else np.zeros(3)
        return tuple(carrot), tuple(vel), max(v, 0.5)

    def remaining(self) -> float:
        return self.length - self.s

    def done(self, pos, speed: float) -> bool:
        return (math.dist(tuple(pos), tuple(self.pts[-1])) < self.tol and speed < 0.8)

    def fraction(self) -> float:
        return 1.0 if self.length == 0 else self.s / self.length
