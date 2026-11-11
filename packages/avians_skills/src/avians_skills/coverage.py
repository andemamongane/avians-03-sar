"""Coverage: search patterns, and a record of what the camera has seen.

Patterns (IAMSAR Manual Vol. II search patterns, as used by SAR aircrews):

    lawnmower         boustrophedon "parallel track" over a polygon: lanes
                      parallel to the long side of its minimum rotated
                      rectangle (fewest turns, Huang 2001), spaced by the
                      camera footprint x (1 - overlap), clipped to the polygon
                      minus what is forbidden, started from the corner nearest
                      the drone.
    expanding_square  from a datum outward, legs s, s, 2s, 2s, 3s, ...: the
                      pattern for a probable position known to within a few
                      hundred metres.
    sector            three passes through the datum at 120 degrees: for a
                      precise datum (a last sighting).

Fields2Cover (Mier et al. 2023) can replace lawnmower() for headland-aware
coverage when its Python bindings are available (Linux); same output.

CoverageMap keeps, per search region, a 5 m grid of cells the camera footprint
has passed over at a useful altitude: "r3: 64 % covered, last seen 210 s ago".
"""
from __future__ import annotations

import math

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import LineString, Polygon


def _lanes(poly: Polygon, spacing: float, rotate: float = 0.0):
    rect = poly.minimum_rotated_rectangle
    xs, ys = rect.exterior.coords.xy
    edges = [((xs[i], ys[i]), (xs[i + 1], ys[i + 1])) for i in range(4)]
    (a, b) = max(edges, key=lambda e: math.dist(*e))
    angle = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) + rotate
    c = poly.centroid
    rot = affinity.rotate(poly, -angle, origin=c)
    x0, y0, x1, y1 = rot.bounds
    # poly is already shrunk by spacing/2: lanes on both of its edges and
    # evenly between, no further apart than `spacing`
    h = y1 - y0
    n = 1 if h < 1e-6 else math.ceil(h / spacing - 1e-9) + 1
    lanes = []
    for k in range(n):
        y = (y0 + y1) / 2 if n == 1 else y0 + 1e-6 + k * (h - 2e-6) / (n - 1)
        seg = LineString([(x0 - 1, y), (x1 + 1, y)]).intersection(rot)
        parts = [seg] if isinstance(seg, LineString) else list(getattr(seg, "geoms", []))
        parts = [p for p in parts if isinstance(p, LineString) and p.length > 1.0]
        for p in sorted(parts, key=lambda p: p.coords[0][0]):
            back = affinity.rotate(p, angle, origin=c)
            lanes.append(list(back.coords))
    return lanes


def lawnmower(poly: Polygon, spacing: float, start: tuple[float, float],
              forbidden: list[Polygon] = (), rotate: float = 0.0) -> list[tuple[float, float]]:
    """Waypoints of a boustrophedon over poly (minus forbidden), nearest corner first."""
    area = poly.buffer(-spacing / 2) if poly.buffer(-spacing / 2).area > 0 else poly
    for f in forbidden:
        area = area.difference(f)
    if area.is_empty:
        return []
    polys = [area] if isinstance(area, Polygon) else [g for g in area.geoms if isinstance(g, Polygon)]
    best = None
    for p in polys:
        lanes = _lanes(p, spacing, rotate)
        if not lanes:
            continue
        for rev_order in (False, True):
            for flip_first in (False, True):
                order = lanes[::-1] if rev_order else lanes
                pts = []
                for k, lane in enumerate(order):
                    lane = lane[::-1] if (k % 2 == 1) ^ flip_first else lane
                    pts += lane
                d = math.dist(start, pts[0])
                if best is None or d < best[0]:
                    best = (d, pts, p)
        start = best[1][-1] if best else start
    return best[1] if best else []


def expanding_square(datum: tuple[float, float], spacing: float, radius: float,
                     heading: float = 0.0) -> list[tuple[float, float]]:
    pts, (e, n) = [tuple(datum)], datum
    leg, k, h = spacing, 0, heading
    while leg <= 2 * radius:
        e, n = e + leg * math.cos(h), n + leg * math.sin(h)
        pts.append((e, n))
        h += math.pi / 2
        k += 1
        if k % 2 == 0:
            leg += spacing
    return pts


def sector(datum: tuple[float, float], radius: float, heading: float = 0.0):
    pts = [tuple(datum)]
    for k in range(3):
        h = heading + k * 2 * math.pi / 3
        out = (datum[0] + radius * math.cos(h), datum[1] + radius * math.sin(h))
        side = (datum[0] + radius * math.cos(h + math.pi / 3), datum[1] + radius * math.sin(h + math.pi / 3))
        pts += [out, side, tuple(datum)]
    return pts


class CoverageMap:
    """Which parts of each search region the camera has seen."""

    def __init__(self, regions, cell_m: float = 5.0, min_alt: float = 5.0, max_alt: float = 50.0):
        self.cell, self.min_alt, self.max_alt = cell_m, min_alt, max_alt
        self.grids = {}
        for r in regions:
            x0, y0, x1, y1 = r.shape.bounds
            xs = np.arange(x0 + cell_m / 2, x1, cell_m)
            ys = np.arange(y0 + cell_m / 2, y1, cell_m)
            X, Y = np.meshgrid(xs, ys)
            inside = shapely.contains_xy(r.shape, X, Y)
            self.grids[r.id] = {"X": X[inside], "Y": Y[inside],
                                "seen": np.full(int(inside.sum()), -1.0)}

    def mark(self, t: float, e: float, n: float, alt: float, radius: float) -> None:
        if not (self.min_alt <= alt <= self.max_alt) or radius <= 0:
            return
        for g in self.grids.values():
            m = (g["X"] - e) ** 2 + (g["Y"] - n) ** 2 <= radius ** 2
            g["seen"][m] = t

    def fraction(self, rid: str, since: float | None = None) -> float:
        s = self.grids[rid]["seen"]
        if len(s) == 0:
            return 1.0
        ok = s >= (since if since is not None else 0.0)
        return float(ok.mean())

    def last_seen(self, rid: str) -> float | None:
        s = self.grids[rid]["seen"]
        return float(s.max()) if len(s) and s.max() >= 0 else None

    def summary(self, t: float) -> dict[str, dict]:
        out = {}
        for rid in self.grids:
            last = self.last_seen(rid)
            out[rid] = {"covered": round(self.fraction(rid), 3),
                        "last_seen_s_ago": None if last is None else round(t - last, 1)}
        return out
