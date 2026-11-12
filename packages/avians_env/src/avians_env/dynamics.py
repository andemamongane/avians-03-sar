"""Random zone-safety dynamics: storm cells and fires, drawn per seed.

    dynamics:
      storms: {rate_per_hour, radius: [a, b], speed: [a, b], lifetime: [a, b], caution_m}
      fires:  {p_ignite, t_ignite: [a, b], spread_delay: [a, b], p_spread}

Storms: a Poisson number of cells over the horizon. Each appears at a random
time and place (up to 100 m outside the area, so some drift in from the
edge), moves in a straight line roughly downwind at its speed, and lives for
its lifetime. While a cell overlaps a zone, the zone is CLOSED ("storm cell
overhead"); zones within caution_m of a cell are marked caution (shown to the
pilot, not enforced).

Fires: with probability p_ignite a fire starts in a random fire-prone zone at
a time in t_ignite; after spread_delay it spreads, with probability p_spread
each, to the neighbouring zones (zones sharing a boundary) that can burn
(fire-prone, or any non-protected zone if `spread_any: true`), and so on. A
burning zone stays closed to the end.

Protected zones (the base) and no-fly zones are never closed by dynamics.
The output is a SafetyEvent timeline (merged with the world file's own) plus
the storm tracks and fire log, which replays and the showcase draw.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from shapely.geometry import Point

from .world import SafetyEvent


@dataclass
class Storm:
    id: str
    t0: float
    t1: float
    x0: float
    y0: float
    vx: float
    vy: float
    radius: float

    def center(self, t: float) -> tuple[float, float]:
        dt = t - self.t0
        return (self.x0 + self.vx * dt, self.y0 + self.vy * dt)

    def active(self, t: float) -> bool:
        return self.t0 <= t <= self.t1

    def summary(self) -> dict:
        return {k: round(v, 2) if isinstance(v, float) else v for k, v in vars(self).items()}


def _u(rng, r):
    return float(rng.uniform(float(r[0]), float(r[1]))) if isinstance(r, (list, tuple)) else float(r)


def closable(world) -> list:
    return [r for r in world.regions.values() if r.kind != "no_fly" and not r.protected]


def draw(world, rng: np.random.Generator, horizon_s: float, wind=None):
    """(timeline events, storms, fires) for one episode."""
    dyn = world.dynamics or {}
    zones = closable(world)
    intervals: dict[str, list[tuple[float, float, str]]] = {z.id: [] for z in zones}
    storms: list[Storm] = []
    fires: list[dict] = []
    # ---- storms
    st = dyn.get("storms")
    if st:
        xs, ys = zip(*world.bounds)
        x0, x1, y0, y1 = min(xs) - 100, max(xs) + 100, min(ys) - 100, max(ys) + 100
        n = int(rng.poisson(float(st.get("rate_per_hour", 3)) * horizon_s / 3600.0))
        wdir = math.atan2(wind.north, wind.east) if wind is not None and (wind.east or wind.north) else None
        step = float(st.get("sample_s", 5.0))
        for k in range(n):
            t0 = float(rng.uniform(0, horizon_s * 0.85))
            life = _u(rng, st.get("lifetime", [300, 700]))
            sp = _u(rng, st.get("speed", [1, 3]))
            hd = (wdir if wdir is not None else rng.uniform(0, 2 * math.pi)) + rng.normal(0, 0.6)
            s = Storm(f"storm{k + 1}", t0, min(horizon_s, t0 + life), float(rng.uniform(x0, x1)),
                      float(rng.uniform(y0, y1)), sp * math.cos(hd), sp * math.sin(hd),
                      _u(rng, st.get("radius", [45, 100])))
            storms.append(s)
            # when does the cell overlap each zone (sampled)
            for z in zones:
                inside, start = False, None
                for t in np.arange(s.t0, s.t1 + step, step):
                    c = s.center(float(t))
                    hit = z.shape.distance(Point(c)) < s.radius
                    if hit and not inside:
                        inside, start = True, float(t)
                    elif not hit and inside:
                        inside = False
                        intervals[z.id].append((start, float(t), f"storm cell overhead ({s.id})"))
                if inside:
                    intervals[z.id].append((start, s.t1, f"storm cell overhead ({s.id})"))
        storms.sort(key=lambda s: s.t0)
    # ---- fires
    fi = dyn.get("fires")
    if fi:
        prone = [z for z in zones if z.fire_prone]
        if prone and rng.random() < float(fi.get("p_ignite", 0.5)):
            first = prone[int(rng.integers(len(prone)))]
            t = _u(rng, fi.get("t_ignite", [120, 900]))
            burning = {first.id: t}
            fires.append({"zone": first.id, "t": round(t, 1), "from": None})
            frontier = [first]
            while frontier:
                z = frontier.pop(0)
                for nb in zones:
                    if nb.id in burning or not nb.shape.touches(z.shape) and not nb.shape.intersects(z.shape):
                        continue
                    if not (nb.fire_prone or fi.get("spread_any")):
                        continue
                    if rng.random() < float(fi.get("p_spread", 0.5)):
                        tn = burning[z.id] + _u(rng, fi.get("spread_delay", [180, 420]))
                        if tn < horizon_s:
                            burning[nb.id] = tn
                            fires.append({"zone": nb.id, "t": round(tn, 1), "from": z.id})
                            frontier.append(nb)
            for zid, t in burning.items():
                intervals[zid].append((t, horizon_s + 1, "fire" if not any(
                    f["zone"] == zid and f["from"] for f in fires) else "fire spreading"))
    # ---- intervals -> transitions (union per zone)
    events: list[SafetyEvent] = []
    for zid, ivs in intervals.items():
        if not ivs:
            continue
        ivs.sort()
        merged = [list(ivs[0])]
        for a, b, why in ivs[1:]:
            if a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b, why])
        for a, b, why in merged:
            note = ("Fire crews closed the zone: the fire is spreading" if why.startswith("fire")
                    else f"Storm cell overhead: zone closed until it passes")
            events.append(SafetyEvent(round(a, 1), zid, False, note))
            if b <= horizon_s:
                events.append(SafetyEvent(round(b, 1), zid, True, "the storm has passed; zone reopened"))
    events.sort(key=lambda e: (e.t, e.zone))
    return events, storms, fires


def cautions(world, storms, t: float, caution_m: float) -> dict[str, str]:
    """Zones near an active storm cell at t (not necessarily closed): id -> why."""
    out = {}
    for s in storms:
        if not s.active(t):
            continue
        c = Point(s.center(t))
        for z in closable(world):
            d = z.shape.distance(c)
            if s.radius <= d < s.radius + caution_m:
                out.setdefault(z.id, f"storm cell {d - s.radius:.0f} m away, moving "
                                     f"{_heading(s.vx, s.vy)} at {math.hypot(s.vx, s.vy):.1f} m/s")
    return out


def _heading(vx: float, vy: float) -> str:
    names = ["east", "north-east", "north", "north-west", "west", "south-west", "south", "south-east"]
    return names[int(round(math.degrees(math.atan2(vy, vx)) / 45.0)) % 8]
