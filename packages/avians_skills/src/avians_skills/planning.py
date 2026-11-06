"""Path planning around what the drone must not fly through.

Default: the visibility graph + A* (Lozano-Perez & Wesley 1979), the exact
shortest path among polygonal obstacles in the plane. The forbidden set is
re-derived at every plan from the world at that moment:

    no-fly regions                       always
    hazard zones while unsafe            from the safety timeline (time-varying)
    obstacles taller than the altitude   minus a vertical margin
    outside the operating area           the geofence

each grown by the safety margin. Altitude is handled by the caller (climb
first, then cross), which is how a multirotor flies a SAR mission anyway.

For 3D or kinodynamic planning (terrain, tight spaces) the OMPL adapter
(RRT*, `pip install ompl`, Linux) can replace this with the same interface;
see README.md.
"""
from __future__ import annotations

import math

import networkx as nx
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from .base import SkillError


class PathPlanner:
    def __init__(self, world, margin_m: float = 6.0, vertical_margin_m: float = 5.0,
                 hazard_margin_m: float = 3.0):
        self.world = world
        self.margin = margin_m
        self.hazard_margin = hazard_margin_m
        self.vmargin = vertical_margin_m
        self._cache: dict = {}

    # ------------------------------------------------------------ forbidden set
    def forbidden(self, alt: float, unsafe=()) -> list[tuple[str, Polygon]]:
        """(id, polygon grown by the margin) for everything blocking at `alt`."""
        out = [(r.id, r.shape.buffer(self.margin, quad_segs=2)) for r in self.world.no_fly()]
        out += [(z.id, z.shape.buffer(self.hazard_margin, quad_segs=2)) for z in unsafe]
        out += [(o.id, o.shape.buffer(self.margin, quad_segs=2)) for o in self.world.obstacles.values()
                if o.height + self.vmargin > alt]
        return out

    def blocked_by(self, e: float, n: float, alt: float, unsafe=()) -> str | None:
        """Id of what (e, n) is inside of (with the margin), or None."""
        pt = Point(e, n)
        for i, p in self.forbidden(alt, unsafe):
            if p.contains(pt):
                return i
        b = self.world.bounds_shape()
        if b is not None and not b.contains(pt):
            return "the operating area (geofence)"
        return None

    # ------------------------------------------------------------ planning
    def plan(self, start: tuple[float, float], goal: tuple[float, float], alt: float,
             unsafe=()) -> list[tuple[float, float]]:
        """Shortest polyline from start to goal avoiding the forbidden set.
        Raises SkillError when the goal is forbidden or unreachable. A start
        inside a forbidden polygon (e.g. a zone just closed around the drone)
        is allowed out by ignoring that polygon."""
        polys = self.forbidden(alt, unsafe)
        why = self.blocked_by(goal[0], goal[1], alt, unsafe)
        if why:
            raise SkillError(f"the goal ({goal[0]:.0f}, {goal[1]:.0f}) is inside {why} "
                             f"(margin {self.margin:g} m)")
        sp = Point(start)
        polys = [(i, p) for i, p in polys if not p.contains(sp)]
        if not polys:
            return [start, goal]
        union = unary_union([p for _, p in polys])
        bounds = self.world.bounds_shape()
        if self._clear(LineString([start, goal]), union, bounds):
            return [start, goal]
        key = (round(alt / 5), tuple(sorted(i for i, _ in polys)))
        if key not in self._cache:
            self._cache[key] = self._graph(polys, union, bounds)
        g = self._cache[key].copy()
        for name, p in (("start", start), ("goal", goal)):
            g.add_node(name, p=p)
            for v, d in list(g.nodes(data=True)):
                if v in ("start", "goal") and v == name:
                    continue
                q = d["p"]
                if self._clear(LineString([p, q]), union, bounds):
                    g.add_edge(name, v, w=math.dist(p, q))
        try:
            nodes = nx.astar_path(g, "start", "goal",
                                  heuristic=lambda a, b: math.dist(g.nodes[a]["p"], g.nodes[b]["p"]),
                                  weight="w")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            raise SkillError(f"no path from ({start[0]:.0f}, {start[1]:.0f}) to "
                             f"({goal[0]:.0f}, {goal[1]:.0f}) at {alt:.0f} m") from None
        return [g.nodes[v]["p"] for v in nodes]

    def _graph(self, polys, union, bounds) -> nx.Graph:
        g = nx.Graph()
        pts = []
        for _, p in polys:
            outer = p.buffer(1.0, quad_segs=2)               # vertices just outside the grown set
            for q in list(outer.exterior.coords)[:-1]:
                if not union.contains(Point(q)) and (bounds is None or bounds.contains(Point(q))):
                    pts.append(q)
        for k, p in enumerate(pts):
            g.add_node(k, p=p)
        if len(pts) > 1:
            pairs = [(a, b) for a in range(len(pts)) for b in range(a + 1, len(pts))]
            lines = shapely.linestrings([[pts[a], pts[b]] for a, b in pairs])
            ok = ~shapely.intersects(lines, union.buffer(-0.01))
            if bounds is not None:
                ok &= shapely.within(lines, bounds)
            for (a, b), good in zip(pairs, ok):
                if good:
                    g.add_edge(a, b, w=math.dist(pts[a], pts[b]))
        return g

    @staticmethod
    def _clear(line: LineString, union, bounds) -> bool:
        if line.intersects(union.buffer(-0.01)):
            return False
        return bounds is None or bounds.contains(line)

    @staticmethod
    def length(path) -> float:
        return sum(math.dist(a, b) for a, b in zip(path, path[1:]))
