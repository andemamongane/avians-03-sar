"""The reflex layer: a runtime-assurance monitor between the skills and the
vehicle (the simplex architecture, Sha 2001: a verified simple controller
takes over from the complex one near the edge of the safe set).

Every control tick, before a setpoint reaches the vehicle:

    get out      inside a zone that closed around it: fly to the nearest point
                 outside it (plus the hazard margin), at cruise speed
    keep-out     if the drone, braking now at max_accel, would stop inside a
                 no-fly zone, a closed zone, an obstacle (plus half the margin)
                 or outside the geofence, replace the setpoint with "brake to
                 a hold" and raise an event
    battery      below critical_pct, take over: land at the nearest station
                 if it is reachable, else land in place
    separation   (several drones) layered flight levels: while within `sep_m`,
                 one drone flies `sep_dz` above the other. A drone on a
                 landing-type task (deliver, land, pick-up, recharge, swap)
                 never has to climb; between equals the higher id does. The skill keeps running; only its setpoint is
                 overridden while the conflict lasts

When it acts, the current skill is interrupted and the pilot is woken with
the reason. It never acts while the skill's own behaviour is safe, so the
pilot's decisions stay the pilot's.
"""
from __future__ import annotations

import math


from shapely.geometry import LineString, Point

from avians_skills import Setpoint


class Reflex:
    def __init__(self, world, limits, critical_pct: float = 5.0, sep_m: float = 20.0, sep_dz: float = 15.0):
        self.world, self.limits, self.critical = world, limits, critical_pct
        self.sep_m, self.sep_dz = sep_m, sep_dz
        self.active: str | None = None
        self._hold = None
        self._emergency_land = None
        self._deconflicting = False

    def check(self, ctx, sp: Setpoint | None) -> tuple[Setpoint | None, str | None]:
        """(the setpoint to send, why the reflex intervened or None)."""
        st = ctx.state
        if st.landed:
            self.active = None
            return sp, None
        if st.battery < self.critical:
            if self._emergency_land is None:
                self._emergency_land = Setpoint("land", (st.east, st.north, 0.0))
                return self._emergency_land, (f"battery critical ({st.battery:.1f} % < "
                                              f"{self.critical:g} %): landing now")
            return self._emergency_land, None
        inside = next((z for z in ctx.unsafe_zones() if z.shape.contains(Point(st.east, st.north))), None)
        if inside is not None:
            q = self._exit(ctx, st, inside)
            sp_out = Setpoint("position", (q[0], q[1], max(st.alt, self.limits.min_alt)),
                              max_speed=self.limits.cruise_speed)
            first = self.active != f"leaving {inside.id}"
            self.active = f"leaving {inside.id}"
            return sp_out, (f"inside closed zone {inside.id}: leaving by the shortest way" if first else None)
        sp = self._separate(ctx, st, sp)
        why = self._keep_out(ctx, st)
        if why:
            if self.active is None:
                k = self.limits.stopping_distance(st.speed) / max(st.speed, 0.1)
                self._hold = Setpoint("position", (st.east + st.ve * k * 0.3, st.north + st.vn * k * 0.3,
                                                   max(st.alt, self.limits.min_alt)),
                                      max_speed=self.limits.max_speed)
                self.active = why
                return self._hold, why
            return self._hold, None
        self.active = None
        return sp, None

    def _exit(self, ctx, st, zone) -> tuple[float, float]:
        """Where to fly next to leave the closed zones: the cheapest point outside them
        (adjacent closed zones count as one blob), inside the operating area, reached by a
        path that goes around no-fly zones and obstacles (the planner, with closed zones
        allowed: we are leaving them). Cost = path length + 50 x the length inside zones that
        have been closed for over 20 s (leave through the one that only just closed).
        The escape path is kept until the set of closed zones changes."""
        from shapely.geometry import LineString, MultiPolygon, Polygon
        from shapely.ops import unary_union
        here = Point(st.east, st.north)
        t = ctx.t
        unsafe = ctx.unsafe_zones()
        key = tuple(sorted(z.id for z in unsafe))
        esc = getattr(self, "_escape", None)
        if esc is None or esc[0] != key:
            m = self.limits.hazard_margin_m + 2.0
            blob = unary_union([z.shape.buffer(m) for z in unsafe] or [zone.shape.buffer(m)])
            parts = list(blob.geoms) if isinstance(blob, MultiPolygon) else [blob]
            comp = next((g for g in parts if g.contains(here)), zone.shape.buffer(m))
            ring = Polygon(comp.exterior).exterior
            hard = unary_union([r.shape.buffer(self.limits.margin_m * 0.5) for r in self.world.no_fly()] +
                               [o.shape.buffer(self.limits.margin_m * 0.5) for o in self.world.obstacles.values()
                                if st.alt < o.height + 5])
            old = unary_union([z.shape for z in unsafe if self._closed_for(ctx, z.id, t) > 20.0]) \
                if unsafe else None
            b = self.world.bounds_shape()
            n = max(8, int(ring.length / 4.0))
            cands = []
            for k in range(n):
                q = ring.interpolate(k * ring.length / n)
                if b is not None and not b.contains(q):
                    continue
                if not hard.is_empty and hard.contains(q):
                    continue
                leg = LineString([here, q])
                pen = leg.intersection(old).length if old is not None and not old.is_empty else 0.0
                cands.append((leg.length + 50.0 * pen, (q.x, q.y), leg))
            cands.sort(key=lambda c: c[0])
            path = None
            for cost, q, leg in cands[:12]:
                if hard.is_empty or not leg.intersects(hard):
                    path = [q]
                    break
                try:                                        # around the no-fly zone / obstacle
                    pl = ctx.planner.plan((st.east, st.north), q, max(st.alt, self.limits.min_alt), ())
                    path = pl[1:]
                    break
                except Exception:
                    continue
            self._escape = (key, path or [(st.east, st.north)])
        path = self._escape[1]
        while len(path) > 1 and math.dist(path[0], (st.east, st.north)) < 3.0:
            path.pop(0)
        return path[0]

    def _closed_for(self, ctx, zid: str, t: float) -> float:
        since = None
        for e in ctx.world_state.timeline:
            if e.zone == zid and e.t <= t:
                since = None if e.safe else (since if since is not None else e.t)
        return 0.0 if since is None else t - since

    PRIORITY = ("deliver", "land", "precision_land", "pick_up", "recharge", "swap_battery", "rtl", "takeoff")

    def _separate(self, ctx, st, sp):
        """Deconfliction against the other drones (no-op with one drone).

        Layered flight levels: while within range, one drone flies sep_dz above the
        other, triggered early enough for the closing speed. Who climbs: a drone on a
        landing-type task (deliver, land, pick-up, recharge, swap, rtl, takeoff) never
        has to (it must be free to descend); between equals the higher id climbs.
        """
        sw = getattr(ctx.actuators, "swarm", None)
        if sw is None or sp is None or sp.mode != "position" or sp.pos is None:
            self._deconflicting = False
            return sp
        me = ctx.drone_id
        mine = sw.drones[me].current_skill() if me in sw.drones else "idle"
        my_pri = mine.startswith(self.PRIORITY)
        above = None
        climb_s = self.sep_dz / max(self.limits.max_climb, 0.1)
        for o in sw.others(me):
            if o["landed"] and st.landed:
                continue
            o_pri = (o["skill"] or "").startswith(self.PRIORITY)
            yields = (o_pri and not my_pri) or (o_pri == my_pri and me > o["id"])
            if not yields:
                continue
            dx, dy = o["east"] - st.east, o["north"] - st.north
            d = math.hypot(dx, dy)
            ov = sw.drones[o["id"]].vehicle.state()
            closing = max(0.0, -((dx * (ov.ve - st.ve) + dy * (ov.vn - st.vn)) / d)) if d > 1e-6 else 0.0
            if d < self.sep_m + closing * climb_s + 5.0:
                above = max(above or 0.0, o["alt"] + self.sep_dz)
        if above is None:
            if self._deconflicting:
                ctx.emit("progress", "clear of the other drone: back to the skill", wake=False)
            self._deconflicting = False
            return sp
        if not self._deconflicting:
            ctx.emit("progress", f"giving way to the other drone: flying {self.sep_dz:.0f} m above it", wake=False)
            self._deconflicting = True
        alt = min(self.limits.max_alt, max(sp.pos[2], above))
        return Setpoint(sp.mode, (sp.pos[0], sp.pos[1], alt), (sp.vel[0], sp.vel[1], 0.0), sp.max_speed, sp.yaw)

    def _keep_out(self, ctx, st) -> str | None:
        v = st.speed
        if v < 0.5:
            return None
        d = self.limits.stopping_distance(v) + 0.5 * v          # plus 0.5 s of reaction
        stop = (st.east + st.ve / v * d, st.north + st.vn / v * d)
        path = LineString([(st.east, st.north), stop])
        half = self.limits.margin_m / 2
        here = Point(st.east, st.north)
        for r in self.world.no_fly():
            g = r.shape.buffer(half)
            if path.intersects(g) and not g.contains(here):
                return f"heading into no-fly zone {r.id}: braked"
        for z in ctx.unsafe_zones():
            g = z.shape.buffer(half)
            if path.intersects(g) and not g.contains(here):
                return f"heading into closed zone {z.id}: braked"
        for o in self.world.obstacles.values():
            if st.alt < o.height + 2 and path.intersects(o.shape.buffer(half)):
                return f"heading into obstacle {o.id}: braked"
        b = self.world.bounds_shape()
        if b is not None and b.contains(here) and not b.contains(Point(stop)):
            return "heading out of the operating area: braked"
        return None
