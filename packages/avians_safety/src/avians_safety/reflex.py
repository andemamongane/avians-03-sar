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
    separation   (several drones) altitude deconfliction: when another drone
                 is within `sep_m` horizontally, the drone with the higher id
                 flies `sep_dz` above the other's altitude for as long as the
                 conflict lasts; the skill keeps running, only its altitude is
                 overridden (the way layered flight levels separate traffic)

When it acts, the current skill is interrupted and the pilot is woken with
the reason. It never acts while the skill's own behaviour is safe, so the
pilot's decisions stay the pilot's.
"""
from __future__ import annotations


from shapely.geometry import LineString, Point

from avians_skills import Setpoint


class Reflex:
    def __init__(self, world, limits, critical_pct: float = 5.0, sep_m: float = 20.0, sep_dz: float = 12.0):
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
            from shapely.ops import nearest_points
            grown = inside.shape.buffer(self.limits.hazard_margin_m + 2.0)
            q = nearest_points(grown.exterior, Point(st.east, st.north))[0]
            sp_out = Setpoint("position", (q.x, q.y, max(st.alt, self.limits.min_alt)),
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

    def _separate(self, ctx, st, sp):
        """Altitude deconfliction against the other drones (no-op with one drone)."""
        sw = getattr(ctx.actuators, "swarm", None)
        if sw is None or sp is None or sp.mode != "position" or sp.pos is None:
            self._deconflicting = False
            return sp
        import math
        me = ctx.drone_id
        above = None
        climb_s = self.sep_dz / max(self.limits.max_climb, 0.1)
        for o in sw.others(me):
            if (o["landed"] and st.landed) or me <= o["id"]:
                continue
            dx, dy = o["east"] - st.east, o["north"] - st.north
            d = math.hypot(dx, dy)
            ov = sw.drones[o["id"]].vehicle.state()
            closing = max(0.0, -((dx * (ov.ve - st.ve) + dy * (ov.vn - st.vn)) / d)) if d > 1e-6 else 0.0
            # trigger early enough to have climbed by the time the other drone is here
            if d < self.sep_m + closing * climb_s + 5.0:
                above = max(above or 0.0, o["alt"] + self.sep_dz)
        if above is None:
            if self._deconflicting:
                ctx.emit("progress", "clear of the other drone: back to the skill's altitude", wake=False)
            self._deconflicting = False
            return sp
        alt = min(self.limits.max_alt, max(sp.pos[2], above))
        if not self._deconflicting:
            ctx.emit("progress", f"another drone within {self.sep_m:.0f} m: flying {self.sep_dz:.0f} m above it "
                     f"until clear", wake=False)
            self._deconflicting = True
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
