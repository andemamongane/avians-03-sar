"""Pre-check: before a skill starts, say plainly if it is a bad idea.

Not the enforcement layer (PX4 and the reflex layer are). It catches what the
pilot can fix by choosing differently, and says why in words a pilot can act
on. Hard problems (a target inside a no-fly zone) reject the call; soft ones
(not enough battery to get back) start it with a warning.
"""
from __future__ import annotations

import math

from avians_skills import SkillError


def battery_check(ctx, goal: tuple[float, float], payload: bool | None = None) -> str:
    """'' or a warning if, after reaching goal, the drone could not reach the
    nearest station with the reserve left."""
    st, lim, en = ctx.state, ctx.limits, ctx.energy
    if en is None:
        return ""
    payload = st.payload if payload is None else payload
    d1 = math.dist((st.east, st.north), goal)
    stations = list(ctx.world.stations.values())
    if not stations:
        return ""
    d2 = min(math.dist(goal, s.position) for s in stations)
    need = en.cost(d1, lim.cruise_speed, payload) + en.cost(d2, lim.cruise_speed, False) + 2.0
    left = st.battery - need
    if left < lim.reserve_pct:
        return (f"battery: getting there and on to the nearest station takes about {need:.0f} %, "
                f"leaving {left:.0f} % (reserve {lim.reserve_pct:g} %)")
    return ""


def goal_of(ctx, name: str, args: dict):
    w = ctx.world
    if name in ("goto", "land", "precision_land") and args.get("place"):
        return w.place(args["place"]) if args["place"] in (*w.stations, *w.depots, *w.regions, "launch") else None
    if name == "goto" and args.get("east") is not None and args.get("north") is not None:
        return (float(args["east"]), float(args["north"]))
    if name == "search_region" and args.get("region") in w.regions:
        return w.regions[args["region"]].centroid
    if name in ("recharge", "swap_battery") and args.get("station") in w.stations:
        return w.stations[args["station"]].position
    if name == "pick_up" and args.get("depot", "base") in w.depots:
        return w.depots[args.get("depot", "base")].position
    if name in ("deliver", "track_person", "inspect", "intercept") and ctx.belief is not None:
        b = ctx.belief()
        return (b.east, b.north)
    return None


def precheck(ctx, name: str, args: dict) -> list[str]:
    """Warnings (possibly empty). Raises SkillError for calls that must not start."""
    st, lim = ctx.state, ctx.limits
    alt = args.get("alt")
    if alt is not None and not lim.min_alt <= float(alt) <= lim.max_alt:
        raise SkillError(f"altitude {alt} m is outside the envelope {lim.min_alt:g}-{lim.max_alt:g} m")
    if args.get("speed") is not None and float(args["speed"]) > lim.max_speed:
        raise SkillError(f"speed {args['speed']} m/s is over the envelope's {lim.max_speed:g} m/s")
    if st.landed and name not in ("takeoff", "pick_up", "recharge", "swap_battery", "wait_until",
                                  "report", "request_help", "set_speed", "drop_marker",
                                  "capture_image", "escalate", "land", "precision_land", "rtl"):
        raise SkillError("on the ground: take off first (or use a skill that takes off itself)")
    goal = goal_of(ctx, name, args)
    warnings = []
    if goal is not None:
        why = ctx.planner.blocked_by(goal[0], goal[1], float(alt or lim.cruise_alt), ctx.unsafe_zones())
        if why and name not in ("search_region",):
            raise SkillError(f"{name} would end inside {why}")
        if name == "search_region":
            z = [u.id for u in ctx.unsafe_zones() if u.shape.intersects(ctx.world.regions[args["region"]].shape)]
            if z:
                warnings.append(f"{', '.join(z)} is closed over part of {args['region']}: that part is skipped")
        w = battery_check(ctx, goal, payload=st.payload or name == "pick_up")
        if w:
            warnings.append(w)
    return warnings
