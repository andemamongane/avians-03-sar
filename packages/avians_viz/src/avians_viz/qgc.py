"""QGroundControl: the world as KML, the flight as a telemetry log to replay.

    world_kml(world)          regions as polygons: QGC Plan view -> import KML
    flight_kml(ep, world)     the flight path as a KML line (Google Earth, QGC overlay)
    tlog(ep, world, path)     a MAVLink telemetry log (.tlog) of the flight:
                              QGC -> Analyze Tools -> Log Replay plays it back
                              like a live flight (map, attitude, battery)

Gazebo-tier flights also have PX4's own .ulg (px4.ulg in the episode), which
QGC's log download and PX4 Flight Review read directly.
"""
from __future__ import annotations

import math
import time
from xml.sax.saxutils import escape

from .episode import Episode

EARTH_R = 6378137.0
KML_RGB = {"search": "3a9d2e", "no_fly": "3b3bd1", "hazard": "1c8af0", "landing": "666666", "other": "818a8a"}


def to_latlon(e: float, n: float, origin) -> tuple[float, float]:
    lat0, lon0 = origin[0], origin[1]
    return (lat0 + math.degrees(n / EARTH_R),
            lon0 + math.degrees(e / (EARTH_R * math.cos(math.radians(lat0)))))


def world_kml(world) -> str:
    marks = []
    for r in world.regions.values():
        ring = r.polygon + r.polygon[:1]
        coords = " ".join(f"{lon:.8f},{lat:.8f},0" for lat, lon in (to_latlon(e, n, world.origin) for e, n in ring))
        c = KML_RGB.get(r.kind, KML_RGB["other"])
        marks.append(f"<Placemark><name>{escape(r.id)}</name><description>{escape(r.kind + ': ' + r.description)}"
                     f"</description><Style><LineStyle><color>ff{c}</color><width>2</width></LineStyle>"
                     f"<PolyStyle><color>55{c}</color></PolyStyle></Style><Polygon><outerBoundaryIs><LinearRing>"
                     f"<coordinates>{coords}</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>")
    return ("<?xml version='1.0' encoding='UTF-8'?><kml xmlns='http://www.opengis.net/kml/2.2'><Document>"
            f"<name>{escape(world.name)}</name>{''.join(marks)}</Document></kml>")


def flight_kml(ep: Episode, world) -> str:
    alt0 = world.origin[2]
    pts = " ".join(f"{lon:.8f},{lat:.8f},{alt0 + a:.1f}" for (lat, lon), a in
                   ((to_latlon(e, n, world.origin), a) for e, n, a in
                    zip(ep.sig["east"], ep.sig["north"], ep.sig["alt"])))
    return ("<?xml version='1.0' encoding='UTF-8'?><kml xmlns='http://www.opengis.net/kml/2.2'><Document>"
            f"<Placemark><name>{escape(ep.meta.get('episode', 'flight'))}</name><LineString>"
            f"<altitudeMode>absolute</altitudeMode><coordinates>{pts}</coordinates></LineString></Placemark>"
            "</Document></kml>")


def tlog(ep: Episode, world, path, start_unix: float | None = None) -> None:
    """MAVLink v2 telemetry log: HEARTBEAT, GLOBAL_POSITION_INT, ATTITUDE, SYS_STATUS
    at the signal rate; each record is an 8-byte big-endian microsecond timestamp
    followed by the packet (QGC's .tlog format)."""
    import struct

    from pymavlink.dialects.v20 import common as mav
    out = open(path, "wb")
    m = mav.MAVLink(None, srcSystem=1, srcComponent=1)
    t0 = start_unix if start_unix is not None else time.time()
    s = ep.sig
    alt0 = world.origin[2]
    prev = None
    for i, t in enumerate(ep.t):
        e, n, a = float(s["east"][i]), float(s["north"][i]), float(s["alt"][i])
        lat, lon = to_latlon(e, n, world.origin)
        if prev is not None and t > prev[0]:
            dt = t - prev[0]
            ve, vn, vu = (e - prev[1]) / dt, (n - prev[2]) / dt, (a - prev[3]) / dt
        else:
            ve = vn = vu = 0.0
        prev = (t, e, n, a)
        yaw = math.atan2(ve, vn) if math.hypot(ve, vn) > 0.3 else 0.0     # NED heading
        armed = s["landed"][i] < 0.5
        msgs = [
            m.heartbeat_encode(mav.MAV_TYPE_QUADROTOR, mav.MAV_AUTOPILOT_PX4,
                               mav.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED | (mav.MAV_MODE_FLAG_SAFETY_ARMED if armed else 0),
                               0, mav.MAV_STATE_ACTIVE if armed else mav.MAV_STATE_STANDBY),
            m.global_position_int_encode(int(t * 1000), int(lat * 1e7), int(lon * 1e7), int((alt0 + a) * 1000),
                                         int(a * 1000), int(vn * 100), int(ve * 100), int(-vu * 100),
                                         int((math.degrees(yaw) % 360) * 100)),
            m.attitude_encode(int(t * 1000), 0.0, 0.0, yaw, 0.0, 0.0, 0.0),
            m.sys_status_encode(0, 0, 0, 0, 16000, -1, int(s["battery"][i]), 0, 0, 0, 0, 0, 0),
        ]
        for msg in msgs:
            out.write(struct.pack(">Q", int((t0 + t) * 1e6)) + msg.pack(m))
    out.close()
