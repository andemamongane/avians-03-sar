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
    """The flight path(s) as KML lines, one per drone."""
    alt0 = world.origin[2]
    marks = []
    for did in ep.drones:
        sg = ep.drone(did)
        pts = " ".join(f"{lon:.8f},{lat:.8f},{alt0 + a:.1f}" for (lat, lon), a in
                       ((to_latlon(e, n, world.origin), a) for e, n, a in zip(sg["east"], sg["north"], sg["alt"])))
        name = f"{ep.meta.get('episode', 'flight')} {did}".strip()
        marks.append(f"<Placemark><name>{escape(name)}</name><LineString><altitudeMode>absolute</altitudeMode>"
                     f"<coordinates>{pts}</coordinates></LineString></Placemark>")
    return ("<?xml version='1.0' encoding='UTF-8'?><kml xmlns='http://www.opengis.net/kml/2.2'><Document>"
            + "".join(marks) + "</Document></kml>")


class _Mav:
    """MAVLink v2 encoding of one drone's state (shared by the .tlog writer and the live feed)."""

    def __init__(self, sysid: int):
        from pymavlink.dialects.v20 import common as mav
        self.mav = mav
        self.m = mav.MAVLink(None, srcSystem=sysid, srcComponent=1)
        self.prev = None

    def state(self, t, e, n, a, battery, landed, origin) -> list[bytes]:
        mav, m = self.mav, self.m
        lat, lon = to_latlon(e, n, origin)
        if self.prev is not None and t > self.prev[0]:
            dt = t - self.prev[0]
            ve, vn, vu = (e - self.prev[1]) / dt, (n - self.prev[2]) / dt, (a - self.prev[3]) / dt
        else:
            ve = vn = vu = 0.0
        self.prev = (t, e, n, a)
        yaw = math.atan2(ve, vn) if math.hypot(ve, vn) > 0.3 else 0.0     # NED heading
        armed = not landed
        msgs = [
            m.heartbeat_encode(mav.MAV_TYPE_QUADROTOR, mav.MAV_AUTOPILOT_PX4,
                               mav.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED | (mav.MAV_MODE_FLAG_SAFETY_ARMED if armed else 0),
                               0, mav.MAV_STATE_ACTIVE if armed else mav.MAV_STATE_STANDBY),
            m.global_position_int_encode(int(t * 1000), int(lat * 1e7), int(lon * 1e7), int((origin[2] + a) * 1000),
                                         int(a * 1000), int(vn * 100), int(ve * 100), int(-vu * 100),
                                         int((math.degrees(yaw) % 360) * 100)),
            m.attitude_encode(int(t * 1000), 0.0, 0.0, yaw, 0.0, 0.0, 0.0),
            m.sys_status_encode(0, 0, 0, 0, 16000, -1, int(battery), 0, 0, 0, 0, 0, 0),
        ]
        return [x.pack(m) for x in msgs]

    def text(self, text: str, severity: int = 6) -> bytes:
        """STATUSTEXT (QGC's message panel); MAVLink caps it at 50 characters per message."""
        return self.m.statustext_encode(severity, text[:50].encode("ascii", "replace")).pack(self.m)


def _texts(ep: Episode) -> list[tuple[float, str, str]]:
    """(t, drone, text) for QGC's message panel: the pilot's decisions and the main events."""
    out = []
    for d in ep.decisions:
        if d["tool"] != "keep" and d.get("t_apply") is not None:
            out.append((float(d["t_apply"]), d.get("drone") or "", f"{d['tool']}: {d['reason']}"))
    for e in ep.events:
        if e["kind"] in ("found", "kit_landed", "zone", "help", "reflex", "message", "news"):
            out.append((float(e["t"]), e.get("drone") or "", e["text"]))
    return sorted(out)


def tlog(ep: Episode, world, path, start_unix: float | None = None) -> None:
    """MAVLink v2 telemetry log: per drone (system id 1, 2, ...) HEARTBEAT,
    GLOBAL_POSITION_INT, ATTITUDE, SYS_STATUS at the signal rate, plus the
    pilot's decisions and the main events as STATUSTEXT. Each record is an
    8-byte big-endian microsecond timestamp followed by the packet (QGC's
    .tlog format); QGC shows every drone and lists the texts in its messages."""
    import struct
    t0 = start_unix if start_unix is not None else time.time()
    ids = ep.drones
    encs = {did: _Mav(k + 1) for k, did in enumerate(ids)}
    texts = _texts(ep)
    ti = 0
    with open(path, "wb") as out:
        for i, t in enumerate(ep.t):
            while ti < len(texts) and texts[ti][0] <= t:
                tt, did, text = texts[ti]
                enc = encs.get(did) or next(iter(encs.values()))
                out.write(struct.pack(">Q", int((t0 + tt) * 1e6)) + enc.text(text))
                ti += 1
            for did, enc in encs.items():
                s = ep.drone(did)
                for pkt in enc.state(t, float(s["east"][i]), float(s["north"][i]), float(s["alt"][i]),
                                     float(s["battery"][i]), s["landed"][i] >= 0.5, world.origin):
                    out.write(struct.pack(">Q", int((t0 + t) * 1e6)) + pkt)


class LiveMavlink:
    """Streams an episode to QGroundControl as it flies: MAVLink over UDP
    (QGC listens on 14550 by default), one system per drone, decisions and
    events as STATUSTEXT. Use with --realtime so QGC sees real-time motion.

        avians fly missions/sar_supply_2drones.yaml --realtime --qgc-live 127.0.0.1:14550
    """

    def __init__(self, host: str, port: int, ids: list[str], origin, rate_hz: float = 5.0):
        import socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr, self.origin, self.dt = (host, port), origin, 1.0 / rate_hz
        self.encs = {did: _Mav(k + 1) for k, did in enumerate(ids)}
        self._next = 0.0
        self.sent = 0

    def tick(self, t: float, states: dict) -> None:
        """states: drone id -> (east, north, alt, battery, landed)."""
        if t + 1e-9 < self._next:
            return
        self._next = t + self.dt
        for did, (e, n, a, bat, landed) in states.items():
            enc = self.encs.get(did)
            if enc is None:
                continue
            for pkt in enc.state(t, e, n, a, bat, landed, self.origin):
                self.sock.sendto(pkt, self.addr)
                self.sent += 1

    def text(self, did: str, text: str) -> None:
        enc = self.encs.get(did) or next(iter(self.encs.values()))
        self.sock.sendto(enc.text(text), self.addr)
        self.sent += 1

    def close(self) -> None:
        self.sock.close()
