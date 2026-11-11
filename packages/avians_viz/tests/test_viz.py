import xml.etree.ElementTree as ET

from avians_agent.loop import run_episode
from avians_agent.mission import Mission, load_config
from avians_agent.pilots import ScriptedPilot
from avians_viz import Episode, flight_kml, summary_png, tlog, world_kml


def test_exports(tmp_path, world):
    m = Mission.load("missions/sar_supply.yaml")
    cfg = load_config("agent.yaml")
    m.horizon_s = 120
    run_episode(m, ScriptedPilot(), 1, tmp_path, cfg, load_config("latency.yaml"))
    ep = Episode.load(tmp_path)
    assert ep.decisions and len(ep.t) > 100
    ET.fromstring(world_kml(world).split("?>", 1)[1])
    ET.fromstring(flight_kml(ep, world).split("?>", 1)[1])
    tlog(ep, world, tmp_path / "f.tlog")
    from pymavlink import mavutil
    f = mavutil.mavlink_connection(str(tmp_path / "f.tlog"))
    types = set()
    while (msg := f.recv_match()) is not None:
        types.add(msg.get_type())
    assert {"HEARTBEAT", "GLOBAL_POSITION_INT", "SYS_STATUS"} <= types
    assert summary_png(tmp_path, world).exists()


def test_two_drone_exports_and_live_feed(tmp_path, world):
    import socket
    from avians_viz import LiveMavlink, render
    m = Mission.load("missions/sar_supply_2drones.yaml")
    m.horizon_s = 90
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(5)
    port = rx.getsockname()[1]
    run_episode(m, ScriptedPilot(), 2, tmp_path, load_config("agent.yaml"), load_config("latency.yaml"),
                make_pilot=lambda did: ScriptedPilot(), qgc_live=f"127.0.0.1:{port}")
    from pymavlink import mavutil
    mav = mavutil.mavlink.MAVLink(None)
    systems = set()
    for _ in range(200):
        data, _ = rx.recvfrom(4096)
        for msg in mav.parse_buffer(data) or []:
            systems.add(msg.get_srcSystem())
    assert systems == {1, 2}
    ep = Episode.load(tmp_path)
    assert ep.drones == ["d1", "d2"] and {d["drone"] for d in ep.decisions} == {"d1", "d2"}
    tlog(ep, world, tmp_path / "f.tlog")
    f = mavutil.mavlink_connection(str(tmp_path / "f.tlog"))
    seen, texts = set(), 0
    while (msg := f.recv_match()) is not None:
        seen.add(msg.get_srcSystem())
        texts += msg.get_type() == "STATUSTEXT"
    assert seen == {1, 2} and texts > 0
    out = render(tmp_path, world, speed=30)
    assert out.exists() and out.stat().st_size > 10_000
