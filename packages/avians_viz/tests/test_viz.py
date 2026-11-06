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
