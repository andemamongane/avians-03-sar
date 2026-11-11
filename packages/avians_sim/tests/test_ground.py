import json
import urllib.request

from avians_sim import Executor, FastSim
from avians_sim.ground_live import Console


def test_report_is_acknowledged_with_the_crews_eta(world, limits):
    sim = FastSim(world, 0, limits, ground={"latency_s": 2.0, "answers": {"eta": "operator: 9 minutes"}})
    ex = Executor(sim)
    sim.vehicle.s.landed, sim.vehicle.s.alt = False, 30
    ok, text = ex.start("report", {"kind": "person_found", "east": 100, "north": 100})
    assert ok and "sent" in text
    assert sim.world_state.help.dispatched_t is not None          # dispatch happens on receipt
    kinds = []
    for _ in range(40):
        ex.tick()
        kinds += [e.kind for e in sim.ctx.events]
        sim.ctx.events.clear()
    assert "message" in kinds                                      # the ack, 2 s later
    ok, text = ex.start("request_help", {"question": "what is the helicopter ETA?"})
    assert ok and "operator will answer" in text
    for _ in range(150):
        ex.tick()
    assert any(m["kind"] == "answer" and "9 minutes" in m["text"] for m in sim.ground.summary())


def test_out_of_range_has_no_link(world, limits):
    sim = FastSim(world, 0, limits, ground={"range_m": 50})
    sim.vehicle.s.landed, sim.vehicle.s.alt, sim.vehicle.s.east = False, 30, 400
    assert "no link" in sim.actuators.report("status", None, None, "hello")


def test_live_console_round_trip(world, limits):
    sim = FastSim(world, 0, limits)
    ex = Executor(sim)
    con = Console(sim.ground, lambda: "state", lambda: sim.t, port=0)
    try:
        port = con.server.server_address[1]
        req = urllib.request.Request(f"http://127.0.0.1:{port}/message", json.dumps(
            {"text": "base: the hiker was seen near the creek", "kind": "tasking"}).encode(),
            {"content-type": "application/json"})
        assert json.loads(urllib.request.urlopen(req, timeout=5).read())["ok"]
        texts = []
        for _ in range(40):
            ex.tick()
            texts += [e.text for e in sim.ctx.events if e.kind == "message"]
            sim.ctx.events.clear()
        assert any("seen near the creek" in t for t in texts)
        log = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/log?since=0", timeout=5).read())
        assert log["messages"] and log["messages"][0]["sender"] == "operator"
        assert b"Ground station" in urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5).read()
    finally:
        con.close()


def test_gazebo_replay_moves_every_drone(tmp_path, world, monkeypatch):
    """GazeboControl.replay against a recording stand-in for docker: one visual
    drone per recorded drone, pose lines for each every row."""
    import subprocess as sp
    from avians_agent.loop import run_episode
    from avians_agent.mission import Mission, load_config
    from avians_agent.pilots import ScriptedPilot
    from avians_sim import gazebo
    m = Mission.load("missions/sar_supply_2drones.yaml")
    m.horizon_s = 20
    ep = tmp_path / "ep"
    run_episode(m, ScriptedPilot(), 1, ep, load_config("agent.yaml"), load_config("latency.yaml"),
                make_pilot=lambda did: ScriptedPilot())
    calls, written = [], []

    def run(cmd, **kw):
        calls.append(" ".join(cmd))
        out = "/clock\n/world/impalpable_greyness/pose/info\n" if "gz topic -l" in calls[-1] else ""
        out = "1.0 2.0 0.2 0.0\n" if "avians_gz.py pose" in calls[-1] else out
        return sp.CompletedProcess(cmd, 0, out, "")

    class P:
        class _In:
            def write(self, b):
                written.append(b.decode())
            def flush(self):
                pass
            def close(self):
                pass
        stdin, stdout = _In(), None
        def wait(self, timeout=None):
            return 0
        def kill(self):
            pass

    monkeypatch.setattr(gazebo.time, "sleep", lambda s: None)
    monkeypatch.setattr(gazebo.time, "time", iter(float(k) for k in range(10 ** 6)).__next__)
    monkeypatch.setattr(gazebo.shutil, "which", lambda x: None)          # no ffmpeg: skip recording
    ctl = gazebo.GazeboControl({}, tmp_path / "gz", run=run, popen=lambda *a, **k: P())
    ctl.replay(ep, world, speed=1000)
    spawned = [c for c in calls if "avians_gz.py spawn" in c]
    assert any("avians_replay_drone_d1" in c for c in spawned) and any("avians_replay_drone_d2" in c for c in spawned)
    text = "".join(written)
    assert "avians_replay_drone_d1 " in text and "avians_replay_drone_d2 " in text
