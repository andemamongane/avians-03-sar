import json

from avians_adp import EpisodeRecorder, load


def test_round_trip(tmp_path):
    rec = EpisodeRecorder(tmp_path, "ep1", {"seed": 1})
    rec.brief("system", [{"name": "takeoff"}])
    rec.observation("TIME t=0", 0.0, decision=1)
    rec.message("taking off", 0.0)
    rec.action("takeoff", {"alt": 30}, "start", 0.0, latency_s=2.5)
    rec.tool_result("started takeoff", 2.5)
    rec.signals(0.0, {"alt": 0.0})
    rec.event(11.0, "skill_done", "takeoff: at 30 m")
    rec.finish({"robustness": 1.0}, {"end": "horizon"})
    tj = load(tmp_path)
    kinds = [s.class_ for s in tj.content]
    assert kinds == ["text_observation", "message_action", "api_action", "text_observation"]
    assert tj.content[2].function == "takeoff" and tj.content[2].details["latency_s"] == 2.5
    assert json.loads((tmp_path / "score.json").read_text())["robustness"] == 1.0
