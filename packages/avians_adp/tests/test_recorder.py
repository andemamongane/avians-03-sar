import json
import warnings

from avians_adp import EpisodeRecorder, decisions, load, validate


def test_round_trip(tmp_path):
    rec = EpisodeRecorder(tmp_path, "ep1", {"seed": 1})
    rec.brief("system", [{"name": "takeoff"}])
    rec.observation("TIME t=0", 0.0, decision=1)
    rec.message("taking off", 0.0)
    rec.action("takeoff", {"alt": 30}, "start", 0.0, latency_s=2.5)
    rec.tool_result("started takeoff", 2.5, accepted=True)
    rec.signals(0.0, {"alt": 0.0})
    rec.event(11.0, "skill_done", "takeoff: at 30 m")
    rec.finish({"robustness": 1.0}, {"end": "horizon"})
    tj = load(tmp_path)
    kinds = [s.class_ for s in tj.content]
    assert kinds == ["text_observation", "message_action", "api_action", "text_observation"]
    assert tj.content[2].function == "takeoff" and tj.content[2].details["latency_s"] == 2.5
    assert json.loads((tmp_path / "score.json").read_text())["robustness"] == 1.0


def test_record_follows_adp_1_3_1(tmp_path):
    rec = EpisodeRecorder(tmp_path, "ep", {"seed": 1})
    rec.brief("system", [{"name": "takeoff"}, {"name": "keep"}])
    rec.observation("P1", 0.0, decision=1)
    rec.message("text", 0.0, thinking="a summary", decision=1)
    rec.action("takeoff", {"alt": 30}, "go", 0.0, t_obs=0.0, t_apply=2.0)
    rec.tool_result("started", 2.0, accepted=True, decision=1)
    rec.observation("P2", 45.0, decision=2)
    rec.message("", 45.0, thinking="", decision=2)
    rec.action("keep", {}, "nothing to do", 45.0, t_obs=45.0, t_apply=46.0)
    rec.tool_result("kept", 46.0, accepted=True, decision=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")             # a valid record must not warn
        rec.finish({"robustness": 1.0}, {"end": "horizon"})
    raw = json.loads((tmp_path / "trajectory.json").read_text())
    assert raw["schema_version"] == "1.3.1" and raw["available_apis"] == ["takeoff", "keep"]
    obs, msg, act, res = raw["content"][:4]
    assert obs["source"] == "environment" and "tool_call_id" not in obs
    assert msg["reasoning_content"] == "a summary" and "thinking" not in msg["details"]
    assert act["tool_call_id"] == res["tool_call_id"] == "call_000001" and res["source"] == "environment"
    assert act["details"]["decision"] == 1 and raw["content"][6]["details"]["decision"] == 2
    assert "reasoning_content" not in raw["content"][5]      # empty thinking is not recorded
    assert raw["content"][6]["tool_call_id"] == "call_000002"
    tj = load(tmp_path)
    assert validate(tj) == []
    assert [(d.n, d.action.function, d.accepted) for d in decisions(tj)] == [(1, "takeoff", True), (2, "keep", True)]


def test_episode_ending_mid_decision_still_validates(tmp_path):
    rec = EpisodeRecorder(tmp_path, "ep", {"seed": 1})
    rec.brief("system", [{"name": "keep"}])
    rec.observation("P1", 0.0, decision=1)
    rec.action("keep", {}, "wait", 0.0)
    rec.tool_result("kept", 1.0, accepted=True, decision=1)
    rec.observation("P2", 45.0, decision=2)
    rec.action("keep", {}, "wait", 45.0)           # the sim ends before this is applied
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rec.finish({"robustness": 0.0}, {"end": "battery"})
    tj = load(tmp_path)
    assert tj.content[-1].tool_call_id is None and tj.content[1].tool_call_id == "call_000001"
    assert validate(tj) == [] and not decisions(tj)[-1].finished


def test_finish_writes_then_warns_on_a_bad_record(tmp_path):
    rec = EpisodeRecorder(tmp_path, "ep", {"seed": 1})
    rec.brief("system", [{"name": "keep"}])
    rec.observation("P1", 0.0, decision=1)
    rec.action("land", {}, "not in the tools", 0.0)     # available_apis does not cover it
    rec.tool_result("rejected", 1.0, accepted=False, decision=1)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rec.finish({}, {})
    assert (tmp_path / "trajectory.json").exists()
    assert len(w) == 1 and "available_apis" in str(w[0].message)
