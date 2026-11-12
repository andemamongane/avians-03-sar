"""Phase 4: router labels, the router's input, offline student calls and the build's router files."""
import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
from avians_adp import EpisodeRecorder, load, sft
from avians_adp.build import BuildConfig, build
from avians_adp.routing import CHOICES, TOLERANCES, args_match, event_kinds, label, router_examples
from avians_adp.student_calls import read_calls, run_student, write_calls

TOOLS = [{"name": n, "description": n, "input_schema": {"type": "object", "properties": {}}}
         for n in ("takeoff", "search_region", "goto", "keep")]


def state(skill="search_region", battery=80.0):
    return {"skill": skill, "skill_status": "running", "battery": battery, "payload": False, "released": False,
            "help_arrived": False, "belief": {"found": False, "conf": 0.1}, "regions": []}


def record(out: Path, calls, *, ep_id="ep_s0", seed=0, events_between=None, meta=None) -> None:
    """calls: (function, kwargs, extra details) per decision. events_between[n] = event kinds recorded
    after decision n's result (they belong to decision n + 1)."""
    rec = EpisodeRecorder(out, ep_id, {"seed": seed, "pilot": "haiku", **(meta or {})})
    rec.brief("SYSTEM", TOOLS)
    for n, (fn, kwargs, extra) in enumerate(calls, start=1):
        t = 45.0 * (n - 1)
        rec.observation(f"PROMPT {n}", t, decision=n)
        rec.action(fn, kwargs, f"why {n}", t, t_obs=t, t_apply=t + 1,
                   **{"model": "claude-haiku-5-5", "by": "local", "error": "", "state": state(), **extra})
        rec.tool_result("ok", t + 1, accepted=True, decision=n)
        for kind in (events_between or {}).get(n, []):
            rec.event(t + 2, kind, f"{kind} happened")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rec.finish({"robustness": 0.5, "satisfied": True}, {"robustness": 0.5, "satisfied": True})


# ------------------------------------------------------------------ matching and labels

@pytest.mark.parametrize("a, b, ok", [
    ({"region": "r1"}, {"region": "r1"}, True),
    ({"region": "r1"}, {"region": "r2"}, False),
    ({"region": "r1", "reason": "x", "notes": "n"}, {"region": "r1", "reason": "y"}, True),   # ignored
    ({"east": 100.0, "north": 50.0}, {"east": 109.0, "north": 41.0}, True),                 # within 10 m
    ({"east": 100.0}, {"east": 111.0}, False),
    ({"alt": 30}, {"alt": 34.9}, True), ({"alt": 30}, {"alt": 35.1}, False),                # 5 m
    ({"laps": 2}, {"laps": 3}, False),                                                      # not listed: exact
    ({"region": "r1", "alt": 30}, {"region": "r1"}, False),                                 # one side only
    ({"region": "r1"}, {"region": "r1", "alt": 30}, False),
    ({"flag": True}, {"flag": 1}, False),                                                   # bool is not a number
    ({"alt": float("nan")}, {"alt": float("nan")}, False),
    ({}, {}, True),
])
def test_args_match(a, b, ok):
    assert args_match(a, b)[0] is ok


def test_tolerances_can_be_replaced():
    assert args_match({"laps": 2}, {"laps": 3}, {"laps": 1})[0]
    assert not args_match({"alt": 30}, {"alt": 34}, {})[0]          # an empty table: everything exact
    assert TOLERANCES["east"] == 10.0


@pytest.mark.parametrize("teacher, student, lab", [
    ({"tool": "keep", "args": {}}, {"tool": "goto", "args": {"east": 1}}, "commit"),
    ({"tool": "keep", "args": {}}, None, "commit"),
    ({"tool": "search_region", "args": {"region": "r1"}}, {"tool": "search_region", "args": {"region": "r1"}}, "local"),
    ({"tool": "goto", "args": {"east": 0.0}}, {"tool": "goto", "args": {"east": 4.0}}, "local"),
    ({"tool": "goto", "args": {"east": 0.0}}, {"tool": "goto", "args": {"east": 40.0}}, "escalate"),
    ({"tool": "search_region", "args": {"region": "r1"}}, {"tool": "keep", "args": {}}, "escalate"),
    ({"tool": "search_region", "args": {"region": "r1"}}, {"tool": "escalate", "args": {}}, "escalate"),
    ({"tool": "search_region", "args": {"region": "r1"}}, None, "escalate"),
    ({"tool": "search_region", "args": {"region": "r1"}}, {"tool": None, "args": {}, "error": "boom"}, "escalate"),
])
def test_label(teacher, student, lab):
    assert label(teacher, student)[0] == lab


# ------------------------------------------------------------------ router examples

def test_router_examples(tmp_path):
    """Replaces the skipped test in test_todo.py: keep, match, near-match within tolerance, mismatch."""
    record(tmp_path / "ep", [("takeoff", {"alt": 30}, {}),
                             ("keep", {}, {}),
                             ("goto", {"east": 100.0, "north": 0.0}, {}),
                             ("search_region", {"region": "r2"}, {})])
    tj = load(tmp_path / "ep")
    student = [{"decision": 1, "tool": "takeoff", "args": {"alt": 30}},
               {"decision": 2, "tool": "search_region", "args": {"region": "r1"}},
               {"decision": 3, "tool": "goto", "args": {"east": 104.0, "north": -3.0}},
               {"decision": 4, "tool": "search_region", "args": {"region": "r3"}}]
    exs = sft.router_examples(tj, student)
    assert [e["label"] for e in exs] == ["local", "commit", "local", "escalate"]
    assert all(e["options"] == list(CHOICES) for e in exs)
    assert exs[3]["meta"]["why"] == "region: 'r2' vs 'r3'"
    assert exs[3]["meta"]["teacher"] == {"tool": "search_region", "args": {"region": "r2"}}
    assert exs[0]["id"] == "ep_s0#d1" and exs[0]["meta"]["seed"] == 0 and exs[0]["text"] is None
    # students matched by decision number, not position
    assert [e["label"] for e in sft.router_examples(tj, list(reversed(student)))] == [e["label"] for e in exs]


def test_router_examples_skip_keeps_the_teacher_never_chose(tmp_path):
    record(tmp_path / "ep", [("takeoff", {"alt": 30}, {}),
                             ("keep", {}, {"error": "529: overloaded"}),
                             ("keep", {}, {"model": "router", "by": "commit"}),
                             ("search_region", {"region": "r1"}, {"model": "student-1b"})])
    tj = load(tmp_path / "ep")
    exs, skipped = router_examples(tj, [])
    assert [e["meta"]["decision"] for e in exs] == [1, 4] and skipped == {"api_error": 1, "router_commit": 1}
    exs, skipped = router_examples(tj, [], models=("claude-haiku-5-5",))
    assert [e["meta"]["decision"] for e in exs] == [1] and skipped["model"] == 1


def test_features_text_and_events(tmp_path):
    record(tmp_path / "ep", [("takeoff", {"alt": 30}, {}), ("keep", {}, {}), ("keep", {}, {})],
           events_between={1: ["skill_done", "zone"], 2: []})
    tj = load(tmp_path / "ep")
    seen = []

    def features(st, events):
        seen.append(st)
        return f"skill={st['skill']} events={','.join(e.kind for e in events) or 'none'}"
    exs = sft.router_examples(tj, [], features=features)
    assert [e["events"] for e in exs] == [[], ["skill_done", "zone"], []]
    assert exs[1]["text"] == "skill=search_region events=skill_done,zone"
    assert all(e["meta"]["events_exact"] for e in exs) and seen[0] == state()


def test_recorder_stores_events_since_the_last_prompt(tmp_path):
    rec = EpisodeRecorder(tmp_path, "ep", {"seed": 0})
    rec.brief("S", TOOLS)
    rec.event(0.0, "start", "before any prompt")
    rec.observation("P1", 0.0, decision=1)
    rec.action("takeoff", {}, "go", 0.0)
    rec.event(0.5, "skill_done", "while waiting")               # after the prompt: belongs to the next one
    rec.tool_result("ok", 1.0, accepted=True, decision=1)
    rec.event(2.0, "zone", "later")
    rec.observation("P2", 45.0, decision=2)
    rec.action("keep", {}, "wait", 45.0)
    rec.tool_result("kept", 46.0, accepted=True, decision=2)
    rec.finish({}, {})
    tj = load(tmp_path)
    prompts = [s for s in tj.content if s.class_ == "text_observation" and s.tool_call_id is None]
    assert [p.details["events"] for p in prompts] == [["start"], ["skill_done", "zone"]]


def test_old_records_fall_back_to_a_time_window(tmp_path):
    record(tmp_path / "ep", [("takeoff", {"alt": 30}, {}), ("keep", {}, {})], events_between={1: ["skill_done"]})
    tj = load(tmp_path / "ep")
    for s in tj.content:
        if s.class_ == "text_observation" and s.tool_call_id is None:
            s.details.pop("events")
    rows = [{"t": 2.0, "kind": "skill_done"}, {"t": 50.0, "kind": "after"}]
    assert event_kinds(tj, rows) == {0: ([], False), 1: (["skill_done"], False)}


# ------------------------------------------------------------------ offline student calls

def test_run_student(tmp_path):
    record(tmp_path / "ep", [("takeoff", {"alt": 30}, {}), ("keep", {}, {}), ("goto", {"east": 1.0}, {})])
    tj = load(tmp_path / "ep")
    got = []

    def decide(system, prompt, tools, st):
        got.append((system, prompt, [t["name"] for t in tools], st["skill"]))
        if prompt == "PROMPT 1":
            return {"tool": "takeoff", "args": {"alt": 30, "reason": "go", "notes": "x"}}
        raise RuntimeError("model down")
    esc = {"name": "escalate", "description": "ask the teacher", "input_schema": {}}
    calls = run_student(tj, decide, model="student-1b", extra_tools=[esc, {"name": "keep"}])
    # the teacher's keep (decision 2) is commit whatever the student says: it is not asked
    assert [g[1] for g in got] == ["PROMPT 1", "PROMPT 3"]
    assert got[0] == ("SYSTEM", "PROMPT 1", ["takeoff", "search_region", "goto", "keep", "escalate"], "search_region")
    assert calls[0] == {"episode": "ep_s0", "decision": 1, "tool": "takeoff", "args": {"alt": 30},
                        "model": "student-1b"}
    assert calls[1]["decision"] == 3 and calls[1]["tool"] is None and "model down" in calls[1]["error"]
    write_calls(tmp_path / "ep", calls)
    assert read_calls(tmp_path / "ep") == calls and read_calls(tmp_path / "nowhere") is None
    exs, skipped = router_examples(tj, calls)
    assert [e["label"] for e in exs] == ["local", "commit"] and skipped == {"student error": 1}


def test_api_failures_and_silent_replies_from_the_student(tmp_path):
    """Review: AnthropicPilot returns keep + error on an API failure, and keep + reason "no tool call" when the
    model answers in text. The first says nothing about the student (skipped); the second is a real failure."""
    record(tmp_path / "ep", [("takeoff", {"alt": 30}, {}), ("goto", {"east": 1.0}, {}), ("goto", {"east": 2.0}, {})])
    tj = load(tmp_path / "ep")
    replies = {"PROMPT 1": SimpleNamespace(tool="keep", args={}, reason="API error", error="529: overloaded"),
               "PROMPT 2": SimpleNamespace(tool="keep", args={}, reason="no tool call", error=""),
               "PROMPT 3": SimpleNamespace(tool="goto", args={"east": 2.0, "reason": "r"}, reason="r", error="")}
    calls = run_student(tj, lambda sy, pr, to, st: replies[pr])
    assert [(c["tool"], bool(c.get("error")), bool(c.get("no_tool_call"))) for c in calls] == [
        (None, True, False), (None, False, True), ("goto", False, False)]
    exs, skipped = router_examples(tj, calls)
    assert [(e["meta"]["decision"], e["label"], e["meta"]["why"]) for e in exs] == [
        (2, "escalate", "student made no tool call"), (3, "local", "match")]
    assert skipped == {"student error": 1}


def test_a_routed_episode_labels_only_the_teachers_decisions(tmp_path):
    """Review: with a router, by="local" decisions were made by the student itself; neither the pilot set nor the
    router set may treat them as the teacher's."""
    record(tmp_path / "ep", [("takeoff", {"alt": 30}, {"by": "escalate"}),
                             ("search_region", {"region": "r1"}, {"by": "local", "model": "student-1b"}),
                             ("keep", {}, {"by": "commit", "model": "router"}),
                             ("goto", {"east": 5.0}, {"by": "escalate"})],
           meta={"router": "SystemOneRouter", "remote_pilot": "haiku"})
    tj = load(tmp_path / "ep")
    kinds = [e["meta"]["kind"] for e in sft.pilot_examples(tj)]
    assert kinds == ["call", "student", "router_commit", "call"]
    exs, skipped = router_examples(tj, [])
    assert [e["meta"]["decision"] for e in exs] == [1, 4] and skipped == {"student": 1, "router_commit": 1}
    assert [d.n for d in __import__("avians_adp.student_calls", fromlist=["x"]).needs_student(tj)] == [1, 4]
    r = build(tmp_path, tmp_path.parent / f"{tmp_path.name}_out", BuildConfig(ratios=(1.0, 0.0, 0.0)))
    assert r["examples"]["train"] == 2 and r["removed"]["decisions"] == {"student": 1, "router_commit": 1}
    # unrouted episode: by="local" is the teacher (no router, the loop always writes local)
    record(tmp_path / "plain", [("search_region", {"region": "r1"}, {"by": "local"})])
    assert sft.pilot_examples(load(tmp_path / "plain"))[0]["meta"]["kind"] == "call"


def test_stale_student_calls_are_ignored(tmp_path):
    eps = tmp_path / "eps"
    record(eps / "a", [("takeoff", {"alt": 30}, {})], ep_id="a_s0")
    write_calls(eps / "a", [{"episode": "old_s0", "decision": 1, "tool": "takeoff", "args": {"alt": 30}}])
    r = build(eps, tmp_path / "out")
    assert r["router"]["episodes"] == 0 and any("belongs to ['old_s0']" in w for w in r["warnings"])
    write_calls(eps / "a", [{"decision": 1, "tool": "takeoff", "args": {"alt": 30}}])     # no provenance: accepted
    assert build(eps, tmp_path / "out2")["router"]["examples"]["train"] == 1


# ------------------------------------------------------------------ the build

def test_build_writes_router_files_with_the_pilot_splits(tmp_path):
    eps = tmp_path / "eps"
    plan = [("takeoff", {"alt": 30}, {}), ("keep", {}, {}), ("search_region", {"region": "r2"}, {})]
    record(eps / "a", plan, ep_id="a_s0", seed=0)          # train
    record(eps / "b", plan, ep_id="b_s4", seed=4)          # test
    record(eps / "c", plan, ep_id="c_s7", seed=7)          # val, but no student calls
    student = [{"decision": 1, "tool": "takeoff", "args": {"alt": 30}},
               {"decision": 2, "tool": "keep", "args": {}},
               {"decision": 3, "tool": "search_region", "args": {"region": "r9"}}]
    write_calls(eps / "a", student)
    write_calls(eps / "b", student[:2])                       # the student never answered decision 3
    r = build(eps, tmp_path / "out")
    assert r["router"]["episodes"] == 2
    assert r["router"]["examples"] == {"train": 3, "val": 0, "test": 3}
    assert r["router"]["labels"] == {"commit": 2, "local": 2, "escalate": 2}
    rows = [json.loads(x) for x in (tmp_path / "out" / "router" / "train.jsonl").read_text().splitlines()]
    assert set(rows[0]) == {"id", "text", "label", "options"} and [x["label"] for x in rows] == [
        "local", "commit", "escalate"]
    meta = [json.loads(x) for x in (tmp_path / "out" / "router" / "test.meta.jsonl").read_text().splitlines()]
    assert meta[2]["why"] == "no student call" and meta[2]["state"]["skill"] == "search_region"
    pilot_ids = {json.loads(x)["episode"] for x in (tmp_path / "out" / "pilot" / "train.meta.jsonl").read_text().splitlines()}
    assert pilot_ids == {"a_s0"}                              # router and pilot sets share the split
    if _has_agent():                                          # the live router's feature string
        assert rows[0]["text"].startswith("skill=search_region:running battery=80")


def test_build_without_student_calls_writes_no_router_files(tmp_path):
    record(tmp_path / "eps" / "a", [("takeoff", {"alt": 30}, {})])
    r = build(tmp_path / "eps", tmp_path / "out")
    assert r["router"]["episodes"] == 0 and not (tmp_path / "out" / "router").exists()
    (tmp_path / "out" / "router").mkdir()
    (tmp_path / "out" / "router" / "old.jsonl").write_text("stale")
    build(tmp_path / "eps", tmp_path / "out", force=True)
    assert not (tmp_path / "out" / "router").exists()         # --force clears a previous router set


def _has_agent() -> bool:
    try:
        import avians_agent.router  # noqa: F401
    except ImportError:
        return False
    return True


# ------------------------------------------------------------------ against the live router (avians_agent)

def _fly(tmp_path, seed, router=None):
    from avians_agent.loop import run_episode
    from avians_agent.mission import Mission, load_config
    from avians_agent.pilots import ScriptedPilot
    root = Path(__file__).resolve().parents[3]
    m = Mission.load(root / "missions" / "sar_supply.yaml")
    d = tmp_path / f"s{seed:04d}"
    run_episode(m, ScriptedPilot(), seed, d, load_config("agent.yaml"), load_config("latency.yaml"), router=router)
    return d


def test_rebuilt_features_equal_what_the_live_router_saw(tmp_path):
    """4.2c: the text in the router set is exactly the string the router read in flight."""
    pytest.importorskip("avians_agent")
    from avians_agent.router import FixedRouter, features

    class Recording(FixedRouter):
        def __init__(self):
            super().__init__("escalate")
            self.seen = []

        def route(self, f):
            self.seen.append(f)
            return super().route(f)
    r = Recording()
    d = _fly(tmp_path, 4, router=r)                       # seed 4: zone closure, reflex, replans
    exs = sft.router_examples(load(d), [], features=features)
    assert len(exs) == len(r.seen) > 20
    assert [e["text"] for e in exs] == r.seen


def test_scripted_stand_in_students(tmp_path):
    """4.5 on fresh flights: a faithful copy of the teacher is labelled local/commit only (a fresh pilot per
    episode; one shared pilot carried its search memory across episodes and disagreed); a student whose
    search altitude is off by 20 m escalates every search, and by 3 m stays local."""
    pytest.importorskip("avians_agent")
    from avians_adp.student_calls import main as student_main
    for seed in (0, 1):
        _fly(tmp_path / "eps", seed)
    assert student_main([str(tmp_path / "eps"), "--pilot", "scripted"]) == 0
    r = build(tmp_path / "eps", tmp_path / "out", BuildConfig(ratios=(1.0, 0.0, 0.0)))
    labels = r["router"]["labels"]
    assert labels["escalate"] == 0 and labels["local"] > 0 and labels["commit"] > 0

    from avians_adp.student_calls import pilot_decide
    for shift, expect in ((20.0, "escalate"), (3.0, "local")):
        for ep in sorted((tmp_path / "eps").iterdir()):
            decide, _ = pilot_decide("scripted")

            def shifted(system, prompt, tools, st, decide=decide, shift=shift):
                d = decide(system, prompt, tools, st)
                args = dict(d.args)
                if d.tool == "search_region" and "alt" in args:
                    args["alt"] = args["alt"] + shift
                return {"tool": d.tool, "args": args}
            write_calls(ep, run_student(load(ep), shifted))
        out = tmp_path / f"out_{shift:g}"
        exs = build(tmp_path / "eps", out, BuildConfig(ratios=(1.0, 0.0, 0.0)))
        meta = [json.loads(x) for x in (out / "router" / "train.meta.jsonl").read_text().splitlines()]
        searches = [m for m in meta if m["teacher"]["tool"] == "search_region"]
        assert searches and all(m["label"] == expect for m in searches)
        assert exs["router"]["labels"]["commit"] == labels["commit"]
