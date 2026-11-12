"""Pilot SFT set (Phase 2): examples, the two layouts, and the build entry point."""
import json
import warnings
from pathlib import Path

import pytest
from avians_adp import EpisodeRecorder, Trajectory, decisions, load
from avians_adp.build import BuildConfig, build, main
from avians_adp.filters import DROPPABLE, DecisionFilter
from avians_adp.formats import sharegpt_dataset_info, to_sharegpt, to_trl
from avians_adp.sft import clean_reason, decision_kind, openai_tools, pilot_examples, target_arguments

FIXTURES = Path(__file__).parent / "fixtures"
TOOLS = [
    {"name": "takeoff", "description": "Take off", "input_schema": {
        "type": "object", "properties": {"alt": {"type": "number"}, "reason": {"type": "string"},
                                         "notes": {"type": "string"}}, "required": ["alt"]}},
    {"name": "search_region", "description": "Search", "input_schema": {
        "type": "object", "properties": {"region": {"type": "string"}, "reason": {"type": "string"},
                                         "notes": {"type": "string"}}, "required": ["region"]}},
    {"name": "keep", "description": "Keep going", "input_schema": {
        "type": "object", "properties": {"reason": {"type": "string"}, "notes": {"type": "string"}}}},
]
META = {"seed": 7, "pilot": "haiku"}
# Phase 2's layout tests: every decision, all in train, nothing dropped or thinned
EVERYTHING = BuildConfig(decisions=DecisionFilter(allow=DROPPABLE), ratios=(1.0, 0.0, 0.0), keep_every=1)


def record_episode(out: Path, ep_id: str = "sar_supply_haiku_s0007", seed: int = 7) -> Trajectory:
    """Four decisions, in the loop's order, covering what scripted episodes never show:
    1 text + thinking + notes; 2 a rejected call with the extra-calls suffix and a stop reason;
    3 an API-error keep; 4 unfinished (the episode ends before the result)."""
    rec = EpisodeRecorder(out, ep_id, {**META, "seed": seed})
    rec.brief("SYSTEM PROMPT", TOOLS)
    common = {"model": "claude-haiku-5-5", "by": "local", "error": ""}
    rec.observation("PROMPT 1", 0.0, decision=1)
    rec.message("Taking off first.", 0.0, thinking="Need altitude before searching.", decision=1)
    rec.action("takeoff", {"alt": 30}, "get airborne", 0.0, notes="plan: r2 then r1", t_obs=0.0, t_apply=2.0,
               usage={"input": 900, "output": 40, "stop": "tool_use"}, **common)
    rec.tool_result("takeoff: started", 2.0, accepted=True, decision=1)
    rec.observation("PROMPT 2", 45.0, decision=2)
    rec.action("search_region", {"region": "r9"}, "highest prior (ignored 1 more call(s))", 45.0, notes=None,
               t_obs=45.0, t_apply=47.0, usage={"stop": "tool_use"}, **common)
    rec.tool_result("rejected: no region r9", 47.0, accepted=False, decision=2)
    rec.observation("PROMPT 3", 90.0, decision=3)
    rec.action("keep", {}, "API error", 90.0, t_obs=90.0, t_apply=91.0,
               **{**common, "error": "529: overloaded"})
    rec.tool_result("kept: takeoff running", 91.0, accepted=True, decision=3)
    rec.observation("PROMPT 4", 135.0, decision=4)
    rec.action("keep", {}, "search still running", 135.0, notes="", t_obs=135.0, t_apply=136.0, **common)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rec.finish({"robustness": 0.4, "satisfied": True}, {"robustness": 0.4, "satisfied": True})
    return load(out)


@pytest.fixture
def episode(tmp_path):
    return record_episode(tmp_path / "ep")


def test_pilot_examples_one_per_decision(episode):
    exs = pilot_examples(episode)
    assert len(exs) == len([d for d in decisions(episode) if d.action is not None]) == 4
    assert [e["id"] for e in exs] == [f"sar_supply_haiku_s0007#d{i}" for i in (1, 2, 3, 4)]
    e1, e2, e3, e4 = exs
    assert e1["system"] == "SYSTEM PROMPT" and e1["prompt"] == "PROMPT 1"
    assert e1["call"] == {"name": "takeoff", "arguments": {"alt": 30, "reason": "get airborne",
                                                            "notes": "plan: r2 then r1"}}
    assert e1["text"] == "Taking off first." and e1["thinking"] == "Need altitude before searching."
    assert e1["meta"] == {"episode": "sar_supply_haiku_s0007", "seed": 7, "pilot": "haiku", "kind": "call",
                          "model": "claude-haiku-5-5", "t": 0.0, "decision": 1, "by": "local", "error": "",
                          "stop": "tool_use", "accepted": True, "robustness": 0.4, "satisfied": True,
                          "extra_calls": 0}
    # the suffix is stripped from the target and counted; notes=None is left out
    assert e2["call"]["arguments"] == {"region": "r9", "reason": "highest prior"}
    assert e2["meta"]["extra_calls"] == 1 and e2["meta"]["accepted"] is False and e2["text"] == ""
    # nothing is filtered here: the error keep is present, flagged for Phase 3
    assert e3["call"]["name"] == "keep" and e3["meta"]["error"] == "529: overloaded"
    assert e3["meta"]["kind"] == "api_error" and e4["meta"]["kind"] == "call"
    # unfinished last decision: an example, accepted unknown; an empty-string note is kept (it clears notes)
    assert e4["meta"]["accepted"] is None and e4["call"]["arguments"] == {"reason": "search still running",
                                                                           "notes": ""}


def test_tools_convert_to_openai_layout(episode):
    tools = pilot_examples(episode)[0]["tools"]
    assert [t["function"]["name"] for t in tools] == ["takeoff", "search_region", "keep"]
    assert tools[0] == {"type": "function", "function": {"name": "takeoff", "description": "Take off",
                                                         "parameters": TOOLS[0]["input_schema"]}}
    assert openai_tools(tools) == tools                     # already converted: unchanged


@pytest.mark.parametrize("raw, reason, n", [
    ("plain", "plain", 0),
    ("go now (ignored 2 more call(s))", "go now", 2),
    ("", "", 0), (None, "", 0),
    ("mentions (ignored 1 more call(s)) mid-sentence", "mentions (ignored 1 more call(s)) mid-sentence", 0),
])
def test_clean_reason(raw, reason, n):
    assert clean_reason(raw) == (reason, n)


def test_legacy_record_gives_examples():
    tj = Trajectory.model_validate_json((FIXTURES / "legacy_trajectory.json").read_text())
    exs = pilot_examples(tj)
    assert [e["call"]["name"] for e in exs] == ["takeoff", "search_region", "inspect"]
    assert all(e["call"]["arguments"]["reason"] for e in exs) and all(e["meta"]["seed"] == 0 for e in exs)


def test_trl_layout(episode):
    e1 = pilot_examples(episode)[0]
    row = to_trl(e1)
    assert set(row) == {"messages", "tools"}
    sys_m, user_m, asst = row["messages"]
    assert (sys_m["role"], user_m["role"], asst["role"]) == ("system", "user", "assistant")
    assert asst["content"] == "" and "reasoning_content" not in asst
    (call,) = asst["tool_calls"]
    assert call["type"] == "function" and call["function"]["name"] == "takeoff"
    assert isinstance(call["function"]["arguments"], str)
    assert json.loads(call["function"]["arguments"]) == e1["call"]["arguments"]
    full = to_trl(e1, include_text=True, include_thinking=True)["messages"][2]
    assert full["content"] == "Taking off first." and full["reasoning_content"] == "Need altitude before searching."
    e2 = pilot_examples(episode)[1]                       # no thinking: no empty reasoning_content key
    assert "reasoning_content" not in to_trl(e2, include_thinking=True)["messages"][2]


def test_sharegpt_layout(episode):
    for e in pilot_examples(episode):
        row = to_sharegpt(e)
        assert set(row) == {"conversations", "system", "tools"}
        roles = [m["from"] for m in row["conversations"]]
        assert roles == ["human", "function_call"]          # human odd, function_call even
        call = json.loads(row["conversations"][1]["value"])
        assert call == {"name": e["call"]["name"], "arguments": e["call"]["arguments"]}
        tools = json.loads(row["tools"])
        assert [t["name"] for t in tools] == ["takeoff", "search_region", "keep"] and "parameters" in tools[0]
    info = sharegpt_dataset_info({"avians_pilot_all": "all.jsonl"})["avians_pilot_all"]
    assert info["formatting"] == "sharegpt" and info["file_name"] == "all.jsonl"
    assert info["columns"] == {"messages": "conversations", "system": "system", "tools": "tools"}


def test_build_writes_both_layouts(tmp_path):
    for seed in (1, 2):
        record_episode(tmp_path / "eps" / f"s{seed:04d}", ep_id=f"ep_s{seed}", seed=seed)
    (tmp_path / "eps" / "broken").mkdir()
    (tmp_path / "eps" / "broken" / "trajectory.json").write_text("{not json")
    r = build(tmp_path / "eps", tmp_path / "out", EVERYTHING)
    assert r["episodes"]["train"] == ["ep_s1", "ep_s2"] and r["examples"]["train"] == 8 and list(r["skipped"]) == [
        str(tmp_path / "eps" / "broken")]
    out = tmp_path / "out"
    trl = [json.loads(x) for x in (out / "pilot" / "train.jsonl").read_text().splitlines()]
    meta = [json.loads(x) for x in (out / "pilot" / "train.meta.jsonl").read_text().splitlines()]
    lf = [json.loads(x) for x in (out / "pilot_lf" / "train.jsonl").read_text().splitlines()]
    assert len(trl) == len(meta) == len(lf) == 8
    assert meta[0]["id"] == "ep_s1#d1" and meta[4]["id"] == "ep_s2#d1"      # same row order everywhere
    for t, l in zip(trl, lf):
        a = json.loads(t["messages"][2]["tool_calls"][0]["function"]["arguments"])
        assert a == json.loads(l["conversations"][1]["value"])["arguments"]
    info = json.loads((out / "pilot_lf" / "dataset_info.json").read_text())
    assert info["avians_pilot_train"]["file_name"] == "train.jsonl"
    assert not any("meta" in row for row in trl + lf)
    with pytest.raises(FileExistsError):
        build(tmp_path / "eps", out, EVERYTHING)
    assert build(tmp_path / "eps", out, EVERYTHING, force=True)["examples"]["train"] == 8


def test_build_rejects_duplicate_episode_ids(tmp_path):
    record_episode(tmp_path / "eps" / "a", ep_id="same")
    record_episode(tmp_path / "eps" / "b", ep_id="same")
    with pytest.raises(ValueError, match="duplicate"):
        build(tmp_path / "eps", tmp_path / "out")


def test_build_cli(tmp_path, capsys):
    record_episode(tmp_path / "eps" / "s0007")
    assert main([str(tmp_path / "eps"), str(tmp_path / "out"), "--thinking", "--split", "100/0/0"]) == 0
    assert json.loads(capsys.readouterr().out)["examples"] == {"train": 1, "val": 0, "test": 0}
    row = json.loads((tmp_path / "out" / "pilot" / "train.jsonl").read_text().splitlines()[0])
    assert row["messages"][2]["reasoning_content"] == "Need altitude before searching."
    assert main([str(tmp_path / "eps"), str(tmp_path / "out")]) == 1          # not empty, no --force
    assert main([str(tmp_path / "empty"), str(tmp_path / "out2")]) == 1        # no episodes


def test_real_scripted_episode_round_trips(tmp_path):
    """Gate 2.6a on a fresh scripted flight: every target parses back to the recorded call."""
    pytest.importorskip("avians_agent")
    from avians_agent.loop import run_episode
    from avians_agent.mission import Mission, load_config
    from avians_agent.pilots import ScriptedPilot
    root = Path(__file__).resolve().parents[3]
    m = Mission.load(root / "missions" / "sar_supply.yaml")
    run_episode(m, ScriptedPilot(), 0, tmp_path / "ep", load_config("agent.yaml"), load_config("latency.yaml"))
    tj = load(tmp_path / "ep")
    _assert_round_trip(tj)


def _assert_round_trip(tj):
    decs = [d for d in decisions(tj) if d.action is not None]
    exs = pilot_examples(tj)
    assert len(exs) == len(decs) > 0
    for d, e in zip(decs, exs):
        a = d.action
        trl_args = json.loads(to_trl(e)["messages"][2]["tool_calls"][0]["function"]["arguments"])
        lf_call = json.loads(to_sharegpt(e)["conversations"][1]["value"])
        for args, name in ((trl_args, e["call"]["name"]), (lf_call["arguments"], lf_call["name"])):
            assert name == a.function
            rest = {k: v for k, v in args.items() if k not in ("reason", "notes")}
            assert rest == a.kwargs
            assert args.get("reason", "") == clean_reason(a.description)[0]
            assert args.get("notes") == a.details.get("notes")


@pytest.mark.parametrize("function, reason, details, kind", [
    ("search_region", "go", {"model": "claude-haiku-5-5", "by": "local", "error": ""}, "call"),
    ("keep", "nothing to do", {"model": "claude-haiku-5-5", "by": "local", "error": ""}, "call"),
    ("keep", "API error", {"model": "claude-haiku-5-5", "error": "529: overloaded"}, "api_error"),
    ("keep", "API unreachable", {"model": "claude-haiku-5-5", "error": "connection reset"}, "api_error"),
    ("keep", "no tool call", {"model": "claude-haiku-5-5", "by": "local", "error": ""}, "no_tool_call"),
    ("keep", "router: commit", {"model": "router", "by": "commit", "error": ""}, "router_commit"),
])
def test_decision_kind(function, reason, details, kind):
    assert decision_kind(function, reason, details) == kind


def test_missing_reason_is_not_invented():
    assert target_arguments({"region": "r1"}, "", None) == {"region": "r1"}
    assert target_arguments({}, "why", "") == {"reason": "why", "notes": ""}


def test_force_never_deletes_episodes_or_unrelated_files(tmp_path):
    out = tmp_path / "work"
    record_episode(out / "eps" / "s0007")                  # episodes inside the output folder
    with pytest.raises(ValueError, match="contains the episodes"):
        build(out / "eps", out, force=True)
    with pytest.raises(ValueError, match="contains the episodes"):
        build(out / "eps", out / "eps", force=True)
    assert (out / "eps" / "s0007" / "trajectory.json").exists()
    eps = tmp_path / "eps"
    record_episode(eps / "s0007")
    out2 = tmp_path / "out2"
    out2.mkdir()
    (out2 / "notes.txt").write_text("keep me")
    with pytest.raises(FileExistsError):
        build(eps, out2)
    build(eps, out2, EVERYTHING, force=True)
    assert (out2 / "notes.txt").read_text() == "keep me" and (out2 / "pilot" / "train.jsonl").exists()
    (tmp_path / "afile").write_text("x")
    with pytest.raises(FileExistsError, match="not a folder"):
        build(eps, tmp_path / "afile")


def test_build_skips_unreadable_trajectory(tmp_path):
    record_episode(tmp_path / "eps" / "good")
    (tmp_path / "eps" / "odd" / "trajectory.json").mkdir(parents=True)      # a folder, not a file
    r = build(tmp_path / "eps", tmp_path / "out", EVERYTHING)
    assert r["examples"]["train"] == 4 and any("odd" in k for k in r["skipped"])


def test_files_load_unchanged_with_datasets(tmp_path):
    """The training loaders read the files exactly as written (no Arrow null padding)."""
    datasets = pytest.importorskip("datasets")
    for seed in (1, 2):
        record_episode(tmp_path / "eps" / f"s{seed}", ep_id=f"ep{seed}", seed=seed)
    build(tmp_path / "eps", tmp_path / "out", EVERYTHING)
    for sub in ("pilot", "pilot_lf"):
        f = tmp_path / "out" / sub / "train.jsonl"
        raw = [json.loads(x) for x in f.read_text().splitlines()]
        ds = datasets.load_dataset("json", data_files=str(f), split="train")
        assert [ds[i] for i in range(len(ds))] == raw
