"""Phase 3: episode filters, decision filters, splits by seed, keep-thinning, and the build wiring."""
import collections
import json
import warnings
from pathlib import Path

import pytest
from avians_adp import EpisodeRecorder, Trajectory, load
from avians_adp.build import BuildConfig, build, main
from avians_adp.filters import DROPPABLE, DecisionFilter, EpisodeFilter, keep_signature, thin_keeps
from avians_adp.splits import check_seed, parse_ratios, split_of, unit

TOOLS = [{"name": n, "description": n, "input_schema": {"type": "object", "properties": {}}}
         for n in ("takeoff", "search_region", "keep")]
CLAUSES = [{"clause": "battery", "robustness": 12.0, "satisfied": True},
           {"clause": "clearance", "robustness": -0.2, "satisfied": False},
           {"clause": "deliver", "robustness": 0.4, "satisfied": True}]


def state(skill="search_region", status="running", battery=74.0, found=False, closed=()):
    return {"skill": skill, "skill_status": status, "battery": battery, "payload": False, "released": False,
            "help_arrived": False, "belief": {"found": found},
            "regions": [{"id": r, "closed": r in closed} for r in ("r1", "r2", "r3")]}


def record(out: Path, ep_id: str, seed: int, calls, *, pilot="haiku", tier="fast", satisfied=True,
           robustness=0.5, clauses=CLAUSES) -> None:
    """calls: (function, state or None, extra details) per decision; every call is answered and accepted
    unless the extra details say accepted=False."""
    rec = EpisodeRecorder(out, ep_id, {"seed": seed, "pilot": pilot, "tier": tier})
    rec.brief("SYSTEM", TOOLS)
    for n, (fn, st, extra) in enumerate(calls, start=1):
        t = 45.0 * (n - 1)
        extra = dict(extra)
        accepted = extra.pop("accepted", True)
        rec.observation(f"PROMPT {n}", t, decision=n)
        rec.action(fn, {}, extra.pop("reason", f"reason {n}"), t, t_obs=t, t_apply=t + 1,
                   **{"model": "claude-haiku-5-5", "by": "local", "error": "", "state": st, **extra})
        rec.tool_result("ok" if accepted else "rejected", t + 1, accepted=accepted, decision=n)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rec.finish({"robustness": robustness, "satisfied": satisfied, "clauses": clauses},
                   {"robustness": robustness, "satisfied": satisfied})


def traj(**kw) -> Trajectory:
    meta = {"seed": 1, "pilot": kw.pop("pilot", "haiku"), "tier": kw.pop("tier", "fast")}
    sat, rob = kw.pop("satisfied", True), kw.pop("robustness", 0.5)
    return Trajectory(id="e", details={"meta": meta, "summary": {"satisfied": sat, "robustness": rob},
                                       "score": {"satisfied": sat, "robustness": rob,
                                                 "clauses": kw.pop("clauses", CLAUSES)}})


def ex(name="keep", st=None, episode="e1", kind="call", accepted=True, model="claude-haiku-5-5", n=0):
    return {"id": f"{episode}#d{n}", "call": {"name": name, "arguments": {}}, "state": st,
            "meta": {"episode": episode, "kind": kind, "accepted": accepted, "model": model, "decision": n}}


# ------------------------------------------------------------------ episode filters

@pytest.mark.parametrize("flt, t, why", [
    (EpisodeFilter(), traj(satisfied=False, robustness=-10), None),
    (EpisodeFilter(satisfied=True), traj(satisfied=False), "not satisfied"),
    (EpisodeFilter(satisfied=True), traj(satisfied=True), None),
    (EpisodeFilter(min_robustness=0.5), traj(robustness=0.49), "robustness"),
    (EpisodeFilter(min_robustness=0.5), traj(robustness=0.5), None),
    (EpisodeFilter(require_clauses=("battery", "deliver")), traj(), None),
    (EpisodeFilter(require_clauses=("clearance",)), traj(), "clause clearance"),
    (EpisodeFilter(require_clauses=("deliver",), clause_margin=1.0), traj(), "clause deliver"),
    (EpisodeFilter(require_clauses=("battery",), clause_margin=10.0), traj(), None),
    (EpisodeFilter(pilots=("haiku",)), traj(pilot="scripted"), "pilot"),
    (EpisodeFilter(pilots=("haiku", "scripted")), traj(pilot="scripted"), None),
    (EpisodeFilter(tiers=("gazebo",)), traj(tier="fast"), "tier"),
])
def test_episode_filter(flt, t, why):
    assert flt.reject_reason(t) == why


def test_unknown_clause_is_an_error_not_a_filter():
    with pytest.raises(ValueError, match="unknown clause"):
        EpisodeFilter(require_clauses=("batery",)).check_clause_names([traj()])
    EpisodeFilter(require_clauses=("battery",)).check_clause_names([traj()])


# ------------------------------------------------------------------ decision filters

@pytest.mark.parametrize("e, why", [
    (ex(), None),
    (ex("search_region"), None),
    (ex(kind="api_error"), "api_error"),
    (ex(kind="no_tool_call"), "no_tool_call"),
    (ex(kind="router_commit"), "router_commit"),
    (ex("search_region", accepted=False), "rejected"),
    (ex(accepted=None), "unanswered"),
])
def test_decision_filter_drops_exactly_its_case(e, why):
    assert DecisionFilter().reject_reason(e) == why


def test_decision_filter_allow_and_models():
    everything = DecisionFilter(allow=DROPPABLE)
    for kind in ("api_error", "no_tool_call", "router_commit"):
        assert everything.reject_reason(ex(kind=kind)) is None
    assert everything.reject_reason(ex(accepted=False)) is None and everything.reject_reason(ex(accepted=None)) is None
    assert DecisionFilter(allow=("rejected",)).reject_reason(ex(accepted=False)) is None
    assert DecisionFilter(allow=("rejected",)).reject_reason(ex(kind="api_error")) == "api_error"
    only = DecisionFilter(models=("claude-haiku-5-5",))
    assert only.reject_reason(ex(model="scripted")) == "model" and only.reject_reason(ex()) is None
    with pytest.raises(ValueError, match="unknown allow"):
        DecisionFilter(allow=("errors",))


# ------------------------------------------------------------------ splits

def test_split_is_a_pure_function_of_the_seed():
    # pinned: a change to the hash would silently reshuffle every dataset
    assert [split_of(s) for s in range(8)] == ["train", "train", "train", "train", "test", "train", "train", "val"]
    assert all(0.0 <= unit(s) < 1.0 for s in range(1000))
    assert split_of(7) == split_of(7)


def test_ratios_hold_over_many_seeds():
    c = collections.Counter(split_of(s) for s in range(1000))
    assert 760 <= c["train"] <= 840 and 70 <= c["val"] <= 130 and 70 <= c["test"] <= 130
    c = collections.Counter(split_of(s, (0.5, 0.5, 0.0)) for s in range(1000))
    assert c["test"] == 0 and 450 <= c["train"] <= 550
    assert {split_of(s, (1.0, 0.0, 0.0)) for s in range(1000)} == {"train"}


def test_parse_ratios():
    assert parse_ratios("80/10/10") == pytest.approx((0.8, 0.1, 0.1))
    assert parse_ratios("0.7,0.2,0.1") == pytest.approx((0.7, 0.2, 0.1))
    assert parse_ratios("1/1/0") == pytest.approx((0.5, 0.5, 0.0))
    for bad in ("80/20", "80/-10/30", "0/0/0", "a/b/c"):
        with pytest.raises(ValueError):
            parse_ratios(bad)


def test_split_without_seed_is_an_error():
    with pytest.raises(ValueError):
        split_of(None)


# ------------------------------------------------------------------ thinning

def test_signature_ignores_what_does_not_matter():
    assert keep_signature(state(battery=74.0)) == keep_signature(state(battery=50.1))     # same 25 % band
    assert keep_signature(state(battery=74.0)) != keep_signature(state(battery=49.9))
    assert keep_signature(state(battery=74.0), 10) != keep_signature(state(battery=69.9), 10)
    assert keep_signature(state(battery=90.0), 0) == keep_signature(state(battery=5.0), 0)   # 0: battery ignored
    assert keep_signature({"skill": "x", "regions": [{"closed": True}]}) is not None         # region without id
    assert keep_signature(state()) != keep_signature(state(status="done"))
    assert keep_signature(state()) != keep_signature(state(found=True))
    assert keep_signature(state()) != keep_signature(state(closed=("r2",)))
    assert keep_signature(None) is None and keep_signature({}) is None


def test_thin_keeps_one_in_k_per_run():
    s = state()
    seq = [ex("search_region", s, n=0)] + [ex("keep", s, n=i) for i in range(1, 8)]
    kept, removed = thin_keeps(seq, 3)
    assert [e["meta"]["decision"] for e in kept] == [0, 1, 4, 7] and removed == 4
    assert thin_keeps(seq, 1) == (seq, 0)
    with pytest.raises(ValueError):
        thin_keeps(seq, 0)


def test_thin_keeps_runs_break_on_calls_signatures_and_episodes():
    a, b = state(), state(status="done")
    seq = [ex("keep", a, n=1), ex("keep", a, n=2),            # run 1: keeps 1
           ex("search_region", a, n=3),                         # a call: always kept, ends the run
           ex("keep", a, n=4), ex("keep", a, n=5),            # run 2: keeps 4
           ex("keep", b, n=6), ex("keep", b, n=7),            # new signature, run 3: keeps 6
           ex("keep", b, episode="e2", n=8),                   # another episode, run 4: keeps 8
           ex("keep", None, n=9), ex("keep", None, n=10)]      # no state: never thinned
    kept, removed = thin_keeps(seq, 3)
    assert [e["meta"]["decision"] for e in kept] == [1, 3, 4, 6, 8, 9, 10] and removed == 3
    assert all(e in kept for e in seq if e["call"]["name"] != "keep")


# ------------------------------------------------------------------ the build

def episode_calls(st):
    """takeoff, 5 identical keeps, a rejected call, an API-error keep."""
    return ([("takeoff", st, {})] + [("keep", st, {}) for _ in range(5)]
            + [("search_region", st, {"accepted": False}),
               ("keep", st, {"error": "529: overloaded", "reason": "API error"})])


def test_build_filters_splits_and_thins_train_only(tmp_path):
    eps = tmp_path / "eps"
    st = state()
    record(eps / "train_a", "a_s0", 0, episode_calls(st))                  # seed 0 -> train
    record(eps / "train_b", "b_s0", 0, episode_calls(st), pilot="sonnet")  # same seed, other pilot -> train
    record(eps / "val", "c_s7", 7, episode_calls(st))                      # seed 7 -> val
    record(eps / "test", "d_s4", 4, episode_calls(st), satisfied=False)    # seed 4 -> test, but unsatisfied
    r = build(eps, tmp_path / "out", BuildConfig(), force=False)
    assert r["episodes"] == {"train": ["a_s0", "b_s0"], "val": ["c_s7"], "test": ["d_s4"]}
    # per episode: 8 decisions; 1 rejected + 1 api_error dropped -> 6; train thins 5 keeps (k=3) to 2 -> 3
    assert r["examples"] == {"train": 6, "val": 6, "test": 6}
    assert r["removed"]["decisions"] == {"rejected": 4, "api_error": 4} and r["removed"]["thinned"] == 6
    meta = [json.loads(x) for x in (tmp_path / "out" / "pilot" / "train.meta.jsonl").read_text().splitlines()]
    assert [m["decision"] for m in meta] == [1, 2, 5, 1, 2, 5]
    info = json.loads((tmp_path / "out" / "pilot_lf" / "dataset_info.json").read_text())
    assert sorted(info) == ["avians_pilot_test", "avians_pilot_train", "avians_pilot_val"]
    for split, n in r["examples"].items():
        for sub, name in (("pilot", f"{split}.jsonl"), ("pilot", f"{split}.meta.jsonl"), ("pilot_lf", f"{split}.jsonl")):
            assert len((tmp_path / "out" / sub / name).read_text().splitlines()) == n

    r = build(eps, tmp_path / "out", BuildConfig(episodes=EpisodeFilter(satisfied=True)), force=True)
    assert r["episodes"]["test"] == [] and r["removed"]["episodes"] == {"not satisfied": 1}
    assert r["warnings"] == ["split 'test' is empty"] and r["examples"]["test"] == 0
    assert (tmp_path / "out" / "pilot" / "test.jsonl").read_text() == ""


def test_no_seed_in_two_splits(tmp_path):
    eps = tmp_path / "eps"
    for seed in range(12):
        for pilot in ("haiku", "scripted"):
            record(eps / f"{pilot}_s{seed}", f"{pilot}_s{seed}", seed, [("takeoff", state(), {})], pilot=pilot)
    r = build(eps, tmp_path / "out")
    seen = {}
    for split, ids in r["episodes"].items():
        for i in ids:
            seed = int(i.rsplit("_s", 1)[1])
            assert seen.setdefault(seed, split) == split
    assert sorted(seen) == list(range(12))
    assert sum(len(v) for v in r["episodes"].values()) == 24


def test_build_rejects_bad_config(tmp_path):
    record(tmp_path / "eps" / "a", "a", 0, [("takeoff", state(), {})])
    with pytest.raises(ValueError, match="unknown clause"):
        build(tmp_path / "eps", tmp_path / "out", BuildConfig(episodes=EpisodeFilter(require_clauses=("nope",))))
    with pytest.raises(ValueError, match="keep_every"):
        build(tmp_path / "eps", tmp_path / "out2", BuildConfig(keep_every=0))


def test_cli_flags(tmp_path, capsys):
    eps = tmp_path / "eps"
    record(eps / "a", "a_s0", 0, episode_calls(state()))
    record(eps / "b", "b_s1", 1, episode_calls(state()), robustness=-1.0, satisfied=False)
    rc = main([str(eps), str(tmp_path / "out"), "--satisfied", "--require-clauses", "battery,deliver",
               "--allow", "rejected", "--keep-every", "1", "--model", "claude-haiku-5-5", "--split", "1/0/0"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["episodes"] == {"train": 1, "val": 0, "test": 0}
    assert out["examples"]["train"] == 7 and out["removed"]["decisions"] == {"api_error": 1}
    assert main([str(eps), str(tmp_path / "o2"), "--allow", "bogus"]) == 1
    assert main([str(eps), str(tmp_path / "o3"), "--split", "80/20"]) == 1
    assert main([str(eps), str(tmp_path / "o4"), "--min-robustness", "5"]) == 1      # everything filtered


def test_real_scripted_episodes(tmp_path):
    """Gate 3.6 on fresh scripted flights: --satisfied keeps exactly the satisfied ones."""
    pytest.importorskip("avians_agent")
    from avians_agent.loop import run_episode
    from avians_agent.mission import Mission, load_config
    from avians_agent.pilots import ScriptedPilot
    root = Path(__file__).resolve().parents[3]
    m = Mission.load(root / "missions" / "sar_supply.yaml")
    sat = {}
    for seed in (0, 1):                      # seed 0 satisfies the spec; seed 1 fails clearance
        s = run_episode(m, ScriptedPilot(), seed, tmp_path / "eps" / f"s{seed}", load_config("agent.yaml"),
                        load_config("latency.yaml"))
        sat[load(tmp_path / "eps" / f"s{seed}").id] = s["satisfied"]
    r = build(tmp_path / "eps", tmp_path / "out", BuildConfig(episodes=EpisodeFilter(satisfied=True),
                                                              ratios=(1.0, 0.0, 0.0)))
    assert r["episodes"]["train"] == sorted(i for i, ok in sat.items() if ok)
    assert r["removed"]["thinned"] > 0          # scripted flights have long keep runs


# ------------------------------------------------------------------ review fixes (Fable review of Phase 3)

def test_infinite_robustness_written_as_strings():
    """The scorer writes infinities as "inf"/"-inf" (avians_spec.stl.fmt)."""
    inf = [{"clause": "battery", "robustness": "inf", "satisfied": True},
           {"clause": "deliver", "robustness": "-inf", "satisfied": False}]
    t = traj(clauses=inf, robustness="inf")
    assert EpisodeFilter(min_robustness=0.1).reject_reason(t) is None
    assert EpisodeFilter(require_clauses=("battery",), clause_margin=5.0).reject_reason(t) is None
    assert EpisodeFilter(require_clauses=("deliver",)).reject_reason(t) == "clause deliver"
    assert EpisodeFilter(min_robustness=0.1).reject_reason(traj(robustness="-inf")) == "robustness"
    assert EpisodeFilter(min_robustness=0.1).reject_reason(traj(robustness="n/a")) == "robustness"


def test_a_bad_option_never_costs_the_previous_build(tmp_path):
    eps = tmp_path / "eps"
    record(eps / "a", "a_s0", 0, [("takeoff", state(), {})])
    out = tmp_path / "out"
    build(eps, out)
    before = (out / "pilot" / "train.jsonl").read_text()
    with pytest.raises(ValueError, match="unknown clause"):
        build(eps, out, BuildConfig(episodes=EpisodeFilter(require_clauses=("batery",))), force=True)
    record(eps / "b", "a_s0", 1, [("takeoff", state(), {})])       # a duplicate id
    with pytest.raises(ValueError, match="duplicate"):
        build(eps, out, force=True)
    assert (out / "pilot" / "train.jsonl").read_text() == before and before


@pytest.mark.parametrize("text", ["nan/1/1", "inf/1/1", "1/1/nan"])
def test_ratios_must_be_finite(text):
    with pytest.raises(ValueError):
        parse_ratios(text)


@pytest.mark.parametrize("seed", [None, True, 1.9, 1.0, "07", "x"])
def test_seeds_must_be_integers(seed):
    with pytest.raises(ValueError):
        check_seed(seed)
    with pytest.raises(ValueError):
        split_of(seed)


def test_episode_with_a_non_integer_seed_is_counted_not_split(tmp_path):
    record(tmp_path / "eps" / "a", "a", 0, [("takeoff", state(), {})])
    record(tmp_path / "eps" / "b", "b", "07", [("takeoff", state(), {})])
    r = build(tmp_path / "eps", tmp_path / "out")
    assert r["episodes"]["train"] == ["a"] and r["removed"]["episodes"] == {"no integer seed": 1}


def test_no_empty_split_warning_for_a_zeroed_split(tmp_path):
    record(tmp_path / "eps" / "a", "a", 0, [("takeoff", state(), {})])
    assert build(tmp_path / "eps", tmp_path / "out", BuildConfig(ratios=(1.0, 0.0, 0.0)))["warnings"] == []
    r = build(tmp_path / "eps", tmp_path / "out2")
    assert r["warnings"] == ["split 'val' is empty", "split 'test' is empty"]


def test_keep_every_matters_on_long_keep_runs():
    """A draining battery must not cut a loiter into tiny runs (review item 6)."""
    seq = [ex("keep", state(battery=95.0 - 2 * i), n=i) for i in range(12)]   # 11 keeps in 75-100 %, then 73 %
    removed = {k: thin_keeps(seq, k)[1] for k in (2, 3, 6)}
    assert removed == {2: 5, 3: 7, 6: 9}          # the 11-keep run keeps 6, 4, 2; the 12th starts a run
