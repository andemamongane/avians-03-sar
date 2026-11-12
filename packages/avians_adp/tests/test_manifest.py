"""Phase 5: the manifest, safe writes, reproducible builds."""
import hashlib
import json
import re
import warnings
from pathlib import Path

import pytest
from avians_adp import EpisodeRecorder, load
from avians_adp import build as build_mod
from avians_adp.build import BuildConfig, build, main
from avians_adp.manifest import MANIFEST, git_info
from avians_adp.sft import pilot_examples
from avians_adp.splits import SPLITS
from avians_adp.student_calls import run_student, write_calls

TOOLS = [{"name": n, "description": n, "input_schema": {"type": "object", "properties": {}}}
         for n in ("takeoff", "search_region", "keep")]


def record(out: Path, ep_id: str, seed: int, n: int = 3) -> None:
    rec = EpisodeRecorder(out, ep_id, {"seed": seed, "pilot": "haiku"})
    rec.brief("SYSTEM", TOOLS)
    for i in range(1, n + 1):
        fn = "takeoff" if i == 1 else "keep"
        rec.observation(f"PROMPT {i}", 45.0 * i, decision=i)
        rec.action(fn, {}, "why", 45.0 * i, t_obs=45.0 * i, t_apply=45.0 * i + 1, model="claude-haiku-5-5",
                   by="local", error="", state=None)
        rec.tool_result("ok", 45.0 * i + 1, accepted=True, decision=i)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rec.finish({"robustness": 0.5, "satisfied": True}, {"robustness": 0.5, "satisfied": True})


@pytest.fixture(scope="module")
def flown(tmp_path_factory):
    """Real scripted flights (seeds 0, 4, 5: train, test, train; 5 fails the spec), with student calls on one."""
    pytest.importorskip("avians_agent")
    from avians_adp.student_calls import pilot_decide
    from avians_agent.loop import run_episode
    from avians_agent.mission import Mission, load_config
    from avians_agent.pilots import ScriptedPilot
    root = Path(__file__).resolve().parents[3]
    m = Mission.load(root / "missions" / "sar_supply.yaml")
    eps = tmp_path_factory.mktemp("flown") / "eps"
    for seed in (0, 4, 5):
        run_episode(m, ScriptedPilot(), seed, eps / f"s{seed:04d}", load_config("agent.yaml"),
                    load_config("latency.yaml"))
    decide, model = pilot_decide("scripted")
    write_calls(eps / "s0000", run_student(load(eps / "s0000"), decide, model=model))
    return eps


def read_rows(p: Path) -> int:
    return len(p.read_text().splitlines())


def test_manifest_describes_the_files_exactly(flown, tmp_path):
    out = tmp_path / "out"
    r = build(flown, out)
    m = json.loads((out / MANIFEST).read_text())
    assert r["manifest"] == str(out / MANIFEST) and r["data_digest"] == m["data_digest"]
    on_disk = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() and p.name != MANIFEST)
    assert sorted(m["files"]) == on_disk
    for rel, entry in m["files"].items():
        data = (out / rel).read_bytes()
        assert entry["sha256"] == hashlib.sha256(data).hexdigest() and entry["bytes"] == len(data)
        if rel.endswith(".jsonl"):
            assert entry["rows"] == read_rows(out / rel)
    for s in SPLITS:
        n = m["pilot"]["examples"][s]
        assert n == m["files"][f"pilot/{s}.jsonl"]["rows"] == m["files"][f"pilot_lf/{s}.jsonl"]["rows"]
        assert sum(m["pilot"]["by_tool"][s].values()) == n == sum(m["pilot"]["by_pilot"][s].values())
        assert m["router"]["examples"][s] == m["files"][f"router/{s}.jsonl"]["rows"]
    assert m["pilot"]["episodes"] == r["episodes"]


def test_manifest_accounts_for_every_decision(flown, tmp_path):
    out = tmp_path / "out"
    r = build(flown, out)
    m = json.loads((out / MANIFEST).read_text())
    kept_eps = {i for s in SPLITS for i in m["pilot"]["episodes"][s]}
    total = sum(len(pilot_examples(load(d))) for d in sorted(flown.iterdir()) if load(d).id in kept_eps)
    removed = m["pilot"]["removed"]
    assert total == sum(m["pilot"]["examples"].values()) + sum(removed["decisions"].values()) + removed["thinned"]
    assert removed["thinned"] > 0 and r["removed"] == removed


def test_manifest_records_source_settings_and_code(flown, tmp_path):
    out = tmp_path / "out"
    cfg = BuildConfig(keep_every=4)
    build(flown, out, cfg, argv=["python", "-m", "avians_adp.build", "x", "y", "--keep-every", "4"])
    m = json.loads((out / MANIFEST).read_text())
    assert m["format"] == "avians_adp.manifest/1" and m["adp_schema"]["version"] == "1.3.1"
    assert m["source"] == {"episodes_dir": str(flown.resolve()), "found": 3, "loaded": 3, "skipped": {}}
    c = m["config"]
    assert c["keep_every"] == 4 and c["ratios"] == [0.8, 0.1, 0.1] and c["battery_band"] == 25.0
    assert c["episodes"]["satisfied"] is False and c["decisions"]["allow"] == []
    assert c["tolerances"]["east"] == 10.0
    assert m["encoding"] == {"arguments": "json_string", "tools_trl": "list", "tools_sharegpt": "json_string",
                             "chat_template": "qwen3"}
    assert m["command"][-2:] == ["--keep-every", "4"]
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00", m["created"])
    g = m["git"]
    assert g is not None and re.fullmatch(r"[0-9a-f]{40}", g["commit"]) and isinstance(g["dirty"], bool)
    assert m["router"]["labels"]["train"].get("commit", 0) > 0 and m["router"]["events_inexact"] == 0


def test_git_info_outside_a_checkout(tmp_path):
    assert git_info(tmp_path) is None


def test_same_inputs_same_data(flown, tmp_path):
    a = build(flown, tmp_path / "a")
    b = build(flown, tmp_path / "b")
    assert a["data_digest"] == b["data_digest"]
    ma, mb = (json.loads((tmp_path / x / MANIFEST).read_text()) for x in ("a", "b"))
    assert ma["files"] == mb["files"]
    c = build(flown, tmp_path / "c", BuildConfig(keep_every=1))
    assert c["data_digest"] != a["data_digest"]


def test_skipped_and_filtered_episodes_are_recorded(tmp_path):
    eps = tmp_path / "eps"
    record(eps / "a", "a_s0", 0)
    record(eps / "b", "b_s3", 3)
    (eps / "bad").mkdir()
    (eps / "bad" / "trajectory.json").write_text("{}")
    out = tmp_path / "out"
    build(eps, out, BuildConfig(episodes=build_mod.EpisodeFilter(min_robustness=0.9)))
    m = json.loads((out / MANIFEST).read_text())
    assert m["source"]["found"] == 3 and m["source"]["loaded"] == 2
    assert list(m["source"]["skipped"]) == [str(eps / "bad")]
    assert m["pilot"]["removed"]["episodes"] == {"robustness": 2} and m["router"] is None
    assert "split 'train' is empty" in m["warnings"]


def test_a_failed_build_leaves_the_previous_one_intact(tmp_path, monkeypatch):
    eps = tmp_path / "eps"
    record(eps / "a", "a_s0", 0)
    out = tmp_path / "out"
    first = build(eps, out)
    before = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(build_mod, "make_manifest", boom)
    with pytest.raises(OSError, match="disk full"):
        build(eps, out, force=True)
    after = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
    assert after == before and first["data_digest"]
    assert not [p for p in tmp_path.iterdir() if ".building-" in p.name]       # no staging folder left


def test_force_replaces_the_manifest_and_keeps_unrelated_files(tmp_path):
    eps = tmp_path / "eps"
    record(eps / "a", "a_s0", 0)
    out = tmp_path / "out"
    build(eps, out)
    (out / "NOTES.md").write_text("mine")
    record(eps / "b", "b_s1", 1)
    r = build(eps, out, force=True)
    m = json.loads((out / MANIFEST).read_text())
    assert m["source"]["loaded"] == 2 and m["data_digest"] == r["data_digest"]
    assert (out / "NOTES.md").read_text() == "mine" and "NOTES.md" not in m["files"]


def test_default_output_folder_is_the_repos_data_folder():
    out = build_mod.default_out()
    repo = Path(__file__).resolve().parents[3]
    assert out.parent == repo / "data" / "datasets" and re.fullmatch(r"build-\d{8}-\d{6}", out.name)


def test_cli_without_an_output_folder(tmp_path, monkeypatch, capsys):
    record(tmp_path / "eps" / "a", "a_s0", 0)
    target = tmp_path / "data" / "datasets" / "build-x"
    monkeypatch.setattr(build_mod, "default_out", lambda: target)
    monkeypatch.chdir(tmp_path)
    assert main(["eps"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert Path(out["out"]) == target
    m = json.loads((target / MANIFEST).read_text())
    assert m["command"] == ["python", "-m", "avians_adp.build", "eps"] and out["data_digest"] == m["data_digest"]


# ------------------------------------------------------------------ review fixes (Fable review of Phase 5)

def snapshot(out: Path) -> dict:
    return {str(p.relative_to(out)): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}


@pytest.mark.parametrize("fail_at", [1, 2, 3, 5])
def test_a_failure_during_the_swap_restores_the_previous_build(tmp_path, monkeypatch, fail_at):
    """Fail the n-th rename of the swap (moving the old build aside, or the new one in)."""
    eps = tmp_path / "eps"
    record(eps / "a", "a_s0", 0)
    out = tmp_path / "out"
    build(eps, out)
    before = snapshot(out)
    record(eps / "b", "b_s1", 1)                           # the new build differs
    real, calls = build_mod.os.replace, []

    def flaky(src, dst):
        calls.append(1)
        if len(calls) == fail_at:
            raise OSError("disk full")
        return real(src, dst)
    monkeypatch.setattr(build_mod.os, "replace", flaky)
    with pytest.raises(OSError, match="disk full"):
        build(eps, out, force=True)
    monkeypatch.setattr(build_mod.os, "replace", real)
    assert snapshot(out) == before
    assert [p.name for p in out.iterdir() if p.name.startswith(".")] == []   # no staging or aside folder left


def test_an_unwritable_output_folder_fails_before_touching_anything(tmp_path):
    import os
    if os.geteuid() == 0:
        pytest.skip("root ignores folder permissions")
    eps = tmp_path / "eps"
    record(eps / "a", "a_s0", 0)
    out = tmp_path / "out"
    build(eps, out)
    before = snapshot(out)
    out.chmod(0o555)
    try:
        with pytest.raises(PermissionError):
            build(eps, out, force=True)
    finally:
        out.chmod(0o755)
    assert snapshot(out) == before


def test_a_symlinked_output_folder(tmp_path):
    eps = tmp_path / "eps"
    record(eps / "a", "a_s0", 0)
    real = tmp_path / "elsewhere"
    real.mkdir()
    link = tmp_path / "out"
    link.symlink_to(real)
    r = build(eps, link)
    assert (real / MANIFEST).exists() and r["examples"]["train"] + r["examples"]["test"] + r["examples"]["val"] > 0
    assert build(eps, link, force=True)["data_digest"] == r["data_digest"]


def test_output_inside_the_episodes_is_refused(tmp_path):
    record(tmp_path / "eps" / "a", "a_s0", 0)
    with pytest.raises(ValueError, match="inside the episodes"):
        build(tmp_path / "eps", tmp_path / "eps" / "datasets" / "b1")


def test_leftovers_of_a_killed_build(tmp_path):
    eps = tmp_path / "eps"
    record(eps / "a", "a_s0", 0)
    out = tmp_path / "out"
    (out / ".building-dead").mkdir(parents=True)              # an unfinished stage: swept, and not "content"
    (out / ".building-dead" / "x.jsonl").write_text("partial")
    build(eps, out)
    assert not (out / ".building-dead").exists()
    (out / ".previous-dead").mkdir()                          # killed mid-swap: may hold the only old copy
    r = build(eps, out, force=True)
    assert (out / ".previous-dead").exists() and any(".previous-dead" in w for w in r["warnings"])


def test_git_dirty_counts_untracked_code(tmp_path):
    import subprocess
    repo = tmp_path / "repo"
    (repo / "packages" / "pkg").mkdir(parents=True)
    (repo / "README.md").write_text("x")
    (repo / "packages" / "pkg" / "a.py").write_text("x = 1\n")

    def git(*a):
        subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)
    git("init", "-q")
    git("-c", "user.email=t@t", "-c", "user.name=t", "add", ".")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")
    where = repo / "packages" / "pkg"
    assert git_info(where)["dirty"] is False
    (repo / "README.md").write_text("changed outside packages/")       # not code a build runs
    assert git_info(where)["dirty"] is False
    (where / "new_module.py").write_text("y = 2\n")                    # untracked code: dirty
    g = git_info(where)
    assert g["dirty"] is True and g["dirty_paths"] == ["packages/pkg/new_module.py"] and g["scope"] == "packages"
