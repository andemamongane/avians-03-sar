"""Build training sets from a folder of episodes.

    python -m avians_adp.build <episodes dir> <out dir> [options]

Pipeline, in this order:
    find        every trajectory.json under <episodes dir>
    validate    invalid or unreadable episodes are skipped and reported
    episodes    EpisodeFilter: --satisfied, --min-robustness, --require-clauses, --pilot, --tier
    split       by seed (splits.py): every episode of a seed in one split
    decisions   DecisionFilter: drop API errors, no-tool-call and router-commit keeps,
                rejected and unanswered calls (--allow keeps some), --model
    thin        train only: near-identical keep runs to one in --keep-every
    write       per split, both layouts:
                    <out>/pilot/<split>.jsonl          TRL (messages + tools)
                    <out>/pilot/<split>.meta.jsonl     id + meta per row, same order
                    <out>/pilot_lf/<split>.jsonl       LLaMA-Factory sharegpt
                    <out>/pilot_lf/dataset_info.json   avians_pilot_train / _val / _test
    router      for episodes with a student_calls.jsonl (student_calls.py), with the
                same episodes and splits; never thinned, no decision filter beyond
                skipping keeps the teacher never chose (routing.py):
                    <out>/router/<split>.jsonl         id, text (the router's features), label, options
                    <out>/router/<split>.meta.jsonl    id + teacher/student calls, why, state, events

The output folder must not contain the episodes, and must be empty unless
--force is given (which removes only pilot/, pilot_lf/ and router/). The
manifest (Phase 5) builds on this.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from .filters import BATTERY_BAND, DROPPABLE, Counts, DecisionFilter, EpisodeFilter, thin_keeps
from .formats import write_jsonl, write_pilot, write_sharegpt_info
from .routing import TOLERANCES, router_examples
from .schema import Trajectory
from .sft import pilot_examples
from .splits import SPLITS, check_seed, parse_ratios, split_of
from .student_calls import read_calls
from .validation import validate

OWNED = ("pilot", "pilot_lf", "router")       # the only things a build writes, and the only things --force removes


@dataclass
class BuildConfig:
    episodes: EpisodeFilter = field(default_factory=EpisodeFilter)
    decisions: DecisionFilter = field(default_factory=DecisionFilter)
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    keep_every: int = 3
    battery_band: float = BATTERY_BAND
    include_text: bool = False
    include_thinking: bool = False
    tolerances: dict[str, float] = field(default_factory=lambda: dict(TOLERANCES))


@dataclass
class Loaded:
    trajectories: list[Trajectory] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)     # episode dir -> why
    dirs: dict[str, Path] = field(default_factory=dict)       # episode id -> its folder


def find_episodes(root: str | Path) -> list[Path]:
    return sorted(p.parent for p in Path(root).rglob("trajectory.json"))


def load_valid(root: str | Path) -> Loaded:
    out = Loaded()
    for ep in find_episodes(root):
        try:
            tj = Trajectory.model_validate_json((ep / "trajectory.json").read_text())
        except (ValidationError, ValueError, OSError) as e:
            out.skipped[str(ep)] = f"does not load: {str(e).splitlines()[0]}"
            continue
        problems = validate(tj)
        if problems:
            out.skipped[str(ep)] = "invalid: " + "; ".join(problems[:3])
            continue
        out.trajectories.append(tj)
        out.dirs.setdefault(tj.id, ep)
    return out


def _router_features():
    """The live router's feature function, or None without avians_agent."""
    try:
        from avians_agent.router import features
    except ImportError:
        return None
    return features


def _read_events(ep: Path) -> list[dict]:
    p = ep / "events.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _check_out(episodes: Path, out: Path, force: bool) -> None:
    """Refuse an output folder that holds the episodes, is a file, or is not
    empty (without force). Deletes nothing: see _clear_out."""
    ep, o = episodes.resolve(), out.resolve()
    if ep == o or o in ep.parents:
        raise ValueError(f"the output folder {out} contains the episodes {episodes}; choose another")
    if out.exists() and not out.is_dir():
        raise FileExistsError(f"{out} exists and is not a folder")
    if out.exists() and any(out.iterdir()) and not force:
        raise FileExistsError(f"{out} is not empty; pass force=True (--force) to replace a previous build")


def _clear_out(out: Path) -> None:
    """With force, remove only what a build writes, never the whole folder
    (it may hold anything). Called only once every input check has passed,
    so a bad option never costs the previous build."""
    for name in OWNED:
        if (out / name).is_dir():
            shutil.rmtree(out / name)


def build(episodes: str | Path, out_dir: str | Path, config: BuildConfig | None = None, *,
          force: bool = False) -> dict:
    cfg = config or BuildConfig()
    if cfg.keep_every < 1:
        raise ValueError("keep_every must be >= 1")
    out = Path(out_dir)
    _check_out(Path(episodes), out, force)

    loaded = load_valid(episodes)
    ids = [t.id for t in loaded.trajectories]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"duplicate episode ids under {episodes}: {dupes}")
    cfg.episodes.check_clause_names(loaded.trajectories)

    counts = Counts()
    by_split: dict[str, list[Trajectory]] = {s: [] for s in SPLITS}
    for tj in loaded.trajectories:
        why = cfg.episodes.reject_reason(tj)
        if why:
            counts.add(counts.episodes, why)
            continue
        seed = tj.details.get("meta", {}).get("seed")
        try:
            check_seed(seed)
        except ValueError:
            counts.add(counts.episodes, "no integer seed")
            continue
        by_split[split_of(seed, cfg.ratios)].append(tj)

    examples: dict[str, list[dict]] = {}
    for split, trajs in by_split.items():
        kept = []
        for tj in trajs:
            for ex in pilot_examples(tj):
                why = cfg.decisions.reject_reason(ex)
                if why:
                    counts.add(counts.decisions, why)
                else:
                    kept.append(ex)
        if split == "train":
            kept, counts.thinned = thin_keeps(kept, cfg.keep_every, cfg.battery_band)
        examples[split] = kept

    router: dict[str, list[dict]] = {s: [] for s in SPLITS}
    router_skipped: dict[str, int] = {}
    router_warnings: list[str] = []
    with_calls = {t.id: read_calls(loaded.dirs[t.id]) for s in SPLITS for t in by_split[s]}
    with_calls = {k: v for k, v in with_calls.items() if v is not None}
    for ep_id, calls in list(with_calls.items()):       # calls written for another (re-flown) episode
        stale = {c["episode"] for c in calls if c.get("episode") not in (None, ep_id)}
        if stale:
            router_warnings.append(f"{ep_id}: student_calls.jsonl belongs to {sorted(stale)}; ignored "
                                   "(rerun student_calls with --force)")
            del with_calls[ep_id]
    features = _router_features() if with_calls else None
    if with_calls and features is None:
        router_warnings.append("avians_agent is not installed: router examples have no `text`")
    for split, trajs in by_split.items():
        for tj in trajs:
            calls = with_calls.get(tj.id)
            if calls is None:
                continue
            exs, skipped = router_examples(tj, calls, features=features, tolerances=cfg.tolerances,
                                           events=_read_events(loaded.dirs[tj.id]), models=cfg.decisions.models)
            router[split].extend(exs)
            for k, v in skipped.items():
                router_skipped[k] = router_skipped.get(k, 0) + v
    inexact = sum(1 for s in SPLITS for e in router[s] if not e["meta"]["events_exact"])
    if inexact:
        router_warnings.append(f"{inexact} router examples rebuilt their events from a time window "
                               "(records made before the recorder stored them): the text may differ")

    _clear_out(out)
    written = {}
    for split in SPLITS:
        written.update(write_pilot(out, split, examples[split], include_text=cfg.include_text,
                                   include_thinking=cfg.include_thinking))
    write_sharegpt_info(out, list(SPLITS))
    if with_calls:
        for split in SPLITS:
            p = out / "router" / f"{split}.jsonl"
            written[str(p)] = write_jsonl(p, ({"id": e["id"], "text": e["text"], "label": e["label"],
                                               "options": e["options"]} for e in router[split]))
            p = out / "router" / f"{split}.meta.jsonl"
            written[str(p)] = write_jsonl(p, ({"id": e["id"], **e["meta"], "state": e["state"],
                                               "events": e["events"]} for e in router[split]))

    warnings = [f"split {s!r} is empty" for s, r in zip(SPLITS, cfg.ratios) if r > 0 and not examples[s]]
    warnings += router_warnings
    return {
        "episodes": {s: [t.id for t in by_split[s]] for s in SPLITS},
        "examples": {s: len(examples[s]) for s in SPLITS},
        "removed": {"episodes": counts.episodes, "decisions": counts.decisions, "thinned": counts.thinned},
        "router": {"episodes": len(with_calls),
                   "examples": {s: len(router[s]) for s in SPLITS},
                   "labels": {lab: sum(1 for s in SPLITS for e in router[s] if e["label"] == lab)
                              for lab in ("commit", "local", "escalate")},
                   "skipped": router_skipped},
        "skipped": loaded.skipped,
        "warnings": warnings,
        "written": written,
    }


def _names(text: str | None) -> tuple[str, ...]:
    return tuple(x.strip() for x in (text or "").split(",") if x.strip())


def config_from_args(a: argparse.Namespace) -> BuildConfig:
    return BuildConfig(
        episodes=EpisodeFilter(satisfied=a.satisfied, min_robustness=a.min_robustness,
                               require_clauses=_names(a.require_clauses), clause_margin=a.clause_margin,
                               pilots=_names(a.pilot), tiers=_names(a.tier)),
        decisions=DecisionFilter(allow=_names(a.allow), models=_names(a.model)),
        ratios=parse_ratios(a.split), keep_every=a.keep_every, battery_band=a.battery_band,
        include_text=a.text, include_thinking=a.thinking)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m avians_adp.build", description="Build pilot SFT sets.")
    ap.add_argument("episodes", help="folder searched recursively for trajectory.json")
    ap.add_argument("out", help="output folder (must be empty unless --force)")
    ap.add_argument("--force", action="store_true", help="replace a previous build (only pilot/ and pilot_lf/)")
    g = ap.add_argument_group("episodes")
    g.add_argument("--satisfied", action="store_true", help="only episodes that satisfied the whole spec")
    g.add_argument("--min-robustness", type=float, help="only episodes with robustness >= this")
    g.add_argument("--require-clauses", help="comma list of clauses that must be satisfied, e.g. battery,deliver")
    g.add_argument("--clause-margin", type=float, default=0.0, help="required clauses need robustness >= this")
    g.add_argument("--pilot", help="comma list of pilots (meta.pilot), e.g. haiku")
    g.add_argument("--tier", help="comma list of tiers, e.g. fast")
    g = ap.add_argument_group("decisions")
    g.add_argument("--allow", help=f"comma list of normally dropped kinds to keep: {', '.join(DROPPABLE)}")
    g.add_argument("--model", help="comma list of models (meta.model), e.g. claude-haiku-5-5")
    g.add_argument("--keep-every", type=int, default=3, help="train: keep 1 in N of a run of identical keeps")
    g.add_argument("--battery-band", type=float, default=BATTERY_BAND,
                   help="battery band (percent) in the keep signature; 0 ignores battery (default 25)")
    g = ap.add_argument_group("output")
    g.add_argument("--split", default="80/10/10", help="train/val/test ratios by seed (default 80/10/10)")
    g.add_argument("--text", action="store_true", help="TRL layout: include the pilot's visible text")
    g.add_argument("--thinking", action="store_true", help="TRL layout: include the thinking summary")
    a = ap.parse_args(argv)
    try:
        r = build(a.episodes, a.out, config_from_args(a), force=a.force)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    for ep, why in r["skipped"].items():
        print(f"skipped {ep}: {why}", file=sys.stderr)
    for w in r["warnings"]:
        print(f"warning: {w}", file=sys.stderr)
    print(json.dumps({"episodes": {s: len(v) for s, v in r["episodes"].items()}, "examples": r["examples"],
                      "removed": r["removed"], "router": r["router"], "skipped": len(r["skipped"])}, indent=1))
    return 0 if sum(r["examples"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
