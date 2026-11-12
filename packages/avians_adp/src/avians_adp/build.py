"""Build training sets from a folder of episodes.

    python -m avians_adp.build <episodes dir> <out dir> [--force] [--text] [--thinking]

Phase 2 (this version): every trajectory.json under <episodes dir> is
loaded and validated; invalid ones are skipped and reported. Every decision
with an action becomes one pilot example, written unsplit as "all":

    <out>/pilot/all.jsonl          TRL layout (messages + tools)
    <out>/pilot/all.meta.jsonl     id + meta per row, same order
    <out>/pilot_lf/all.jsonl       LLaMA-Factory sharegpt layout
    <out>/pilot_lf/dataset_info.json

Filters and splits (Phase 3), the router set (Phase 4) and the manifest
(Phase 5) build on this. The output folder must be empty or absent unless
--force is given.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from .formats import write_pilot, write_sharegpt_info
from .schema import Trajectory
from .sft import pilot_examples
from .validation import validate


@dataclass
class Loaded:
    trajectories: list[Trajectory] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)     # episode dir -> why


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
    return out


OWNED = ("pilot", "pilot_lf")       # the only things a build writes, and the only things --force removes


def _prepare_out(episodes: Path, out: Path, force: bool) -> None:
    """Refuse an output folder that holds the episodes, is a file, or is not
    empty (without force). With force, remove only what a build writes:
    never the whole folder, which may hold anything."""
    ep, o = episodes.resolve(), out.resolve()
    if ep == o or o in ep.parents:
        raise ValueError(f"the output folder {out} contains the episodes {episodes}; choose another")
    if out.exists() and not out.is_dir():
        raise FileExistsError(f"{out} exists and is not a folder")
    if out.exists() and any(out.iterdir()):
        if not force:
            raise FileExistsError(f"{out} is not empty; pass force=True (--force) to replace a previous build")
        for name in OWNED:
            if (out / name).is_dir():
                shutil.rmtree(out / name)


def build(episodes: str | Path, out_dir: str | Path, *, force: bool = False, include_text: bool = False,
          include_thinking: bool = False) -> dict:
    out = Path(out_dir)
    _prepare_out(Path(episodes), out, force)
    loaded = load_valid(episodes)
    ids = [t.id for t in loaded.trajectories]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"duplicate episode ids under {episodes}: {dupes}")
    examples = [ex for tj in loaded.trajectories for ex in pilot_examples(tj)]
    written = write_pilot(out, "all", examples, include_text=include_text, include_thinking=include_thinking)
    write_sharegpt_info(out, ["all"])
    return {"episodes": ids, "skipped": loaded.skipped, "examples": len(examples), "written": written}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m avians_adp.build", description=__doc__.split("\n\n")[0])
    ap.add_argument("episodes", help="folder searched recursively for trajectory.json")
    ap.add_argument("out", help="output folder (must be empty unless --force)")
    ap.add_argument("--force", action="store_true", help="replace a non-empty output folder")
    ap.add_argument("--text", action="store_true", help="TRL layout: include the pilot's visible text")
    ap.add_argument("--thinking", action="store_true", help="TRL layout: include the thinking summary")
    a = ap.parse_args(argv)
    try:
        r = build(a.episodes, a.out, force=a.force, include_text=a.text, include_thinking=a.thinking)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    for ep, why in r["skipped"].items():
        print(f"skipped {ep}: {why}", file=sys.stderr)
    print(json.dumps({"episodes": len(r["episodes"]), "skipped": len(r["skipped"]), "examples": r["examples"],
                      "written": r["written"]}, indent=1))
    return 0 if r["episodes"] else 1


if __name__ == "__main__":
    sys.exit(main())
