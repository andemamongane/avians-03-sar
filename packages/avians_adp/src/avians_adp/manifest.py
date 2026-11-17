"""manifest.json: what a dataset build was made from, how, and what it holds.

Every build writes one next to its files. It answers, without the episodes
at hand: which episodes went into which split, what each filter removed, the
code (git commit) and the settings that made it, and whether the files are
the ones it describes (a sha256 and row count per file).

Two builds of the same episodes with the same settings and code give the
same data files; only `created` (and `git.dirty` if the tree changed)
differs. `data_digest` is one hash over all data files, to compare builds
at a glance.
"""
from __future__ import annotations

import dataclasses
import hashlib
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .formats import ENCODING
from .schema import SCHEMA_VERSION, UPSTREAM
from .splits import SPLITS

MANIFEST = "manifest.json"
FORMAT = "avians_adp.manifest/1"
CHAT_TEMPLATE = "qwen3"


def git_info(where: Path | None = None) -> dict | None:
    """The commit of the code that ran; None when that cannot be known.

    Looks at the checkout this package runs from. An installed copy (inside
    site-packages) is not that checkout's code, so it gives None rather than
    a commit that may not match. `dirty` covers the code a build runs (the
    repo's packages/ folder, untracked files included), not the whole repo;
    `dirty_paths` lists the first changes."""
    here = Path(where or Path(__file__).resolve().parent)
    if where is None and any(p in ("site-packages", "dist-packages") for p in here.parts):
        return None
    cwd = str(here)

    def run(*args):
        return subprocess.run(["git", "-C", cwd, *args], capture_output=True, text=True, timeout=10, check=True)
    try:
        top = Path(run("rev-parse", "--show-toplevel").stdout.strip())
        commit = run("rev-parse", "HEAD").stdout.strip()
        branch = run("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        scope = str(top / "packages") if (top / "packages").is_dir() else str(top)
        changes = run("status", "--porcelain", "--untracked-files=all", "--", scope).stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        return None
    paths = [c[3:] for c in changes if c.strip() and "__pycache__" not in c and not c.endswith(".pyc")]
    return {"commit": commit, "branch": branch, "dirty": bool(paths), "dirty_paths": paths[:20],
            "scope": Path(scope).name}


def package_version() -> str:
    try:
        from importlib.metadata import version
        return version("avians-adp")
    except Exception:  # noqa: BLE001  (not installed as a distribution: still build)
        return "unknown"


def file_entry(path: Path) -> dict:
    data = path.read_bytes()
    entry = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    if path.suffix == ".jsonl":
        entry["rows"] = data.count(b"\n")
    return entry


def plain(obj):
    """Dataclasses and tuples as JSON-ready dicts and lists."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    return obj


def make_manifest(*, root: Path, episodes_dir: Path, config, found: int, loaded_ids: list[str],
                  skipped: dict[str, str], by_split: dict[str, list[str]], examples: dict[str, list[dict]],
                  removed: dict, router: dict[str, list[dict]] | None, router_skipped: dict[str, int],
                  warnings: list[str], argv: list[str] | None = None) -> dict:
    """`root` is the folder holding the written files (paths are recorded relative to it)."""
    files = {str(p.relative_to(root)): file_entry(p)
             for p in sorted(root.rglob("*")) if p.is_file() and p.name != MANIFEST}
    digest = hashlib.sha256("".join(f"{k}:{v['sha256']}\n" for k, v in files.items()).encode()).hexdigest()
    pilot = {
        "episodes": {s: list(by_split[s]) for s in SPLITS},
        "examples": {s: len(examples[s]) for s in SPLITS},
        "by_tool": {s: dict(sorted(Counter(e["call"]["name"] for e in examples[s]).items())) for s in SPLITS},
        "by_pilot": {s: dict(sorted(Counter(str(e["meta"].get("pilot")) for e in examples[s]).items()))
                     for s in SPLITS},
        "by_model": {s: dict(sorted(Counter(str(e["meta"].get("model")) for e in examples[s]).items()))
                     for s in SPLITS},
        "removed": removed,
    }
    out = {
        "format": FORMAT,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "avians_adp": package_version(),
        "adp_schema": {"version": SCHEMA_VERSION, "upstream": UPSTREAM},
        "git": git_info(),
        "command": argv,
        "source": {"episodes_dir": str(Path(episodes_dir).resolve()), "found": found,
                   "loaded": len(loaded_ids), "skipped": dict(sorted(skipped.items()))},
        "config": plain(config),
        "encoding": {**ENCODING, "chat_template": CHAT_TEMPLATE},
        "pilot": pilot,
        "router": None,
        "warnings": list(warnings),
        "files": files,
        "data_digest": digest,
    }
    if router is not None:
        out["router"] = {
            "examples": {s: len(router[s]) for s in SPLITS},
            "labels": {s: dict(sorted(Counter(e["label"] for e in router[s]).items())) for s in SPLITS},
            "episodes": {s: sorted({e["meta"]["episode"] for e in router[s]}) for s in SPLITS},
            "skipped": dict(sorted(router_skipped.items())),
            "events_inexact": sum(1 for s in SPLITS for e in router[s] if not e["meta"]["events_exact"]),
        }
    return out
