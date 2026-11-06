"""A mission file -> everything an episode needs."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from avians_env import WorldSpec
from avians_safety import load as load_limits
from avians_spec import MissionSpec, sar_registry


def repo_root() -> Path:
    env = os.environ.get("AVIANS_ROOT")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "missions").is_dir() and (p / "configs").is_dir():
            return p
    return Path.cwd()


@dataclass
class Mission:
    name: str
    brief: str
    world: WorldSpec
    spec: MissionSpec
    limits: object
    extras: dict
    horizon_s: float
    path: Path
    raw: dict

    @classmethod
    def load(cls, path: str | Path, envelope: str | None = None) -> "Mission":
        path = Path(path)
        root = repo_root()
        if not path.exists():
            path = root / path
        raw = yaml.safe_load(path.read_text())
        world = WorldSpec.load(root / raw["world"])
        limits, extras = load_limits(root / "configs/limits.yaml", envelope or raw.get("envelope"))
        reg = sar_registry(world, max_speed=extras["monitor_max_speed"], max_climb=limits.max_climb * 1.3)
        spec = MissionSpec.from_dict(raw["spec"], reg, world.regions)
        return cls(raw["name"], raw.get("brief", ""), world, spec, limits, extras,
                   float(raw.get("horizon_s", spec.horizon)), path, raw)


def load_config(name: str) -> dict:
    return yaml.safe_load((repo_root() / "configs" / name).read_text())
