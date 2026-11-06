"""The envelope file (configs/limits.yaml) -> what each layer needs."""
from __future__ import annotations

from pathlib import Path

import yaml

from avians_skills import Limits


def load(path: str | Path, envelope: str | None = None) -> tuple[Limits, dict]:
    """(Limits for the skills, extras: critical_pct, monitor_max_speed, name)."""
    cfg = yaml.safe_load(Path(path).read_text())
    name = envelope or cfg.get("default", "standard")
    if name not in cfg["envelopes"]:
        raise ValueError(f"no envelope {name!r} in {path} (have {', '.join(cfg['envelopes'])})")
    env = dict(cfg["envelopes"][name])
    critical = float(env.pop("critical_pct", 5.0))
    lim = Limits(**{k: float(v) for k, v in env.items()})
    extras = {"envelope": name, "critical_pct": critical,
              "monitor_max_speed": lim.max_speed * float(cfg.get("monitor_speed_margin", 1.3))}
    return lim, extras


def px4_params(lim: Limits) -> dict[str, float]:
    """PX4 v1.17 parameters enforcing the envelope (Gazebo tier). PX4's own
    battery failsafe is pushed below ours: the avians battery model is the
    battery of record (PX4 SITL cannot recharge mid-flight)."""
    return {
        "MPC_XY_VEL_MAX": lim.max_speed * 1.2,       # headroom so tracking can catch up
        "MPC_XY_CRUISE": lim.cruise_speed,
        "MPC_ACC_HOR_MAX": lim.max_accel,
        "MPC_JERK_MAX": lim.max_jerk,
        "MPC_Z_VEL_MAX_UP": lim.max_climb,
        "MPC_Z_VEL_MAX_DN": lim.max_descent,
        "GF_MAX_VER_DIST": lim.max_alt + 10.0,
        "GF_ACTION": 1.0,                            # warning; the reflex layer acts first
        "BAT_CRIT_THR": 0.02, "BAT_EMERGEN_THR": 0.01, "BAT_LOW_THR": 0.03,
        "SIM_BAT_DRAIN": 3600.0,
    }
