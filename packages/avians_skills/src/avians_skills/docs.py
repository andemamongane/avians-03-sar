"""The action space, written out from the registry (docs/action_space.md).

    python -m avians_skills.docs > docs/action_space.md

A test checks the committed file against this output, so the document can
never drift from the code.
"""
from __future__ import annotations

import json

from . import BENCHMARK_TOOLS, REGISTRY, STUDENT_ONLY, SWARM_ONLY

ALGORITHMS = {
    "goto": "visibility graph + A* (exact shortest path among polygons); Ruckig jerk-limited reference; pure pursuit",
    "fly_waypoints": "as goto, leg by leg",
    "path_plan": "visibility graph + A*; the battery model",
    "hover": "position hold",
    "wait_until": "position hold / idle on the ground",
    "orbit": "circular carrot with tangential feed-forward",
    "takeoff": "vertical reference",
    "land": "vertical reference; AAS land action in Gazebo",
    "precision_land": "transit + vertical reference",
    "rtl": "as precision_land to the launch point",
    "set_speed": "changes the envelope the planners use",
    "emergency_stop": "jerk-limited brake to a hold",
    "search_region": "IAMSAR parallel track (boustrophedon on the minimum rotated rectangle), expanding square, sector; crosshatch = two perpendicular passes",
    "scan": "straight low-altitude leg",
    "inspect": "hover over the estimate; the tracker's log-odds update",
    "track_person": "Kalman estimate + orbit / follow carrot",
    "intercept": "constant-velocity intercept (closed form), re-aimed every 3 s",
    "capture_image": "the detector on one frame",
    "drop_marker": "shared map annotation",
    "pick_up": "transit, land, depot load time",
    "deliver": "ballistic drop with quadratic drag and the wind estimate (release point by fixed-point iteration)",
    "recharge": "land on the pad; the battery model's charge rate",
    "swap_battery": "land on the pad; the station's swap time",
    "report": "ground station message; dispatches help",
    "request_help": "ground station question",
    "escalate": "hands the decision to the remote model",
    "handoff": "transfers the running skill (swarm)",
    "formation": "offset setpoints relative to a leader (swarm)",
    "assign_tasks": "Hungarian method (scipy linear_sum_assignment) (swarm)",
}


def render() -> str:
    out = ["# The action space", "",
           "Generated from the skill registry by `python -m avians_skills.docs`; do not edit by hand.",
           "Every tool also takes `reason` (one sentence) and `notes` (the pilot's memory). `keep` lets",
           "the running skill continue.", ""]
    cats = {}
    for name, cls in sorted(REGISTRY.items()):
        cats.setdefault(cls.category or "other", []).append((name, cls))
    for cat in ("movement", "search", "targets", "payload", "energy", "comms", "swarm", "other"):
        if cat not in cats:
            continue
        out += [f"## {cat}", "", "| tool | parameters | what it does | built on | set |", "|---|---|---|---|---|"]
        for name, cls in cats[cat]:
            params = []
            for k, v in cls.params.items():
                p = k if k in cls.required else f"{k}?"
                if "enum" in v:
                    p += f" ∈ {{{', '.join(map(str, v['enum']))}}}"
                if "default" in v:
                    p += f"={json.dumps(v['default'])}"
                params.append(p)
            which = ("benchmark" if name in BENCHMARK_TOOLS else "swarm" if name in SWARM_ONLY
                     else "student" if name in STUDENT_ONLY else "")
            kind = "instant" if cls.instant else "runs"
            out.append(f"| `{name}` ({kind}) | {', '.join(params) or '-'} | {cls.summary} | "
                       f"{ALGORITHMS.get(name, '')} | {which} |")
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    print(render(), end="")
