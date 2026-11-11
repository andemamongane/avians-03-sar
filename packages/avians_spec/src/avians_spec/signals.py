"""The signal registry: every signal a spec may name, with what bounds it.

A signal is a number per time step that the simulator writes and the spec
reads. The registry records, per signal, what the online monitor (ROSI) may
assume about its unobserved future:

    lo, hi      the physical range (battery 0..100, booleans 0..1)
    rate        the fastest it can change, per second (inf: it can jump)
    monotone    "up": it never decreases (latched events: found, released, help)

These bounds must be TRUE of the simulator, with margin, or the monitor can
call a violation the flight would have avoided. Every signal also has a
default, so every trace is defined at every step (the person estimate exists
before any detection: the prior, with a large sigma).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

INF = math.inf


@dataclass(frozen=True)
class Signal:
    name: str
    unit: str
    description: str
    lo: float = -INF
    hi: float = INF
    rate: float = INF
    monotone: str = ""          # "" or "up"
    default: float = 0.0
    boolean: bool = False       # 0/1; a bare name in a spec means "is 1"


@dataclass
class Registry:
    signals: dict[str, Signal] = field(default_factory=dict)
    domains: dict[str, list[str]] = field(default_factory=dict)    # "drones" -> ["d1", "d2"]

    def add(self, s: Signal) -> None:
        self.signals[s.name] = s

    def __contains__(self, name: str) -> bool:
        return name in self.signals

    def __getitem__(self, name: str) -> Signal:
        return self.signals[name]

    def names(self) -> list[str]:
        return list(self.signals)

    def defaults(self) -> dict[str, float]:
        return {k: s.default for k, s in self.signals.items()}

    def table(self) -> str:
        """Markdown table, for docs and the pilot's prompt."""
        rows = ["| signal | unit | meaning |", "|---|---|---|"]
        rows += [f"| `{s.name}` | {s.unit} | {s.description} |" for s in self.signals.values()]
        return "\n".join(rows)


POSITION = ("east", "north", "alt")


PER_DRONE = ("east", "north", "alt", "speed", "battery", "clearance", "payload", "landed", "dist_person")


def sar_registry(world=None, max_speed: float = 15.0, max_climb: float = 5.0,
                 max_drain_pct_s: float = 0.4, max_charge_pct_s: float = 1.0,
                 person_speed: float = 0.5, drones: list[str] | None = None) -> Registry:
    """The signals of the search-and-rescue benchmark. `max_speed` etc. are
    the vehicle's TRUE limits (envelope x margin), not the commanded ones.
    With `drones` (ids), the per-drone signals exist once per drone
    (battery_d1, ...), plus min_sep and min_dist_person; the unsuffixed
    names then belong to no drone and are absent."""
    r = Registry()
    v = max_speed
    per = [
        Signal("east", "m", "drone position east of launch", rate=v),
        Signal("north", "m", "drone position north of launch", rate=v),
        Signal("alt", "m", "drone altitude above ground", lo=0.0, rate=max_climb),
        Signal("speed", "m/s", "drone ground speed", lo=0.0, hi=v),
        Signal("battery", "%", "battery remaining", lo=0.0, hi=100.0, default=100.0,
               rate=max(max_drain_pct_s, max_charge_pct_s)),
        Signal("clearance", "m", "distance to the nearest obstacle", lo=0.0, rate=v, default=1e3),
        Signal("payload", "0/1", "carrying the medical kit", lo=0.0, hi=1.0, boolean=True),
        Signal("dist_person", "m", "true horizontal distance from the drone to the person",
               lo=0.0, rate=v + person_speed, default=1e3),
        Signal("landed", "0/1", "on the ground, disarmed or idle", lo=0.0, hi=1.0, boolean=True,
               default=1.0),
    ]
    if drones:
        r.domains["drones"] = list(drones)
        for d in drones:
            for s in per:
                r.add(Signal(f"{s.name}_{d}", s.unit, f"{s.description} ({d})", s.lo, s.hi, s.rate,
                             s.monotone, s.default, s.boolean))
        r.add(Signal("min_sep", "m", "smallest 3D distance between any two drones", lo=0.0,
                     rate=2 * v, default=1e3))
        r.add(Signal("min_dist_person", "m", "distance from the nearest drone to the person", lo=0.0,
                     rate=v + person_speed, default=1e3))
    else:
        for s in per:
            r.add(s)
    for s in (
        Signal("est_east", "m", "estimated person position east (the prior mean before any sighting)"),
        Signal("est_north", "m", "estimated person position north"),
        Signal("est_sigma", "m", "1-sigma uncertainty of the person estimate", lo=0.0),
        Signal("conf", "0..1", "detection confidence that the estimate is the person", lo=0.0, hi=1.0),
        Signal("found", "0/1", "confidence has reached the found threshold (latched)",
               lo=0.0, hi=1.0, monotone="up", boolean=True),
        Signal("released", "0/1", "a kit has been released (latched)", lo=0.0, hi=1.0,
               monotone="up", boolean=True),
        Signal("drop_err", "m", "distance from where the closest kit landed to the person (1000 before release)",
               lo=0.0, default=1e3),
        Signal("help_dispatched", "0/1", "help has been dispatched by a report (latched)",
               lo=0.0, hi=1.0, monotone="up", boolean=True),
        Signal("help", "0/1", "help has arrived at the person (latched)", lo=0.0, hi=1.0,
               monotone="up", boolean=True),
    ):
        r.add(s)
    zones = world.hazard_zones() if world is not None else []
    for z in zones:
        r.add(Signal(f"unsafe_{z.id}", "0/1", f"zone {z.id} is closed (time-varying)", lo=0.0,
                     hi=1.0, boolean=True))
    return r
