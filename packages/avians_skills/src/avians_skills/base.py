"""The skill framework: what every high-level tool the pilot calls has in common.

A skill is one tool call ("search_region r3", "deliver to the person") that
runs for many control ticks until it succeeds, fails or is replaced. Each tick
it returns a Setpoint for the vehicle; the backend (fast kinematic model, or
PX4 through AAS) turns setpoints into motion. Skills never touch the vehicle
directly except through the context's actuators (payload, battery service,
radio), so the same skill code runs in every simulation tier.

    class Search(Skill):
        name = "search_region"
        summary = "Cover a search region with a lawnmower pattern..."
        params = {"region": {"type": "string", ...}}
        required = ["region"]

        def start(self, ctx): ...          # validate, plan; raise SkillError(why)
        def tick(self, ctx, dt) -> Setpoint: ...

Life cycle: status is running -> succeeded | failed | interrupted. Progress and
noteworthy moments go through ctx.emit(); the agent loop turns them into
decision points ("search of r3 done: 100 % covered, nothing found").
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol

REGISTRY: dict[str, type["Skill"]] = {}


class SkillError(ValueError):
    """A skill cannot start (or continue); the message says why, for the pilot."""


# ---------------------------------------------------------------- vehicle
@dataclass
class VehicleState:
    t: float = 0.0
    east: float = 0.0
    north: float = 0.0
    alt: float = 0.0
    ve: float = 0.0
    vn: float = 0.0
    vu: float = 0.0
    yaw: float = 0.0                # rad, ENU (0 = facing east)
    battery: float = 100.0          # %
    landed: bool = True
    payload: bool = False

    @property
    def pos(self) -> tuple[float, float, float]:
        return (self.east, self.north, self.alt)

    @property
    def speed(self) -> float:
        return math.hypot(self.ve, self.vn)

    @property
    def speed3(self) -> float:
        return math.sqrt(self.ve ** 2 + self.vn ** 2 + self.vu ** 2)


@dataclass
class Setpoint:
    """What the vehicle should do this tick.

    mode position: fly to `pos` (east, north, alt) arriving with velocity `vel`
                   (feed-forward; zero = stop there), no faster than max_speed.
    mode land:     descend at the landing speed where you are, then disarm.
    mode idle:     on the ground, disarmed (charging, loading, waiting).
    """
    mode: str = "position"
    pos: tuple[float, float, float] | None = None
    vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    max_speed: float | None = None
    yaw: float | None = None


class Vehicle(Protocol):
    """What a backend implements for skills (fast tier: avians_sim.fast;
    Gazebo tier: avians_sim.aas)."""

    def state(self) -> VehicleState: ...
    def command(self, sp: Setpoint) -> None: ...


@dataclass
class Limits:
    """The envelope skills plan with (avians_safety.Limits makes these)."""
    cruise_speed: float = 9.0
    max_speed: float = 12.0
    max_accel: float = 4.0
    max_jerk: float = 10.0
    max_climb: float = 3.0
    max_descent: float = 2.0
    min_alt: float = 5.0
    max_alt: float = 60.0
    cruise_alt: float = 30.0
    margin_m: float = 6.0            # keep this far from no-fly zones and obstacles
    hazard_margin_m: float = 3.0     # ...and this far from closed (hazard) zones
    reserve_pct: float = 10.0        # the battery floor

    def stopping_distance(self, v: float) -> float:
        return v * v / (2 * self.max_accel)


@dataclass
class Event:
    t: float
    kind: str                       # skill_done, skill_failed, progress, found, zone, ...
    text: str
    data: dict = field(default_factory=dict)
    wake: bool = True               # should the pilot be asked to decide?


# ---------------------------------------------------------------- context
@dataclass
class Context:
    """Everything a skill may read or use. The simulator fills it."""
    world: Any                      # avians_env.WorldSpec
    world_state: Any                # avians_env.WorldState (zone safety, reports)
    vehicle: Vehicle
    limits: Limits
    planner: Any                    # avians_skills.planning.PathPlanner
    camera: Any                     # avians_perception.CameraModel (footprint for coverage)
    actuators: Any                  # payload, battery service, radio (avians_sim implements)
    belief: Any = None              # callable -> avians_perception.Belief
    coverage: Any = None            # avians_skills.coverage.CoverageMap
    energy: Any = None              # avians_skills.energy.BatteryModel
    markers: list = field(default_factory=list)
    events: list = field(default_factory=list)

    @property
    def t(self) -> float:
        return self.vehicle.state().t

    @property
    def state(self) -> VehicleState:
        return self.vehicle.state()

    def emit(self, kind: str, text: str, wake: bool = True, **data) -> None:
        self.events.append(Event(self.t, kind, text, data, wake))

    def unsafe_zones(self):
        return self.world_state.unsafe_zones(self.t)


# ---------------------------------------------------------------- skills
class Skill:
    name: ClassVar[str] = ""
    summary: ClassVar[str] = ""
    params: ClassVar[dict] = {}
    required: ClassVar[list] = []
    instant: ClassVar[bool] = False         # completes inside start() (report, set_speed, ...)
    category: ClassVar[str] = ""

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if cls.name:
            REGISTRY[cls.name] = cls

    def __init__(self, **args):
        self.args = args
        self.status = "running"
        self.message = ""
        self.started_t: float | None = None

    # -- override
    def start(self, ctx: Context) -> None:
        pass

    def tick(self, ctx: Context, dt: float) -> Setpoint | None:
        return None

    def progress(self, ctx: Context) -> str:
        return ""

    def eta_s(self, ctx: Context) -> float | None:
        return None

    # -- helpers
    def succeed(self, ctx: Context, text: str, **data) -> None:
        self.status, self.message = "succeeded", text
        ctx.emit("skill_done", f"{self.name}: {text}", skill=self.name, **data)

    def fail(self, ctx: Context, text: str, **data) -> None:
        self.status, self.message = "failed", text
        ctx.emit("skill_failed", f"{self.name} failed: {text}", skill=self.name, **data)

    def interrupt(self, ctx: Context, why: str = "replaced") -> None:
        if self.status == "running":
            self.status, self.message = "interrupted", why

    @property
    def done(self) -> bool:
        return self.status != "running"

    def describe(self) -> str:
        args = ", ".join(f"{k}={v}" for k, v in self.args.items())
        return f"{self.name}({args})"

    @classmethod
    def tool(cls) -> dict:
        """The tool definition the pilot sees (Anthropic tool format; strict
        JSON schema)."""
        props = {}
        for k, v in cls.params.items():
            p = {kk: vv for kk, vv in v.items() if kk != "default"}
            if "default" in v:
                p["description"] = f"{p.get('description', '')} (default {v['default']})".strip()
            props[k] = p
        return {"name": cls.name, "description": cls.summary,
                "input_schema": {"type": "object", "properties": props,
                                 "required": list(cls.required), "additionalProperties": False}}


def get(name: str) -> type[Skill]:
    if name not in REGISTRY:
        raise SkillError(f"no skill named {name!r} (have: {', '.join(sorted(REGISTRY))})")
    return REGISTRY[name]


def create(name: str, args: dict) -> Skill:
    cls = get(name)
    unknown = set(args) - set(cls.params)
    if unknown:
        raise SkillError(f"{name} has no parameter(s) {sorted(unknown)}")
    missing = [k for k in cls.required if k not in args]
    if missing:
        raise SkillError(f"{name} needs {missing}")
    full = {k: v["default"] for k, v in cls.params.items() if "default" in v}
    full.update(args)
    for k, v in full.items():
        spec = cls.params[k]
        if "enum" in spec and v not in spec["enum"]:
            raise SkillError(f"{name}.{k} must be one of {spec['enum']}, not {v!r}")
        if spec.get("type") == "number":
            try:
                full[k] = float(v)
            except (TypeError, ValueError):
                raise SkillError(f"{name}.{k} must be a number, not {v!r}") from None
    return cls(**full)


def tools(names: list[str] | None = None) -> list[dict]:
    names = names or sorted(REGISTRY)
    return [REGISTRY[n].tool() for n in names]
