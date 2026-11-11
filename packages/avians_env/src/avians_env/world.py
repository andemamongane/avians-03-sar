"""The world: static geometry plus the timelines that make it dynamic.

One YAML file (worlds/<name>/world.yaml) describes everything a mission flies
in. Coordinates are metres east/north of the launch point, altitudes metres
above ground. Three layers:

    WorldSpec   what the file says: regions, obstacles, stations, depots, the
                missing person's prior, decoys, safety timeline, help timeline.
    Scenario    one draw of the random parts (person's position and motion,
                help delay), fixed by a seed: the same seed is the same episode.
    WorldState  the running world during an episode: where the person is now,
                which zones are safe now, whether help has been dispatched or
                has arrived. The simulator owns one and steps it.

Everything the pilot reads about the world comes from describe() (static) and
status() (what changed), so the wording lives here, next to the data.

Contract (other packages rely on these names; change them together):
    WorldSpec.load(path), .regions, .obstacles, .stations, .depots, .launch,
    .bounds, .max_alt, .search_regions(), .hazard_zones(), .sample(seed)
    Region.robustness(e, n, a) -> signed distance array (+ inside)
    Scenario.person_position(t)
    WorldState.zone_safe(zone, t), .changes(t0, t1), .report_found(t, e, n),
    .help_arrived(t), .signals(t), .describe(), .status(t)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import shapely
import yaml
from shapely.geometry import Point, Polygon

REGION_KINDS = ("search", "no_fly", "hazard", "landing", "other")
STATION_KINDS = ("charge", "swap")


class WorldError(ValueError):
    pass


# ---------------------------------------------------------------- geometry
@dataclass
class Region:
    id: str
    kind: str
    polygon: list[tuple[float, float]]
    name: str = ""
    altitude: tuple[float, float] | None = None     # (floor, ceiling); None: ground to sky
    description: str = ""

    def __post_init__(self):
        self.polygon = [tuple(map(float, p)) for p in self.polygon]
        self.shape = Polygon(self.polygon)
        if not self.shape.is_valid or self.shape.area <= 0:
            raise WorldError(f"region {self.id}: polygon is not a valid simple polygon")

    def contains(self, e: float, n: float, a: float | None = None) -> bool:
        return bool(self.robustness(np.array([e]), np.array([n]),
                                    np.array([a if a is not None else self._mid_alt()]))[0] > 0)

    def _mid_alt(self) -> float:
        return 0.5 * sum(self.altitude) if self.altitude else 10.0

    def robustness(self, e, n, a) -> np.ndarray:
        """Signed distance to the region's boundary, metres, + inside. With an
        altitude band, the vertical margins count too (min of the three)."""
        e, n, a = np.atleast_1d(e).astype(float), np.atleast_1d(n).astype(float), np.atleast_1d(a)
        d = shapely.distance(self.shape.exterior, shapely.points(e, n))
        inside = shapely.contains_xy(self.shape, e, n)
        rho = np.where(inside, d, -d)
        if self.altitude is not None:
            lo, hi = self.altitude
            rho = np.minimum(rho, np.minimum(a - lo, hi - a))
        return rho

    @property
    def centroid(self) -> tuple[float, float]:
        c = self.shape.centroid
        return (c.x, c.y)

    @property
    def size(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.shape.bounds
        return (x1 - x0, y1 - y0)


@dataclass
class Obstacle:
    """Something solid the drone can hit: a prism from the ground to `height`."""
    id: str
    polygon: list[tuple[float, float]]
    height: float
    name: str = ""
    description: str = ""

    def __post_init__(self):
        self.polygon = [tuple(map(float, p)) for p in self.polygon]
        self.shape = Polygon(self.polygon)

    def clearance(self, e, n, a) -> np.ndarray:
        """Distance from the point to the obstacle (0 inside): horizontal if
        below its top, else the 3D distance to its top face's footprint."""
        e, n, a = np.atleast_1d(e).astype(float), np.atleast_1d(n).astype(float), np.atleast_1d(a)
        h = shapely.distance(self.shape, shapely.points(e, n))
        above = np.maximum(0.0, a - self.height)
        return np.hypot(h, above)


@dataclass
class Station:
    id: str
    kind: str                       # charge | swap
    position: tuple[float, float]
    name: str = ""
    charge_rate_pct_s: float = 0.8  # charge: battery % per second on the pad
    swap_s: float = 40.0            # swap: seconds on the pad for a fresh battery
    description: str = ""


@dataclass
class Depot:
    id: str
    position: tuple[float, float]
    supplies: int = 1
    load_s: float = 8.0             # seconds on the pad to load one package
    name: str = ""
    description: str = ""


@dataclass
class PersonSpec:
    """The missing person: where they probably are, and how they move."""
    region_weights: dict[str, float] = field(default_factory=dict)   # prior over search regions
    motion: str = "static"          # static | random_walk
    speed: float = 0.0              # m/s, random_walk only
    description: str = ""


@dataclass
class Decoy:
    """Something that looks like a person to a detector, more or less."""
    id: str
    position: tuple[float, float]
    strength: float = 0.3           # 0..1, how person-like
    description: str = ""


@dataclass
class SafetyEvent:
    t: float
    zone: str
    safe: bool
    note: str = ""


@dataclass
class HelpSpec:
    delay_mean_s: float = 600.0     # from the report to arrival
    delay_std_s: float = 60.0
    wrong_location_m: float = 30.0  # a report further than this from the person...
    search_penalty_s: float = 180.0  # ...costs the crew this long to find them
    description: str = "Rescue crew"


@dataclass
class Wind:
    east: float = 0.0
    north: float = 0.0
    gust_std: float = 0.0


@dataclass
class Kit:
    """The medical kit as a dropped object (avians_skills.ballistics)."""
    mass: float = 1.0          # kg
    cd_area: float = 0.03      # drag coefficient x frontal area, m^2
    rho: float = 1.225         # air density, kg/m^3

    def params(self) -> dict:
        return {"mass": self.mass, "cd_area": self.cd_area, "rho": self.rho}


# ---------------------------------------------------------------- the file
@dataclass
class WorldSpec:
    name: str
    regions: dict[str, Region]
    description: str = ""
    origin: tuple[float, float, float] = (47.397742, 8.545594, 488.0)   # lat, lon, alt of launch
    launch: tuple[float, float] = (0.0, 0.0)
    bounds: list[tuple[float, float]] = field(default_factory=list)    # operating area (geofence)
    max_alt: float = 60.0
    obstacles: dict[str, Obstacle] = field(default_factory=dict)
    stations: dict[str, Station] = field(default_factory=dict)
    depots: dict[str, Depot] = field(default_factory=dict)
    person: PersonSpec = field(default_factory=PersonSpec)
    decoys: list[Decoy] = field(default_factory=list)
    safety_timeline: list[SafetyEvent] = field(default_factory=list)
    help: HelpSpec = field(default_factory=HelpSpec)
    wind: Wind = field(default_factory=Wind)
    kit: Kit = field(default_factory=Kit)
    source: Path | None = None

    # ------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: str | Path) -> "WorldSpec":
        path = Path(path)
        if path.is_dir():
            path = path / "world.yaml"
        w = cls.from_dict(yaml.safe_load(path.read_text()))
        w.source = path
        return w

    @classmethod
    def from_dict(cls, d: dict) -> "WorldSpec":
        def pos(v):
            return (float(v[0]), float(v[1]))
        regions = {}
        for r in d.get("regions", []):
            alt = tuple(map(float, r["altitude"])) if r.get("altitude") else None
            regions[r["id"]] = Region(r["id"], r["kind"], r["polygon"], r.get("name", ""),
                                      alt, r.get("description", ""))
        obstacles = {}
        for o in d.get("obstacles", []):
            if "circle" in o:
                c, rad = o["circle"]["center"], float(o["circle"]["radius"])
                poly = [(c[0] + rad * math.cos(k * math.pi / 8), c[1] + rad * math.sin(k * math.pi / 8))
                        for k in range(16)]
            else:
                poly = o["polygon"]
            obstacles[o["id"]] = Obstacle(o["id"], poly, float(o["height"]), o.get("name", ""),
                                          o.get("description", ""))
        stations = {s["id"]: Station(s["id"], s["kind"], pos(s["position"]), s.get("name", ""),
                                     float(s.get("charge_rate_pct_s", 0.8)),
                                     float(s.get("swap_s", 40.0)), s.get("description", ""))
                    for s in d.get("stations", [])}
        depots = {x["id"]: Depot(x["id"], pos(x["position"]), int(x.get("supplies", 1)),
                                 float(x.get("load_s", 8.0)), x.get("name", ""),
                                 x.get("description", ""))
                  for x in d.get("depots", [])}
        p = d.get("person", {})
        person = PersonSpec({k: float(v) for k, v in p.get("region_weights", {}).items()},
                            p.get("motion", "static"), float(p.get("speed", 0.0)),
                            p.get("description", ""))
        decoys = [Decoy(x["id"], pos(x["position"]), float(x.get("strength", 0.3)),
                        x.get("description", "")) for x in d.get("decoys", [])]
        dyn = d.get("dynamics", {})
        timeline = sorted((SafetyEvent(float(e["t"]), e["zone"], bool(e["safe"]), e.get("note", ""))
                           for e in dyn.get("zone_safety", [])), key=lambda e: e.t)
        h = dyn.get("help", {})
        help_ = HelpSpec(float(h.get("delay_mean_s", 600)), float(h.get("delay_std_s", 60)),
                         float(h.get("wrong_location_m", 30)), float(h.get("search_penalty_s", 180)),
                         h.get("description", "Rescue crew"))
        wd = d.get("wind", {})
        kd = d.get("kit", {})
        o = d.get("origin", {})
        w = cls(
            name=d["name"], regions=regions, description=d.get("description", ""),
            origin=(float(o.get("lat", 47.397742)), float(o.get("lon", 8.545594)),
                    float(o.get("alt", 488.0))),
            launch=pos(d.get("launch", [0, 0])),
            bounds=[pos(b) for b in d.get("bounds", [])], max_alt=float(d.get("max_alt", 60)),
            obstacles=obstacles, stations=stations, depots=depots, person=person, decoys=decoys,
            safety_timeline=timeline, help=help_,
            wind=Wind(float(wd.get("east", 0)), float(wd.get("north", 0)), float(wd.get("gust_std", 0))),
            kit=Kit(float(kd.get("mass", 1.0)), float(kd.get("cd_area", 0.03)), float(kd.get("rho", 1.225))))
        errors = w.validate()
        if errors:
            raise WorldError("; ".join(errors))
        return w

    def validate(self) -> list[str]:
        errs = []
        for r in self.regions.values():
            if r.kind not in REGION_KINDS:
                errs.append(f"region {r.id}: kind must be one of {REGION_KINDS}")
        for s in self.stations.values():
            if s.kind not in STATION_KINDS:
                errs.append(f"station {s.id}: kind must be one of {STATION_KINDS}")
        for zid in {e.zone for e in self.safety_timeline}:
            if zid not in self.regions:
                errs.append(f"safety timeline names unknown zone {zid}")
        for rid in self.person.region_weights:
            if rid not in self.regions:
                errs.append(f"person prior names unknown region {rid}")
        if self.person.motion not in ("static", "random_walk"):
            errs.append("person motion must be static or random_walk")
        ids = [*self.regions, *self.obstacles, *self.stations, *self.depots]
        if len(ids) != len(set(ids)):
            errs.append("ids must be unique across regions, obstacles, stations and depots")
        return errs

    # ------------------------------------------------------------ queries
    def search_regions(self) -> list[Region]:
        return [r for r in self.regions.values() if r.kind == "search"]

    def hazard_zones(self) -> list[Region]:
        """Regions whose safety changes over time (the safety timeline's zones)."""
        zones = {e.zone for e in self.safety_timeline}
        return [r for r in self.regions.values() if r.id in zones or r.kind == "hazard"]

    def no_fly(self) -> list[Region]:
        return [r for r in self.regions.values() if r.kind == "no_fly"]

    def bounds_shape(self) -> Polygon | None:
        return Polygon(self.bounds) if len(self.bounds) >= 3 else None

    def place(self, pid: str) -> tuple[float, float]:
        """Position of a station, depot, region (centroid) or 'launch'."""
        if pid == "launch":
            return self.launch
        for table in (self.stations, self.depots):
            if pid in table:
                return table[pid].position
        if pid in self.regions:
            return self.regions[pid].centroid
        raise KeyError(pid)

    def sample(self, seed: int, horizon_s: float = 3600.0) -> "Scenario":
        return Scenario.draw(self, seed, horizon_s)

    def describe(self) -> str:
        """The static map, in words, for the pilot's prompt."""
        lines = [f"WORLD {self.name}: {self.description}".strip(),
                 f"Coordinates are metres east/north of the launch point (0, 0); max altitude "
                 f"{self.max_alt:g} m."]
        if self.bounds:
            xs, ys = zip(*self.bounds)
            lines.append(f"Operating area: east {min(xs):g}..{max(xs):g}, north {min(ys):g}..{max(ys):g}.")
        for r in self.regions.values():
            (w, h), (cx, cy) = r.size, r.centroid
            band = f", altitude {r.altitude[0]:g}-{r.altitude[1]:g} m" if r.altitude else ""
            lines.append(f"- {r.id} ({r.kind}) '{r.name}': {w:.0f} x {h:.0f} m centred at "
                         f"({cx:.0f}, {cy:.0f}){band}. {r.description}".rstrip())
        for o in self.obstacles.values():
            c = o.shape.centroid
            lines.append(f"- obstacle {o.id} '{o.name}': {o.height:g} m tall at ({c.x:.0f}, {c.y:.0f}), "
                         f"{2 * math.sqrt(o.shape.area / math.pi):.0f} m across. {o.description}".rstrip())
        for s in self.stations.values():
            what = (f"charges at {s.charge_rate_pct_s:g} %/s" if s.kind == "charge"
                    else f"swaps the battery in {s.swap_s:g} s")
            lines.append(f"- station {s.id} ({s.kind}) at ({s.position[0]:g}, {s.position[1]:g}): "
                         f"{what}. {s.description}".rstrip())
        for d in self.depots.values():
            lines.append(f"- depot {d.id} at ({d.position[0]:g}, {d.position[1]:g}): {d.supplies} "
                         f"medical kit(s), {d.load_s:g} s to load. {d.description}".rstrip())
        if self.person.description:
            lines.append(f"Missing person: {self.person.description}")
        if self.person.region_weights:
            prior = ", ".join(f"{k} {v:.0%}" for k, v in sorted(self.person.region_weights.items(),
                                                              key=lambda kv: -kv[1]))
            lines.append(f"Where they probably are (prior): {prior}.")
        return "\n".join(lines)


# ---------------------------------------------------------------- one draw
@dataclass
class Scenario:
    world: WorldSpec
    seed: int
    person_track: np.ndarray        # (k, 3): t, east, north, 1 s apart
    help_delay_s: float

    @classmethod
    def draw(cls, w: WorldSpec, seed: int, horizon_s: float = 3600.0) -> "Scenario":
        rng = np.random.default_rng(seed)
        start = _sample_person(w, rng)
        n = int(horizon_s) + 1
        track = np.zeros((n, 3))
        track[:, 0] = np.arange(n)
        track[0, 1:] = start
        if w.person.motion == "random_walk" and w.person.speed > 0:
            heading = rng.uniform(0, 2 * math.pi)
            for k in range(1, n):
                heading += rng.normal(0, 0.3)
                step = w.person.speed * np.array([math.cos(heading), math.sin(heading)])
                nxt = track[k - 1, 1:] + step
                if not _free(w, nxt):
                    heading += math.pi
                    nxt = track[k - 1, 1:]
                track[k, 1:] = nxt
        else:
            track[:, 1:] = start
        delay = max(60.0, rng.normal(w.help.delay_mean_s, w.help.delay_std_s))
        return cls(w, seed, track, float(delay))

    def person_position(self, t: float) -> tuple[float, float]:
        tr = self.person_track
        k = min(max(t, 0.0), tr[-1, 0])
        return (float(np.interp(k, tr[:, 0], tr[:, 1])), float(np.interp(k, tr[:, 0], tr[:, 2])))


def _free(w: WorldSpec, p) -> bool:
    pt = Point(float(p[0]), float(p[1]))
    if any(o.shape.buffer(2.0).contains(pt) for o in w.obstacles.values()):
        return False
    b = w.bounds_shape()
    return b is None or b.contains(pt)


def _sample_person(w: WorldSpec, rng) -> tuple[float, float]:
    weights = w.person.region_weights or {r.id: 1.0 for r in w.search_regions()}
    ids = sorted(weights)
    p = np.array([weights[i] for i in ids], float)
    region = w.regions[ids[rng.choice(len(ids), p=p / p.sum())]]
    x0, y0, x1, y1 = region.shape.buffer(-8.0).bounds   # not on the very edge
    inner = region.shape.buffer(-8.0)
    for _ in range(1000):
        e, n = rng.uniform(x0, x1), rng.uniform(y0, y1)
        if inner.contains(Point(e, n)) and _free(w, (e, n)):
            return (float(e), float(n))
    raise WorldError(f"could not place the person in {region.id}")


# ---------------------------------------------------------------- running
@dataclass
class HelpStatus:
    dispatched_t: float | None = None
    reported_at: tuple[float, float] | None = None
    arrival_t: float | None = None


class WorldState:
    """The world during one episode. Pure function of time plus the reports
    the drone makes; the simulator calls report_found() when the pilot's
    report skill reports the person."""

    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.world = scenario.world
        self.help = HelpStatus()

    def person_position(self, t: float) -> tuple[float, float]:
        return self.scenario.person_position(t)

    def zone_safe(self, zone: str, t: float) -> bool:
        safe = True
        for e in self.world.safety_timeline:
            if e.zone == zone and e.t <= t:
                safe = e.safe
        return safe

    def unsafe_zones(self, t: float) -> list[Region]:
        return [r for r in self.world.hazard_zones() if not self.zone_safe(r.id, t)]

    def changes(self, t0: float, t1: float) -> list[SafetyEvent]:
        """Safety changes in (t0, t1]."""
        return [e for e in self.world.safety_timeline if t0 < e.t <= t1]

    def report_found(self, t: float, east: float, north: float) -> str:
        """The drone reports the person at (east, north). Dispatches help the
        first time; returns what the crew says back."""
        if self.help.dispatched_t is not None:
            return (f"{self.world.help.description} already dispatched at "
                    f"t={self.help.dispatched_t:.0f} s; ETA t={self.help.arrival_t:.0f} s")
        pe, pn = self.person_position(t)
        delay = self.scenario.help_delay_s
        if math.hypot(east - pe, north - pn) > self.world.help.wrong_location_m:
            delay += self.world.help.search_penalty_s           # they have to look around
        self.help = HelpStatus(t, (east, north), t + delay)
        return (f"{self.world.help.description} dispatched to ({east:.0f}, {north:.0f}); "
                f"ETA about {delay / 60:.0f} min")

    def help_arrived(self, t: float) -> bool:
        return self.help.arrival_t is not None and t >= self.help.arrival_t

    def signals(self, t: float) -> dict[str, float]:
        """The world's own signals at t (see avians_spec.signals)."""
        out = {"help": 1.0 if self.help_arrived(t) else 0.0,
               "help_dispatched": 1.0 if self.help.dispatched_t is not None and
               t >= self.help.dispatched_t else 0.0}
        for z in self.world.hazard_zones():
            out[f"unsafe_{z.id}"] = 0.0 if self.zone_safe(z.id, t) else 1.0
        return out

    def describe(self) -> str:
        return self.world.describe()

    def status(self, t: float) -> str:
        """What is true now that the static map does not say."""
        lines = []
        for z in self.world.hazard_zones():
            since = max((e.t for e in self.world.safety_timeline if e.zone == z.id and e.t <= t),
                        default=None)
            notes = [e.note for e in self.world.safety_timeline
                     if e.zone == z.id and e.t <= t and e.note]
            state = "SAFE" if self.zone_safe(z.id, t) else "UNSAFE (do not enter)"
            lines.append(f"- {z.id}: {state}" + (f" since t={since:.0f} s" if since else "")
                         + (f" ({notes[-1]})" if notes else ""))
        h = self.help
        if h.dispatched_t is None:
            lines.append("- help: not dispatched (report the person's location to dispatch it)")
        elif not self.help_arrived(t):
            lines.append(f"- help: dispatched at t={h.dispatched_t:.0f} s, arriving in about "
                         f"{max(0.0, h.arrival_t - t) / 60:.0f} min")
        else:
            lines.append(f"- help: ARRIVED at t={h.arrival_t:.0f} s")
        return "\n".join(lines)
