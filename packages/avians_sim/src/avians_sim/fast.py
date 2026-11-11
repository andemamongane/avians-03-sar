"""The fast tier: a kinematic multirotor in a simulated world, ~1000x real time.

Built for rollouts (behaviour cloning data, RL, benchmark sweeps): no
rendering, no ROS, a fixed step (0.1 s). What it models:

    vehicle     PX4's multicopter position-control structure: P on position
                plus velocity feed-forward, horizontal speed/accel and
                vertical climb/descent limits; ground contact; take-off and
                landing; the avians battery model
    camera      SimulatedDetector at its frame rate -> PersonTracker belief
    coverage    the camera footprint marks each search region's grid
    payload     loaded at depots; released kits fall (drag-factored fall time)
                and drift with the wind, landing with noise
    world       avians_env.WorldState: zone closures, help dispatch/arrival

The Gazebo tier (aas.py) implements the same Vehicle interface over PX4, so
skills, the reflex layer and the pilot loop are the same in both.
"""
from __future__ import annotations

import math

import numpy as np

from avians_env import WorldState
from avians_perception import CameraModel, PersonTracker, SimulatedDetector
from avians_safety import Reflex
from avians_skills import BatteryModel, Context, CoverageMap, Limits, PathPlanner, Setpoint, VehicleState
from avians_skills.ballistics import landing
from avians_skills.reference import ReferenceGenerator

from .ground import GroundStation


class FastVehicle:
    """The reference generator (Ruckig, avians_skills.reference) plus ground
    contact, battery and heading. In the fast tier the drone IS its reference:
    the tracking error PX4 adds is left to the Gazebo tier."""

    def __init__(self, limits: Limits, battery: BatteryModel, launch=(0.0, 0.0), dt: float = 0.1):
        self.lim, self.bat, self.dt = limits, battery, dt
        self.s = VehicleState(east=launch[0], north=launch[1])
        self.sp = Setpoint("idle")
        self.ref = ReferenceGenerator(limits, dt)
        self.ref.reset((launch[0], launch[1], 0.0))

    def state(self) -> VehicleState:
        return self.s

    def command(self, sp: Setpoint | None) -> None:
        if sp is not None:
            self.sp = sp

    def step(self) -> None:
        s, sp, dt = self.s, self.sp, self.dt
        if any(abs(a - b) > 1e-9 for a, b in zip(self.ref.pos, (s.east, s.north, s.alt))):
            self.ref.reset((s.east, s.north, s.alt), (s.ve, s.vn, s.vu))   # state set from outside
        if s.landed and (sp.mode != "position" or sp.pos is None or sp.pos[2] < 0.3):
            s.ve = s.vn = s.vu = 0.0
            self.ref.reset((s.east, s.north, 0.0))
            s.t += dt
            return
        if sp.mode == "land" and s.alt < 2.0:          # the last metres: slow, straight down
            sp = Setpoint("land", (s.east, s.north, 0.0))
        pos, vel, _ = self.ref.step(sp)
        s.east, s.north, s.alt = pos[0], pos[1], max(0.0, pos[2])
        s.ve, s.vn, s.vu = vel[0], vel[1], vel[2]
        s.landed = False
        if s.alt <= 0.02 and (sp.mode == "land" or s.vu < 0):
            s.alt, s.ve, s.vn, s.vu, s.landed = 0.0, 0.0, 0.0, 0.0, True
            self.ref.reset((s.east, s.north, 0.0))
        if not s.landed:
            s.battery = max(0.0, s.battery - self.bat.drain(s.speed, s.vu, s.payload) * dt)
        if s.speed > 0.3:
            s.yaw = math.atan2(s.vn, s.ve)
        if sp.yaw is not None:
            s.yaw = sp.yaw
        s.t += dt


class FastActuators:
    """Payload, battery service, radio and camera, for the skills."""

    def __init__(self, sim: "FastSim"):
        self.sim = sim
        self.swarm = None

    def load(self, depot_id: str) -> str | None:
        left = self.sim.supplies.get(depot_id, 0)
        if left <= 0:
            return f"depot {depot_id} has no kits left"
        self.sim.supplies[depot_id] = left - 1
        self.sim.vehicle.s.payload = True
        return None

    def release(self) -> str:
        return self.sim.release_payload()

    def charge(self, station_id: str, dt: float) -> None:
        st = self.sim.world.stations[station_id]
        v = self.sim.vehicle.s
        if v.landed and math.dist((v.east, v.north), st.position) < 3.0:
            v.battery = min(100.0, v.battery + st.charge_rate_pct_s * dt)

    def swap(self, station_id: str) -> None:
        st = self.sim.world.stations[station_id]
        v = self.sim.vehicle.s
        if v.landed and math.dist((v.east, v.north), st.position) < 3.0:
            v.battery = 100.0

    def wind_estimate(self) -> tuple[float, float]:
        """What the drone infers from its tilt while holding: the wind plus noise."""
        w, rng = self.sim.scenario.wind, self.sim.rng
        return (w.east + rng.normal(0, 0.3), w.north + rng.normal(0, 0.3))

    def report(self, kind: str, east, north, message: str) -> str:
        sim, t = self.sim, self.sim.t
        sim.reports.append({"t": t, "kind": kind, "east": east, "north": north, "message": message})
        text = message or kind
        if kind == "person_found":
            text = f"person found at ({east:.0f}, {north:.0f})" + (f": {message}" if message else "")
        return sim.ground.send(t, sim.drone_id, kind, text, sim.vehicle.state().pos, sim.world_state,
                               east=east, north=north)

    def ask_operator(self, question: str) -> str:
        sim = self.sim
        sim.reports.append({"t": sim.t, "kind": "question", "message": question})
        return sim.ground.send(sim.t, sim.drone_id, "question", question, sim.vehicle.state().pos)

    def capture(self, note: str) -> str:
        sim = self.sim
        v = sim.vehicle.s
        dets = sim.detector.frame(sim.t, v.pos, sim.world_state.person_position(sim.t), sim.scenario.decoys)
        sim.images.append({"t": sim.t, "note": note, "pos": v.pos, "detections": len(dets)})
        r = sim.camera.footprint(v.alt)
        if not dets:
            return f"photo at {v.alt:.0f} m ({r:.0f} m around): no person-like objects"
        parts = [f"({d.east:.0f}, {d.north:.0f}) score {d.score:.2f}" for d in dets]
        return f"photo at {v.alt:.0f} m ({r:.0f} m around): {len(dets)} object(s): " + "; ".join(parts)


class FastSim:
    """One episode's world + vehicle + perception, stepped at dt."""

    def __init__(self, world, seed: int, limits: Limits, critical_pct: float = 5.0,
                 camera: CameraModel | None = None, battery: BatteryModel | None = None,
                 dt: float = 0.1, horizon_s: float = 3600.0, ground: dict | None = None,
                 realtime: bool = False, drone_id: str = "drone1"):
        self.world, self.seed, self.dt = world, seed, dt
        self.realtime, self.drone_id = realtime, drone_id
        self._wall_next = None
        self.rng = np.random.default_rng(seed + 1000)
        self.scenario = world.sample(seed, horizon_s)
        self.world_state = WorldState(self.scenario)
        self.camera = camera or CameraModel()
        self.energy = battery or BatteryModel()
        self.vehicle = FastVehicle(limits, self.energy, world.launch, dt)
        self.detector = SimulatedDetector(self.camera, seed + 2000)
        self.tracker = PersonTracker.from_world(world)
        self.coverage = CoverageMap(world.search_regions())
        self.planner = PathPlanner(world, limits.margin_m, hazard_margin_m=limits.hazard_margin_m)
        self.reflex = Reflex(world, limits, critical_pct)
        self.supplies = {d.id: d.supplies for d in world.depots.values()}
        self.reports: list[dict] = []
        self.images: list[dict] = []
        self.kits: list[dict] = []           # released kits: in flight, then landed
        self.ground = GroundStation(world, ground, np.random.default_rng(seed + 3000))
        self.actuators = FastActuators(self)
        self.ctx = Context(world=world, world_state=self.world_state, vehicle=self.vehicle,
                           limits=limits, planner=self.planner, camera=self.camera,
                           actuators=self.actuators, belief=self.tracker.belief,
                           coverage=self.coverage, energy=self.energy)
        self._next_frame = 0.0
        self._found_emitted = False
        self._battery_marks = sorted({30.0, 20.0, limits.reserve_pct + 5}, reverse=True)

    realtime = False

    @property
    def t(self) -> float:
        return self.vehicle.state().t

    # ------------------------------------------------------------ physics
    def release_payload(self) -> str:
        v = self.vehicle.s
        v.payload = False
        w, kit = self.scenario.wind, self.world.kit.params()
        gust = self.rng.normal(0, w.gust_std, 2) if w.gust_std > 0 else np.zeros(2)
        kit["cd_area"] *= self.rng.normal(1.0, 0.1)                # tumbling: the drag area varies
        (le, ln), tf = landing(v.pos, (v.ve, v.vn, v.vu), (w.east + gust[0], w.north + gust[1]), **kit)
        noise = self.rng.normal(0, 0.3 + 0.03 * v.alt, 2)          # release jitter, bounce
        land = (le + noise[0], ln + noise[1])
        self.kits.append({"t_release": self.t, "t_land": self.t + tf, "at": land, "landed": False,
                          "err": None})
        return f"(falls for ~{tf:.1f} s)"

    def step(self, sp: Setpoint | None) -> None:
        t0 = self.t
        self.vehicle.command(sp)
        self.vehicle.step()
        if self.realtime:                        # the fast tier slowed to the wall clock
            import time
            if self._wall_next is None:
                self._wall_next = time.monotonic()
            self._wall_next += self.dt
            delay = self._wall_next - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        self._observe(t0)

    def _observe(self, t0: float) -> None:
        """Everything after the vehicle moved: camera, kits, world timeline, battery marks."""
        ctx = self.ctx
        t, v = self.t, self.vehicle.state()
        # camera
        if t + 1e-9 >= self._next_frame:
            self._next_frame = t + 1.0 / self.camera.rate_hz
            person = self.world_state.person_position(t)
            dets = self.detector.frame(t, v.pos, person, self.scenario.decoys) if not v.landed else []
            r = self.camera.footprint(v.alt)
            b = self.tracker.update(t, dets, (v.east, v.north, r) if not v.landed else None)
            if not v.landed:
                self.coverage.mark(t, v.east, v.north, v.alt, r)
            if b.found and not self._found_emitted:
                self._found_emitted = True
                ctx.emit("found", f"person found: estimate ({b.east:.0f}, {b.north:.0f}) +/- "
                         f"{b.sigma:.1f} m, confidence {b.conf:.2f}")
        # kits in flight
        for k in self.kits:
            if not k["landed"] and t >= k["t_land"]:
                k["landed"] = True
                pe, pn = self.world_state.person_position(t)
                k["err"] = math.dist(k["at"], (pe, pn))
                b = self.tracker.belief()
                seen = math.dist(k["at"], (b.east, b.north))
                ctx.emit("kit_landed", f"kit landed at ({k['at'][0]:.0f}, {k['at'][1]:.0f}), "
                         f"{seen:.1f} m from the person estimate")
        # world timeline
        for e in self.world_state.changes(t0, t):
            ctx.emit("zone", f"{e.zone} is now {'SAFE' if e.safe else 'CLOSED'}" + (f": {e.note}" if e.note else ""))
        for n in self.world_state.news(t0, t):
            self.ground.push(t, "scenario", "news", n.text + (f" (the description of {n.about} is updated)"
                                                             if n.description else ""))
        for m in self.ground.inbox(t0, t, self.drone_id):
            ctx.emit("message", f"radio from {m.sender} [{m.kind}]: {m.text}", sender=m.sender, kind_=m.kind)
        h = self.world_state.help
        if h.arrival_t is not None and t0 < h.arrival_t <= t:
            ctx.emit("help", f"{self.world.help.description} has arrived at the person")
        # battery marks
        while self._battery_marks and v.battery < self._battery_marks[0]:
            ctx.emit("battery", f"battery at {self._battery_marks.pop(0):.0f} %")

    # ------------------------------------------------------------ signals
    def signals(self) -> dict[str, float]:
        t, v = self.t, self.vehicle.s
        pe, pn = self.world_state.person_position(t)
        clearance = min((float(o.clearance(v.east, v.north, v.alt)[0])
                         for o in self.world.obstacles.values()), default=1e3)
        landed = [k for k in self.kits if k["landed"]]
        out = {"east": v.east, "north": v.north, "alt": v.alt, "speed": v.speed,
               "battery": v.battery, "clearance": clearance,
               "payload": 1.0 if v.payload else 0.0,
               "released": 1.0 if self.kits else 0.0,
               "drop_err": min((k["err"] for k in landed), default=1e3),
               "dist_person": math.hypot(v.east - pe, v.north - pn),
               "landed": 1.0 if v.landed else 0.0}
        out.update(self.tracker.belief().signals())
        out.update(self.world_state.signals(t))
        return out

    def truth(self) -> dict:
        """Ground truth for logs and replays (never shown to the pilot)."""
        pe, pn = self.world_state.person_position(self.t)
        return {"person": (pe, pn), "kits": [k["at"] for k in self.kits if k["landed"]]}
