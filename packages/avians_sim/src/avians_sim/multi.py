"""Several drones in one episode: one FastSim per drone around one Shared world.

    ms = MultiSim(world, seed, limits, n=2)
    ms.sims["d1"], ms.sims["d2"]        # each with its own Executor, events, reflex
    ms.step({"d1": sp1, "d2": sp2})     # one tick for all; the world advances once
    ms.signals()                        # battery_d1, ..., min_sep, min_dist_person, + shared

Drones launch a few metres apart on the pad. The per-drone signal names
match avians_spec.sar_registry(drones=[...]).
"""
from __future__ import annotations

import math

from .fast import FastSim, Shared


class MultiSim:
    realtime = False

    def __init__(self, world, seed: int, limits, critical_pct: float = 5.0, n: int = 2,
                 dt: float = 0.1, horizon_s: float = 3600.0, ground: dict | None = None,
                 realtime: bool = False, ids: list[str] | None = None, **kw):
        self.ids = ids or [f"d{k + 1}" for k in range(n)]
        self.shared = Shared(world, seed, limits, kw.get("camera"), kw.get("battery"), horizon_s, ground)
        self.sims: dict[str, FastSim] = {}
        for k, did in enumerate(self.ids):
            launch = (world.launch[0] + 25.0 * k, world.launch[1])      # a pad each, 25 m apart
            self.sims[did] = FastSim(world, seed, limits, critical_pct, dt=dt, horizon_s=horizon_s,
                                     drone_id=did, shared=self.shared, launch=launch, realtime=False,
                                     camera=kw.get("camera"), battery=kw.get("battery"))
        self.realtime = realtime
        self.dt = dt
        self._wall_next = None
        self.world, self.scenario, self.world_state = world, self.shared.scenario, self.shared.world_state
        self.tracker, self.coverage, self.ground, self.kits = (self.shared.tracker, self.shared.coverage,
                                                               self.shared.ground, self.shared.kits)
        self.reports, self.energy, self.supplies = self.shared.reports, self.shared.energy, self.shared.supplies

    @property
    def t(self) -> float:
        return next(iter(self.sims.values())).t

    @property
    def vehicle(self):                 # the first drone, for code that expects one
        return next(iter(self.sims.values())).vehicle

    @property
    def ctx(self):
        return next(iter(self.sims.values())).ctx

    def step(self, setpoints: dict) -> None:
        t0 = self.t
        for did, sim in self.sims.items():
            sim.vehicle.command(setpoints.get(did))
            sim.vehicle.step()
        if self.realtime:
            import time
            if self._wall_next is None:
                self._wall_next = time.monotonic()
            self._wall_next += self.dt
            delay = self._wall_next - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        t = self.t
        events = self.shared.advance(t0, t)
        for sim in self.sims.values():
            sim._observe(t0, events)

    def signals(self) -> dict[str, float]:
        out = {}
        states = []
        for did, sim in self.sims.items():
            for k, v in sim.drone_signals().items():
                out[f"{k}_{did}"] = v
            states.append(sim.vehicle.state())
        out.update(next(iter(self.sims.values())).shared_signals())
        pe, pn = self.world_state.person_position(self.t)
        out["min_dist_person"] = min(math.hypot(s.east - pe, s.north - pn) for s in states)
        seps = [math.dist(a.pos, b.pos) for i, a in enumerate(states) for b in states[i + 1:]]
        out["min_sep"] = min(seps) if seps else 1e3
        return out
