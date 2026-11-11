"""The reference generator: Setpoints become jerk-limited motion.

Ruckig (Berscheid & Kröger 2021) computes, every control tick, the
time-optimal jerk-limited trajectory from the current state (position,
velocity, acceleration) to the setpoint's target (position, velocity) under
per-axis velocity, acceleration and jerk limits, and returns the next state
along it. It is the same algorithm PX4's own setpoint smoothing descends
from, and it is what both tiers track:

    fast tier    FastVehicle IS the reference (plus ground contact, battery)
    Gazebo tier  AASVehicle streams the reference to the avians_trajectory
                 controller, which tracks it with PX4's position controller

The horizontal speed limit is a norm, not per-axis: the per-axis limits are
scaled by the direction to the target (with a floor) and the result is
clamped to the norm, which keeps straight flights at full speed and corners
inside the envelope. Vertical limits are asymmetric (climb vs descent).
"""
from __future__ import annotations

import math

from ruckig import ControlInterface, InputParameter, OutputParameter, Result, Ruckig, RuckigError

from .base import Limits, Setpoint


class ReferenceGenerator:
    def __init__(self, limits: Limits, dt: float):
        self.lim, self.dt = limits, dt
        self.otg = Ruckig(3, dt)
        self.inp = InputParameter(3)
        self.out = OutputParameter(3)
        self.inp.max_jerk = [limits.max_jerk] * 3
        self.pos = [0.0, 0.0, 0.0]
        self.vel = [0.0, 0.0, 0.0]
        self.acc = [0.0, 0.0, 0.0]

    def reset(self, pos, vel=(0.0, 0.0, 0.0)) -> None:
        self.pos, self.vel, self.acc = list(map(float, pos)), list(map(float, vel)), [0.0, 0.0, 0.0]

    def _prepare(self, sp: Setpoint) -> float:
        """Fill the Ruckig input from the setpoint; returns the horizontal speed cap.

        Two interfaces: a setpoint with a feed-forward velocity is path following,
        tracked through the VELOCITY interface (pure pursuit: the commanded velocity
        points at the carrot, the speed profile is the follower's); a setpoint
        without one (hover, hold, land, a waypoint to stop at) goes through the
        POSITION interface with zero target velocity."""
        lim, inp = self.lim, self.inp
        s = 0.98
        if sp.mode == "position" and sp.pos is not None:
            target = list(map(float, sp.pos))
            tvel = list(map(float, sp.vel))
            vmax = min(sp.max_speed or lim.max_speed, lim.max_speed)
        elif sp.mode == "land":
            target = [sp.pos[0], sp.pos[1], 0.0] if sp.pos else [self.pos[0], self.pos[1], 0.0]
            tvel = [0.0, 0.0, 0.0]
            vmax = 2.0
        else:                                   # idle / hold
            target, tvel, vmax = list(self.pos), [0.0, 0.0, 0.0], 2.0
        dx, dy, dz = target[0] - self.pos[0], target[1] - self.pos[1], target[2] - self.pos[2]
        d = math.hypot(dx, dy)
        vz_up = lim.max_climb if sp.mode != "land" else lim.max_descent
        following = math.hypot(tvel[0], tvel[1]) > 0.05 and d > 0.5
        if following:
            inp.control_interface = ControlInterface.Velocity
            v = min(vmax, math.hypot(tvel[0], tvel[1]))
            ux, uy = dx / d, dy / d                               # toward the carrot
            fx, fy = max(abs(ux), 0.35), max(abs(uy), 0.35)
            vz = max(-lim.max_descent, min(vz_up, 1.0 * dz + tvel[2]))
            inp.target_velocity = [s * v * ux, s * v * uy, s * vz]
            inp.target_position = target                          # ignored by the interface
        else:
            inp.control_interface = ControlInterface.Position
            fx, fy = (max(abs(dx) / d, 0.35), max(abs(dy) / d, 0.35)) if d > 1e-6 else (1.0, 1.0)
            inp.target_position = target
            inp.target_velocity = [0.0, 0.0, 0.0]
        inp.target_acceleration = [0.0, 0.0, 0.0]
        inp.max_velocity = [vmax * fx, vmax * fy, vz_up]
        inp.min_velocity = [-vmax * fx, -vmax * fy, -lim.max_descent]
        inp.max_acceleration = [lim.max_accel] * 2 + [lim.max_accel * 0.6]
        inp.current_position = self.pos
        # current state strictly inside the limits: the binding raises on equality within float noise
        inp.current_velocity = [max(lo * s, min(hi * s, v)) for v, lo, hi in
                                zip(self.vel, inp.min_velocity, inp.max_velocity)]
        inp.current_acceleration = [max(-a * s, min(a * s, x)) for x, a in zip(self.acc, inp.max_acceleration)]
        return vmax

    def step(self, sp: Setpoint) -> tuple[list[float], list[float], list[float]]:
        """Advance one tick toward the setpoint; returns (pos, vel, acc)."""
        lim = self.lim
        vmax = self._prepare(sp)
        try:
            res = self.otg.update(self.inp, self.out)
        except RuckigError:
            res = Result.ErrorInvalidInput
        if res not in (Result.Working, Result.Finished):
            # invalid input (e.g. current speed over the new limit): brake toward the target
            self.vel = [v * max(0.0, 1 - lim.max_accel * self.dt / max(math.hypot(*self.vel), 1e-6))
                        for v in self.vel]
            self.pos = [p + v * self.dt for p, v in zip(self.pos, self.vel)]
            self.acc = [0.0, 0.0, 0.0]
            return self.pos, self.vel, self.acc
        self.pos = list(self.out.new_position)
        self.vel = list(self.out.new_velocity)
        self.acc = list(self.out.new_acceleration)
        vh = math.hypot(self.vel[0], self.vel[1])
        if vh > vmax:                            # norm clamp (per-axis limits over-approximate)
            self.vel[0] *= vmax / vh
            self.vel[1] *= vmax / vh
        return self.pos, self.vel, self.acc

    def eta(self, sp: Setpoint) -> float:
        """Duration of the trajectory to the setpoint from the current state, s (no state change)."""
        self._prepare(sp)
        out = OutputParameter(3)
        try:
            res = self.otg.update(self.inp, out)
        except RuckigError:
            return math.inf
        return float(out.trajectory.duration) if res in (Result.Working, Result.Finished) else math.inf
