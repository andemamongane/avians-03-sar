"""Battery model: drain while flying, charge or swap at stations.

Drain rate (% per second) = hover + k_speed * v^2 + k_climb * max(0, vu):
the standard first-order multirotor power model (hover power dominates; parasitic
drag grows with v^2). The defaults give ~17 min of hover and ~14 min at 9 m/s
cruise from 100 %, a small quadrotor carrying a camera and a 1 kg kit.

The same model drives the fast simulator and the pilot's estimates (precheck,
"enough battery to reach pad_swap?"), so the pilot's arithmetic is checkable.
In the Gazebo tier it is still the battery of record: PX4's simulated battery
cannot be recharged mid-flight, so its failsafe is set below ours.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BatteryModel:
    hover_pct_s: float = 0.095
    k_speed: float = 0.00035          # % s / m^2
    k_climb: float = 0.02             # % per metre climbed
    payload_factor: float = 1.15      # carrying the kit costs 15 % more

    def drain(self, speed: float, climb: float, payload: bool) -> float:
        r = self.hover_pct_s + self.k_speed * speed ** 2 + self.k_climb * max(0.0, climb)
        return r * (self.payload_factor if payload else 1.0)

    def cost(self, distance_m: float, speed: float, payload: bool = False,
             hover_s: float = 0.0, climb_m: float = 0.0) -> float:
        """% needed to fly `distance_m` at `speed`, hover `hover_s`, climb `climb_m`."""
        t = distance_m / max(speed, 0.1)
        f = self.payload_factor if payload else 1.0
        return f * (self.drain(speed, 0.0, False) * t + self.hover_pct_s * hover_s
                    + self.k_climb * climb_m)

    def endurance_s(self, pct: float, speed: float, payload: bool = False) -> float:
        return pct / self.drain(speed, 0.0, payload)
