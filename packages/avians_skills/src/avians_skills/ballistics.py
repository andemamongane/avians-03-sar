"""Where a dropped kit lands: a point mass with quadratic drag in a steady
wind, integrated to the ground (the standard airdrop computation, e.g. the
CARP "computed air release point" of military airdrop, reduced to one object).

    a = g - (rho * Cd * A / 2m) * |v - w| (v - w)

The kit's mass, drag area and the air density come from the world file (so
Taylor's payload model and this one agree). `release_point` inverts the
model: where to release, with the drone's velocity, so that the kit lands at
the target (fixed-point iteration, converges in a few steps because drift is
small compared with the fall).
"""
from __future__ import annotations

import math

G = 9.81


def landing(release_pos, release_vel, wind, mass: float = 1.0, cd_area: float = 0.03,
            rho: float = 1.225, dt: float = 0.01) -> tuple[tuple[float, float], float]:
    """((east, north) where it lands, seconds in the air)."""
    k = rho * cd_area / (2 * mass)
    x, y, z = map(float, release_pos)
    vx, vy, vz = map(float, release_vel)
    we, wn = float(wind[0]), float(wind[1])
    t = 0.0
    while z > 0 and t < 60:
        rx, ry, rz = vx - we, vy - wn, vz
        r = math.sqrt(rx * rx + ry * ry + rz * rz)
        ax, ay, az = -k * r * rx, -k * r * ry, -G - k * r * rz
        vx, vy, vz = vx + ax * dt, vy + ay * dt, vz + az * dt
        x, y, z = x + vx * dt, y + vy * dt, z + vz * dt
        t += dt
    return (x, y), t


def release_point(target, alt: float, drone_vel, wind, **kit) -> tuple[float, float]:
    """(east, north) to be at, at `alt` with velocity `drone_vel`, so the kit lands on target."""
    e, n = float(target[0]), float(target[1])
    for _ in range(6):
        (le, ln), _t = landing((e, n, alt), drone_vel, wind, **kit)
        e, n = e + (target[0] - le), n + (target[1] - ln)
    return (e, n)


def fall_time(alt: float, **kit) -> float:
    return landing((0.0, 0.0, alt), (0.0, 0.0, 0.0), (0.0, 0.0), **kit)[1]
