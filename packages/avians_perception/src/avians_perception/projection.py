"""From an image detection to a ground point (the Gazebo/YOLO tier).

AAS's yolo_py publishes vision_msgs/Detection2DArray: boxes in pixels. A box
centre (u, v) is a ray through a pinhole camera; with the drone's pose and the
camera's mounting (pitch down from horizontal), the ray meets flat ground at
one point. Flat ground is exact for sar_valley's plain ground; with terrain,
intersect a height map instead (TODO when Taylor adds terrain).
"""
from __future__ import annotations

import math

import numpy as np


def ground_point(u: float, v: float, width: int, height: int, hfov: float,
                 drone: tuple[float, float, float], yaw: float, pitch: float = math.pi / 2,
                 ) -> tuple[float, float] | None:
    """(east, north) where pixel (u, v) looks at the ground, or None if the ray
    points above the horizon. yaw: heading of the camera's forward axis, rad
    from east counter-clockwise (ENU). pitch: down from horizontal (pi/2 = nadir)."""
    f = (width / 2) / math.tan(hfov / 2)
    # camera frame: x forward (optical axis), y left, z up
    ray = np.array([f, -(u - width / 2), -(v - height / 2)])
    cp, sp = math.cos(pitch), math.sin(pitch)
    pitch_m = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])     # rotate the axis down by pitch
    cy, sy = math.cos(yaw), math.sin(yaw)
    yaw_m = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    d = yaw_m @ pitch_m @ ray
    if d[2] >= -1e-9:
        return None
    e0, n0, alt = drone
    s = alt / -d[2]
    return (e0 + s * d[0], n0 + s * d[1])


def detection_from_box(box_center, image_size, hfov, drone, yaw, pitch, t, score, alt_sigma=0.03):
    """A Detection (as the simulated detector makes) from one YOLO box."""
    from .detector import Detection
    p = ground_point(box_center[0], box_center[1], image_size[0], image_size[1], hfov, drone, yaw,
                     pitch)
    if p is None:
        return None
    return Detection(t, p[0], p[1], float(score), 0.5 + alt_sigma * drone[2], "yolo")
