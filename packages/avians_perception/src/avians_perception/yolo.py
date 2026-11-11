"""YOLO detections from the aerial-autonomy-stack, as the tracker's input.

AAS's yolo_py node publishes vision_msgs/Detection2DArray on the drone's
camera topic. This bridge (the ONLY perception module that imports rclpy)
subscribes, keeps the latest frame, and `frame(t, drone_pose)` turns the
person-class boxes into Detection objects by projecting each box's bottom
centre to the ground with the drone's pose and the camera mounting
(projection.ground_point), so the tracker downstream is the same as in the
fast tier.

UNTESTED: written without ROS at hand. Topic and class names live in
configs/aas.yaml (`perception:`). Calibrate the simulated detector against
what this reports (avians_perception.calibrate) so both tiers agree.
"""
from __future__ import annotations

import math
import threading

from .detector import Detection
from .projection import ground_point


class YoloBridge:
    def __init__(self, node, cfg: dict):
        from vision_msgs.msg import Detection2DArray
        self.cfg = cfg
        self.person_classes = set(str(c) for c in cfg.get("person_classes", ["person", "0"]))
        self.min_score = float(cfg.get("min_score", 0.25))
        self.width, self.height = int(cfg.get("width", 640)), int(cfg.get("height", 480))
        self.hfov = float(cfg.get("hfov", 1.2))
        self.pitch = float(cfg.get("pitch", math.pi / 2))     # pi/2 = nadir
        self.sigma_per_m = float(cfg.get("sigma_per_m", 0.03))
        self._lock = threading.Lock()
        self._latest = None
        self._stamp = None
        self.count = 0
        node.create_subscription(Detection2DArray, cfg["topic"], self._on_detections, 10)

    def _on_detections(self, msg) -> None:
        with self._lock:
            self._latest = msg
            self._stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.count += 1

    def frame(self, t: float, drone: tuple[float, float, float], yaw: float) -> list[Detection] | None:
        """The latest frame's person detections as ground points, or None if no
        new frame since the last call."""
        with self._lock:
            msg, self._latest = self._latest, None
        if msg is None:
            return None
        out = []
        for det in msg.detections:
            best = max(det.results, key=lambda r: r.hypothesis.score, default=None)
            if best is None or str(best.hypothesis.class_id) not in self.person_classes:
                continue
            if best.hypothesis.score < self.min_score:
                continue
            u = det.bbox.center.position.x
            v = det.bbox.center.position.y + det.bbox.size_y / 2       # feet, not the box centre
            p = ground_point(u, v, self.width, self.height, self.hfov, drone, yaw, self.pitch)
            if p is None:
                continue
            out.append(Detection(t, p[0], p[1], float(best.hypothesis.score),
                                 0.5 + self.sigma_per_m * drone[2], "yolo"))
        return out
