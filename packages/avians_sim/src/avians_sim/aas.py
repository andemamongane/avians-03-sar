"""The Gazebo tier: PX4 in Gazebo through the aerial-autonomy-stack, behind the
same Vehicle interface as the fast tier. The ONLY module that imports rclpy.

UNTESTED in avians-03 (written on a Mac without ROS; the Linux machine was
down). It reuses avians-02's tested pieces: the AAS action clients, the frame
handling, and the plan-ahead patch's `avians_trajectory` controller (built by
scripts/aas_plan_ahead.sh), which tracks a trajectory streamed on
/DroneN/avians/trajectory. What is new is how skills drive it:

    position setpoint   airborne: the shared reference generator (Ruckig, the
                        same one the fast tier flies) advances one tick and
                        its state is streamed as a 2-sample trajectory [now,
                        +1 s]; the controller tracks it and holds its last
                        point. The reference is re-seeded from the measured
                        state when PX4 falls more than 3 m behind it
                        on the ground: AAS takeoff action to the setpoint's
                        altitude, then offboard
    land                end offboard (patched px4_interface), AAS land action
    idle                nothing (on the ground)

Frames (avians-02, fact 1): east/north/alt are measured from where the drone
rests when it connects; PX4's local frame is NED.

Battery: the avians battery model stays the battery of record (PX4 SITL cannot
recharge); GazeboSim drains it from the measured speed.
"""
from __future__ import annotations

import math
import threading
import time

from avians_skills import Setpoint, VehicleState
from avians_skills.reference import ReferenceGenerator

from .fast import FastSim


class AASVehicle:
    def __init__(self, cfg: dict, limits=None, dt: float = 0.1):
        import rclpy
        from rclpy.action import ActionClient
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from autopilot_interface_msgs.action import Land, Offboard, Takeoff
        from px4_msgs.msg import VehicleLocalPosition, VehicleStatus
        from std_msgs.msg import Empty, Float64MultiArray

        self.msg = {"Land": Land, "Offboard": Offboard, "Takeoff": Takeoff, "Empty": Empty,
                    "F64": Float64MultiArray, "Status": VehicleStatus}
        if not rclpy.ok():
            rclpy.init()
        self.node = Node("avians_vehicle")
        ns = cfg.get("namespace", "/Drone1").rstrip("/")
        name = lambda k: f"{ns}/{cfg[k]}"                     # noqa: E731
        px4_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
                             history=HistoryPolicy.KEEP_LAST, depth=10)
        traj_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                              history=HistoryPolicy.KEEP_LAST, depth=1)
        self.cfg = cfg
        self.ref = ReferenceGenerator(limits, dt) if limits is not None else None
        self.s = VehicleState()
        self._lock = threading.Lock()
        self._raw = (0.0, 0.0, 0.0)
        self.origin: tuple | None = None
        self.armed = False
        self.phase = "ground"                 # ground, taking_off, offboard_starting, offboard, landing
        self._t0 = time.monotonic()
        self.node.create_subscription(VehicleLocalPosition, name("local_position"), self._on_pos, px4_qos)
        self.node.create_subscription(VehicleStatus, name("status"), self._on_status, px4_qos)
        self.takeoff_client = ActionClient(self.node, Takeoff, name("takeoff_action"))
        self.land_client = ActionClient(self.node, Land, name("land_action"))
        self.offboard_client = ActionClient(self.node, Offboard, name("offboard_action"))
        self.traj_pub = self.node.create_publisher(Float64MultiArray, name("trajectory_topic"), traj_qos)
        self.end_pub = self.node.create_publisher(Empty, name("offboard_end_topic"), QoSProfile(depth=1))
        self._spin = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self._spin.start()

    # ------------------------------------------------------------ state
    def _on_pos(self, m) -> None:
        with self._lock:
            self._raw = (float(m.x), float(m.y), float(m.z))
            ox, oy, oz = self.origin or (0.0, 0.0, 0.0)
            s = self.s
            s.north, s.east, s.alt = float(m.x) - ox, float(m.y) - oy, -(float(m.z) - oz)
            s.vn, s.ve, s.vu = float(m.vx), float(m.vy), -float(m.vz)
            if math.isfinite(m.heading):
                s.yaw = math.pi / 2 - float(m.heading)             # NED heading -> ENU yaw
            s.t = time.monotonic() - self._t0

    def _on_status(self, m) -> None:
        self.armed = m.arming_state == self.msg["Status"].ARMING_STATE_ARMED
        self.s.landed = not self.armed

    def state(self) -> VehicleState:
        self.s.t = time.monotonic() - self._t0
        return self.s

    def connect(self, timeout: float = 60.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and self.s.t == 0.0:
            time.sleep(0.1)
        if self.armed:
            raise RuntimeError("the drone is already armed: restart the simulation so the frame origin is the pad")
        with self._lock:
            self.origin = self._raw
        for c, label in ((self.takeoff_client, "takeoff_action"), (self.land_client, "land_action"),
                         (self.offboard_client, "offboard_action")):
            if not c.wait_for_server(timeout_sec=10):
                raise RuntimeError(f"AAS px4_interface does not serve {label}")
        if self.traj_pub.get_subscription_count() == 0:
            raise RuntimeError("no avians_trajectory controller: run scripts/aas_plan_ahead.sh and restart AAS")
        self._t0 = time.monotonic()

    # ------------------------------------------------------------ commands
    def _send(self, client, goal, on_done=None) -> None:
        def accepted(fut):
            gh = fut.result()
            if gh is None or not gh.accepted:
                if on_done:
                    on_done(False)
                return
            gh.get_result_async().add_done_callback(lambda r: on_done and on_done(bool(r.result().result.success)))
        client.send_goal_async(goal).add_done_callback(accepted)

    def _start_offboard(self) -> None:
        self.phase = "offboard_starting"
        g = self.msg["Offboard"].Goal()
        g.controller_name = "avians_trajectory"
        g.max_duration_sec = float(self.cfg.get("offboard_max_s", 7200))
        self._send(self.offboard_client, g, lambda ok: setattr(self, "phase", "ground" if not self.armed else "hover"))
        threading.Timer(1.5, lambda: self.phase == "offboard_starting" and setattr(self, "phase", "offboard")).start()

    def _stream(self, sp: Setpoint) -> None:
        ox, oy, oz = self.origin
        if self.ref is not None:
            s = self.s
            if math.dist(self.ref.pos, s.pos) > 3.0:
                self.ref.reset(s.pos, (s.ve, s.vn, s.vu))
            (e, n, a), (ve, vn, vu), _ = self.ref.step(sp)
        else:
            e, n, a = sp.pos
            ve, vn, vu = sp.vel
            v = math.sqrt(ve * ve + vn * vn + vu * vu)
            cap = sp.max_speed or v
            if v > cap > 0:
                ve, vn, vu = ve * cap / v, vn * cap / v, vu * cap / v
        data = []
        for t in (0.0, 1.0):
            data += [t, n + vn * t + ox, e + ve * t + oy, oz - (a + vu * t), vn, ve, -vu]
        self.traj_pub.publish(self.msg["F64"](data=data))

    def command(self, sp: Setpoint | None) -> None:
        if sp is None:
            return
        if sp.mode == "position" and sp.pos is not None:
            if self.phase == "ground" and not self.armed and sp.pos[2] > 0.3:
                self.phase = "taking_off"
                g = self.msg["Takeoff"].Goal()
                g.takeoff_altitude = float(sp.pos[2])
                self._send(self.takeoff_client, g, lambda ok: self._start_offboard() if ok else
                           setattr(self, "phase", "ground"))
            elif self.phase in ("hover",):
                self._start_offboard()
            elif self.phase == "offboard":
                self._stream(sp)
        elif sp.mode == "land" and self.phase not in ("landing", "ground"):
            if self.phase == "offboard":
                self.end_pub.publish(self.msg["Empty"]())
            self.phase = "landing"
            g = self.msg["Land"].Goal()
            g.landing_altitude = 0.0
            self._send(self.land_client, g, lambda ok: setattr(self, "phase", "ground"))

    def disconnect(self) -> None:
        try:
            self.node.destroy_node()
        except Exception:
            pass


class GazeboSim(FastSim):
    """FastSim's world, perception and actuators around the AAS vehicle, in
    real time. The camera is the simulated detector fed with the true person
    position (the YOLO path, avians_perception.projection, is the next step);
    released kits are shown in Gazebo where the drop model lands them."""

    realtime = True

    def __init__(self, world, seed, limits, critical_pct, aas_cfg: dict, control, **kw):
        super().__init__(world, seed, limits, critical_pct, **kw)
        self.control = control
        self.vehicle = AASVehicle(aas_cfg, limits, self.dt)
        self.vehicle.connect(float(aas_cfg.get("connect_timeout_s", 60)))
        self.ctx.vehicle = self.vehicle
        self._wall_next = time.monotonic()
        self._kits_shown = 0

    def step(self, sp: Setpoint | None) -> None:
        t0 = self.t
        self.vehicle.command(sp)
        self._wall_next += self.dt
        delay = self._wall_next - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        v = self.vehicle.state()
        if not v.landed:
            v.battery = max(0.0, v.battery - self.energy.drain(v.speed, v.vu, v.payload) * self.dt)
        self._observe(t0)
        for k in self.kits[self._kits_shown:]:
            if k["landed"]:
                self.control.kit(k["at"], self._kits_shown)
                self._kits_shown += 1
