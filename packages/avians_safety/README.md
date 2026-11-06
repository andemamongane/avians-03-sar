# avians_safety: PX4 enforces, the reflex catches, the pre-check explains

- **`limits.py`:** `configs/limits.yaml`, translated for each layer: the
  `Limits` skills plan with, the PX4 parameters for the Gazebo tier, and the
  online monitor's speed bound.
- **`precheck.py`:** before a skill starts, it rejects calls that cannot be
  right (a target inside a no-fly zone or obstacle, outside the envelope,
  airborne skills on the ground) and warns about risky ones (not enough
  battery to get there and on to a station). The reasons are written for the
  pilot.
- **`reflex.py`:** runtime assurance in the simplex architecture. Every
  control tick it checks whether braking now would still stop the drone
  short of a no-fly zone, closed zone, obstacle or the geofence; if not, it
  brakes. It flies the drone out of a zone that closed around it, and lands
  on a critical battery. When it acts it interrupts the skill and wakes the
  pilot with the reason.
