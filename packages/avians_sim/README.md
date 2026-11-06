# avians_sim: simulation tiers behind one Vehicle interface

| Tier | Module | Speed | Use |
|---|---|---|---|
| fast | `fast.py`: `FastSim`, `FastVehicle` | ~300× real time with the pilot loop | rollouts, data, RL, tests |
| Gazebo | `aas.py`: `AASVehicle`, `GazeboSim` (PX4 through the aerial-autonomy-stack) | real time | demos, sim-to-real checks, video (**untested**) |

- **`executor.py`:** runs one skill at a time: the pre-check, the skill's
  setpoints, then the reflex layer, then the vehicle. It also collects
  events.
- **`gazebo.py`:** `GazeboControl` (ported from avians-02). It draws the world
  into AAS's Gazebo, sets the PX4 parameters, plays the wind, records the
  chase and overview videos, copies `px4.ulg`, and replays any recorded
  episode as a visual-only drone.
- **`simside/avians_gz.py`:** the helper that runs inside AAS's simulation
  container (Gazebo transport only).

What the fast tier models:

- **Vehicle:** PX4's position-controller structure (P on position plus
  velocity feed-forward, speed, acceleration and climb limits).
- **Ground:** take-off and landing.
- **Battery:** the avians battery model.
- **Camera:** the simulated detector at 4 Hz, feeding the tracker.
- **Coverage:** the search grid.
- **Kits:** they fall with drag and drift with the wind.
- **The world's timeline:** zone closures and help.
