# avians_viz: look at episodes

- **`replay2d.py`:** the sped-up 2D replay (`replay.mp4`). The map shows
  regions, closed zones (hatched once closed), obstacles and pads, plus the
  drone's trail, the true person, the estimate with its 1-sigma circle, and
  landed kits. Beside it are battery and distance-to-person timelines, and
  the decision being executed with its reason. `summary_png` draws a static
  overview. ffmpeg comes from PATH, or from imageio-ffmpeg.
- **`qgc.py`:** QGroundControl exports. `world_kml` (Plan view → import),
  `flight_kml`, and `tlog`: a MAVLink v2 telemetry log of the flight
  (heartbeat, global position, attitude, battery) that QGC's Log Replay
  plays like a live flight.
- **`showcase.py`:** the 1080p sponsor film (`avians showcase`): terrain,
  search heat map, drones and footprints, zone closures, kit drop, the
  helicopter, the pilots' captions, the STL promises as live chips; variable
  speed around key moments, title and results cards.
- **`episode.py`:** reads an episode folder back (any number of drones).

Gazebo replays live in `avians_sim.gazebo` (`avians gazebo-replay`), because
they need the simulator.
