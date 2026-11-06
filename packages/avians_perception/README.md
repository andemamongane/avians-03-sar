# avians_perception: detections in, a belief about the person out

- **`detector.py`:** the simulated detector for the fast tier. Detection
  probability falls with altitude and with distance from the image centre.
  Decoys are detected with low scores, false positives appear at a set rate,
  and position noise grows with altitude.
- **`tracker.py`:** `PersonTracker`. One Kalman filter per candidate (filterpy,
  with mean-reverting velocity), plus a log-odds that it is the person.
  Detections are gated (Mahalanobis and physical), and weak boxes never move
  a credible track. Misses inside the footprint lower confidence. The belief
  (`est_east/north/sigma`, `conf`, `found`) always exists: before any
  sighting it is the prior from the world's region weights.
- **`projection.py`:** a pixel to a ground point (pinhole camera, drone pose,
  flat ground) for AAS's YOLO `Detection2DArray` boxes in the Gazebo tier.

**Calibration (to do):** fit `CameraModel`'s detection probabilities and
scores to YOLO's behaviour on Gazebo renders of the person model from 10–50 m,
so that both tiers agree.
