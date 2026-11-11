# avians_perception: detections in, a belief about the person out

- **`detector.py`:** the simulated detector for the fast tier. Detection
  probability falls with altitude and with distance from the image centre.
  Decoys are detected with low scores, false positives appear at a set rate,
  and position noise grows with altitude.
- **`tracker.py`:** `PersonTracker`. One Kalman filter per candidate
  (filterpy; Ornstein-Uhlenbeck velocity, so a walker's uncertainty grows
  honestly between sightings), and two log-likelihood ratios: existence
  (object vs clutter, from P_D and P_FA per hit or miss) and identity (person
  vs decoy, from the detector scores). conf is their product, so a tent seen
  every frame is a solid object that is not the person. Detections are gated
  (Mahalanobis and physical); weak boxes never move a credible track. Once
  found, the belief stays on the person's track (re-adopting a new track when
  they are re-acquired near it) and never jumps to a stray. The belief always
  exists: before any sighting it is the prior.
- **`projection.py`:** a pixel to a ground point (pinhole camera, drone pose,
  flat ground) for AAS's YOLO `Detection2DArray` boxes in the Gazebo tier.
- **`yolo.py`:** the Gazebo-tier bridge: subscribes to AAS's YOLO detections,
  projects person boxes (feet, not box centre) to the ground, feeds the same
  tracker. Enable with `perception: source: yolo` in `configs/aas.yaml`
  (untested).
- **`calibrate.py`:** fits the simulated camera's P_D(alt, dist) and score
  distributions to logged real detections by maximum likelihood.
- **Occlusion:** `detector.occluded` hides targets under a canopy or behind an
  obstacle's prism along the sight line.

**Calibration (to do on the Linux machine):** fly the Gazebo tier with
`perception: source: yolo`, then fit with `calibrate.fit_camera` on the
logged `detector_sample` events, and put the result in the camera model.
