import math

import pytest

from avians_perception import CameraModel, PersonTracker, SimulatedDetector, ground_point


def test_ground_projection():
    assert ground_point(320, 240, 640, 480, 1.2, (10, 20, 30), yaw=0.0) == pytest.approx((10, 20), abs=1e-6)
    # oblique camera looking east, pitched 45 degrees down: the centre ray lands 30 m east
    e, n = ground_point(320, 240, 640, 480, 1.2, (0, 0, 30), yaw=0.0, pitch=math.pi / 4)
    assert e == pytest.approx(30) and n == pytest.approx(0, abs=1e-6)
    assert ground_point(320, 0, 640, 480, 1.2, (0, 0, 30), yaw=0.0, pitch=0.1) is None   # above the horizon


def test_tracker_finds_a_static_person_and_doubts_a_decoy():
    cam = CameraModel(false_rate=0.0)
    det = SimulatedDetector(cam, seed=1)

    class Decoy:
        id, position, strength = "tent", (40.0, 0.0), 1.0

    tr = PersonTracker(prior=(0, 0, 100))
    for k in range(40):                       # hover over the decoy for 10 s
        t = k * 0.25
        tr.update(t, det.frame(t, (40, 0, 20), (500, 500), [Decoy]), (40, 0, cam.footprint(20)))
    assert not tr.belief().found
    for k in range(40, 80):                   # then over the person
        t = k * 0.25
        tr.update(t, det.frame(t, (0, 0, 20), (1.0, 2.0), [Decoy]), (0, 0, cam.footprint(20)))
    b = tr.belief()
    assert b.found and math.dist((b.east, b.north), (1.0, 2.0)) < 2.0 and b.sigma < 2.0
    for k in range(80, 80 + 4 * 300):         # 5 minutes without seeing them: stays put
        tr.update(k * 0.25, [], None)
    b = tr.belief()
    assert math.dist((b.east, b.north), (1.0, 2.0)) < 3.0 and b.sigma < 25


def test_prior_belief_exists_before_any_detection(world):
    b = PersonTracker.from_world(world).belief()
    assert b.source == "prior" and b.sigma > 30 and not b.found
