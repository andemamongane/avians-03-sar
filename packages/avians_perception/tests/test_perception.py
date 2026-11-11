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
    # the OU motion model lets the mean drift by about v * tau and the uncertainty grow honestly
    assert math.dist((b.east, b.north), (1.0, 2.0)) < 8.0 and 15 < b.sigma < 30


def test_prior_belief_exists_before_any_detection(world):
    b = PersonTracker.from_world(world).belief()
    assert b.source == "prior" and b.sigma > 30 and not b.found


def test_occlusion_by_a_canopy(world):
    from avians_perception.detector import occluded
    pines = world.obstacles["pines"]                 # 300..335 x 280..305, 28 m tall
    assert occluded((317, 292, 40), (317, 292), [pines])           # under the canopy
    assert occluded((317, 250, 20), (317, 320), [pines])           # behind it, sight line too low
    assert not occluded((317, 250, 200), (317, 330), [pines])      # high enough to see over
    assert not occluded((250, 250, 30), (260, 260), [pines])       # clear line


def test_identity_separates_a_decoy_from_the_person():
    cam = CameraModel(false_rate=0.0)
    det = SimulatedDetector(cam, seed=3)

    class Decoy:
        id, position, strength = "tent", (0.0, 0.0), 1.0

    tr = PersonTracker(prior=(0, 0, 100), camera=cam)
    for k in range(60):
        t = k * 0.25
        tr.update(t, det.frame(t, (0, 0, 20), (500, 500), [Decoy]), (0, 0, cam.footprint(20)), alt=20)
    t1 = tr.tracks[0]
    assert t1.llr > 5 and t1.idr < -5 and not tr.belief().found     # a solid object, not a person


def test_probability_map_moves_mass_away_from_searched_regions(world):
    from avians_skills.coverage import CoverageMap
    cam = CameraModel()
    cov = CoverageMap(world.search_regions(), camera=cam)
    prior = {"r1": 0.25, "r2": 0.25, "r3": 0.25, "r4": 0.25}
    r1 = world.regions["r1"]
    x0, y0, x1, y1 = r1.shape.bounds
    for e in range(int(x0), int(x1), 10):
        for n in range(int(y0), int(y1), 10):
            for _ in range(8):
                cov.mark(0, e, n, 20, cam.footprint(20))
    pm = cov.probability_map(prior)
    assert pm["r1"] < 0.05 and abs(sum(pm.values()) - 1) < 1e-9 and pm["r2"] > 0.3
