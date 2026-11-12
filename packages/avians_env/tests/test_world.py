import numpy as np
import pytest
from shapely.geometry import Point

from avians_env import WorldError, WorldSpec, WorldState


def test_the_valley_is_tiled(world):
    assert world.tiled and world.tiling_errors() == []
    assert len(world.regions) == 17 and {r.id for r in world.search_regions()} == {"r1", "r2", "r3", "r4"}
    # every point of the area is in exactly one zone
    rng = np.random.default_rng(0)
    for _ in range(300):
        e, n = rng.uniform(-79, 519), rng.uniform(-79, 479)
        inside = [r.id for r in world.regions.values() if r.shape.contains(Point(e, n))]
        assert len(inside) <= 1 and world.zone_at(e, n) is not None
    assert world.zone_at(0, 0).id == "home_base" and world.regions["home_base"].protected
    text = world.describe()
    assert "r3 (search)" in text and "pad_swap (swap)" in text


def test_tiling_errors_are_caught():
    base = {"name": "w", "tiled": True, "bounds": [[0, 0], [10, 0], [10, 10], [0, 10]],
            "regions": [{"id": "a", "kind": "other", "polygon": [[0, 0], [6, 0], [6, 10], [0, 10]]},
                        {"id": "b", "kind": "other", "polygon": [[5, 0], [10, 0], [10, 10], [5, 10]]}]}
    with pytest.raises(WorldError, match="overlap"):
        WorldSpec.from_dict(base)
    base["regions"][1]["polygon"] = [[7, 0], [10, 0], [10, 10], [7, 10]]
    with pytest.raises(WorldError, match="cover"):
        WorldSpec.from_dict(base)


def test_region_robustness_is_signed_distance(world):
    r = world.regions["r1"]
    for e, n in ((130, 130), (60, 60), (300, 300)):
        rho = float(r.robustness(np.array([e]), np.array([n]), np.array([30]))[0])
        d = r.shape.exterior.distance(Point(e, n))
        assert rho == pytest.approx(d if r.shape.contains(Point(e, n)) else -d)


def test_scenarios_are_reproducible_and_inside_search_regions(world):
    a, b = world.sample(5), world.sample(5)
    assert a.person_position(0) == b.person_position(0) and a.timeline == b.timeline
    for seed in range(30):
        p = world.sample(seed).person_position(0)
        assert any(r.contains(*p) for r in world.search_regions())


def test_storms_and_fires_close_zones_per_seed(world):
    draws = [world.sample(s, 1600) for s in range(30)]
    assert sum(len(d.storms) for d in draws) > 10 and sum(len(d.fires) > 0 for d in draws) > 5
    assert any(not d.timeline for d in draws)                     # some calm days
    for d in draws:
        for e in d.timeline:
            z = world.regions[e.zone]
            assert z.kind != "no_fly" and not z.protected          # base and no-fly never close
        for s in d.storms:                                         # a storm closes what it covers
            t = (s.t0 + s.t1) / 2
            st = WorldState(d)
            c = Point(s.center(t))
            hit = [z for z in world.dynamic_zones() if z.shape.distance(c) < s.radius - 3]
            for z in hit:
                assert not st.zone_safe(z.id, t), (s.id, z.id)
        for f in d.fires:
            assert not WorldState(d).zone_safe(f["zone"], f["t"] + 1)


def test_status_lists_closures_and_cautions(world):
    sc = next(world.sample(s, 1600) for s in range(50) if world.sample(s, 1600).storms)
    st = WorldState(sc)
    s = sc.storms[0]
    t = (s.t0 + s.t1) / 2
    text = st.status(t)
    closed = [z.id for z in st.unsafe_zones(t)]
    assert all(f"{z} (" in text for z in closed) and ("CLOSED" in text or not closed)
    assert st.signals(t)[f"unsafe_{closed[0]}"] == 1.0 if closed else True
    assert set(st.cautions(t)).isdisjoint(closed)


def test_help_timeline(world):
    s = WorldState(world.sample(1))
    pe, pn = s.person_position(100)
    s.report_found(100, pe + 50, pn)                # wrong by 50 m: the crew searches longer
    assert s.help.arrival_t == pytest.approx(100 + s.scenario.help_delay_s + world.help.search_penalty_s)
    assert not s.help_arrived(s.help.arrival_t - 1) and s.help_arrived(s.help.arrival_t)


def test_rejects_bad_worlds():
    base = {"name": "w", "regions": [{"id": "a", "kind": "search", "polygon": [[0, 0], [10, 0], [10, 10]]}]}
    WorldSpec.from_dict(base)
    with pytest.raises(WorldError, match="kind"):
        WorldSpec.from_dict({**base, "regions": [{**base["regions"][0], "kind": "lake"}]})
    with pytest.raises(WorldError, match="unknown zone"):
        WorldSpec.from_dict({**base, "dynamics": {"zone_safety": [{"t": 1, "zone": "zz", "safe": False}]}})


def test_gazebo_models_parse(world):
    import xml.etree.ElementTree as ET
    from avians_env.gazebo import all_models
    models = all_models(world, (100, 100), anchor=(2, -1))
    assert "avians_person" in models and "avians_obstacle_tower" in models
    for sdf in models.values():
        ET.fromstring(sdf.split("?>", 1)[1])
