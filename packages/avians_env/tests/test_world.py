import numpy as np
import pytest

from avians_env import WorldError, WorldSpec, WorldState


def test_loads_the_benchmark_world(world):
    assert {r.id for r in world.search_regions()} == {"r1", "r2", "r3", "r4"}
    assert [z.id for z in world.hazard_zones()] == ["zone_fire"]
    assert world.stations["pad_charge"].kind == "charge" and world.depots["base"].supplies == 1
    text = world.describe()
    assert "r3 (search)" in text and "pad_swap (swap)" in text


def test_region_robustness_is_signed_distance(world):
    r = world.regions["r1"]                        # 40..200 east, 60..180 north
    rho = r.robustness(np.array([120, 120, 0]), np.array([120, 65, 120]), np.array([30, 30, 30]))
    assert rho[0] == pytest.approx(60) and rho[1] == pytest.approx(5) and rho[2] == pytest.approx(-40)


def test_scenarios_are_reproducible_and_inside_search_regions(world):
    a, b = world.sample(5), world.sample(5)
    assert a.person_position(0) == b.person_position(0) and a.help_delay_s == b.help_delay_s
    for seed in range(30):
        p = world.sample(seed).person_position(0)
        assert any(r.contains(*p) for r in world.search_regions())
        assert not world.regions["zone_fire"].contains(*p)


def test_zone_safety_and_help_timeline(world):
    s = WorldState(world.sample(1))
    assert s.zone_safe("zone_fire", 419) and not s.zone_safe("zone_fire", 420)
    assert [e.zone for e in s.changes(400, 430)] == ["zone_fire"]
    assert s.signals(500)["unsafe_zone_fire"] == 1.0
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
