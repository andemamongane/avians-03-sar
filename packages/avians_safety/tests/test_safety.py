import pytest

from avians_safety import precheck, px4_params
from avians_sim import Executor, FastSim
from avians_skills import Setpoint, SkillError


def test_px4_params(limits):
    p = px4_params(limits)
    assert p["MPC_XY_VEL_MAX"] == pytest.approx(limits.max_speed * 1.2)
    assert p["BAT_CRIT_THR"] < 0.05           # avians' battery model is the one of record


def test_precheck_rejects_and_warns(world, limits):
    sim = FastSim(world, 0, limits)
    with pytest.raises(SkillError, match="take off"):
        precheck(sim.ctx, "search_region", {"region": "r1"})
    sim.vehicle.s.landed, sim.vehicle.s.alt = False, 30
    with pytest.raises(SkillError, match="nofly_power"):
        precheck(sim.ctx, "goto", {"east": 300, "north": 205})
    with pytest.raises(SkillError, match="envelope"):
        precheck(sim.ctx, "goto", {"east": 10, "north": 10, "alt": 200})
    sim.vehicle.s.battery = 14
    assert any("battery" in w for w in precheck(sim.ctx, "search_region", {"region": "r3"}))


def test_reflex_brakes_before_the_no_fly_zone(world, limits):
    sim = FastSim(world, 0, limits)
    ex = Executor(sim)
    v = sim.vehicle.s
    v.landed, v.east, v.north, v.alt = False, 300, 160, 30
    reasons = []
    for _ in range(300):                      # a raw setpoint straight through the power line
        sp, why = sim.reflex.check(sim.ctx, Setpoint("position", (300, 260, 30), (0, 9, 0), 12))
        reasons += [why] if why else []
        sim.step(sp)
    assert reasons and "nofly_power" in reasons[0]
    assert not world.regions["nofly_power"].contains(v.east, v.north, v.alt)
    del ex
