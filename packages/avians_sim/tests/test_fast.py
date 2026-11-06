import math

from avians_sim import Executor, FastSim


def run(ex, sim, until, max_t=600):
    while sim.t < max_t:
        ex.tick()
        if until():
            return True
    return False


def test_takeoff_goto_land(world, limits):
    sim = FastSim(world, 0, limits)
    ex = Executor(sim)
    assert ex.start("takeoff", {"alt": 20})[0]
    assert run(ex, sim, lambda: ex.skill.done)
    assert abs(sim.vehicle.s.alt - 20) < 1
    ok, text = ex.start("goto", {"place": "pad_swap"})
    assert ok and "about" in text
    assert run(ex, sim, lambda: ex.skill.done) and ex.skill.status == "succeeded"
    assert math.dist((sim.vehicle.s.east, sim.vehicle.s.north), world.stations["pad_swap"].position) < 2
    v = sim.vehicle.s
    assert v.battery < 100 and v.speed <= limits.max_speed + 1e-6
    ex.start("land", {})
    assert run(ex, sim, lambda: sim.vehicle.s.landed)


def test_search_finds_and_kit_lands_near(world, limits):
    seed = next(s for s in range(50) if world.regions["r1"].contains(*world.sample(s).person_position(0)))
    sim = FastSim(world, seed, limits)
    ex = Executor(sim)
    ex.start("takeoff", {})
    run(ex, sim, lambda: ex.skill.done)
    ex.start("search_region", {"region": "r1"})
    assert run(ex, sim, lambda: sim.tracker.belief().found)
    assert ex.start("report", {"kind": "person_found"})[0]
    assert sim.world_state.help.dispatched_t is not None
    ex.start("pick_up", {})
    assert run(ex, sim, lambda: ex.skill.done, 900) and sim.vehicle.s.payload
    ex.start("deliver", {})
    assert run(ex, sim, lambda: ex.skill.done, 1200)
    run(ex, sim, lambda: any(k["landed"] for k in sim.kits), 1300)
    assert sim.signals()["drop_err"] < 5


def test_rejections_explain(world, limits):
    sim = FastSim(world, 0, limits)
    ex = Executor(sim)
    ok, text = ex.start("deliver", {})
    assert not ok and "take off" in text
    ok, text = ex.start("recharge", {"station": "pad_swap"})
    assert not ok and "not a charge station" in text
