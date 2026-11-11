from avians_sim import Executor
from avians_sim.multi import MultiSim
from avians_skills import SkillError


def test_two_drones_share_the_world_and_claims(world, limits):
    ms = MultiSim(world, 3, limits, n=2)
    ex = {d: Executor(s) for d, s in ms.sims.items()}
    for e in ex.values():
        e.start("takeoff", {})
    while not all(e.skill.done for e in ex.values()):
        ms.step({d: e.setpoint() for d, e in ex.items()})
    ok, text = ex["d1"].start("assign_tasks", {})
    assert ok and "d1:" in text and "d2:" in text
    r2 = [r for r, w in ms.shared.claims.items() if w == "d2"][0]
    ok, text = ex["d1"].start("search_region", {"region": r2})
    assert not ok and "being searched by d2" in text
    ok, _ = ex["d2"].start("search_region", {"region": r2})
    assert ok
    for _ in range(40):                      # the tasking reaches d2 after the link latency
        ms.step({d: e.setpoint() for d, e in ex.items()})
    msgs = [e for e in ms.sims["d2"].ctx.events if e.kind == "message"]
    assert msgs and "assigned you" in msgs[0].text
    sig = ms.signals()
    assert {"battery_d1", "battery_d2", "min_sep", "min_dist_person", "found"} <= set(sig)
    assert sig["min_sep"] > 10


def test_pads_are_exclusive(world, limits):
    ms = MultiSim(world, 0, limits, n=2)
    ex = {d: Executor(s) for d, s in ms.sims.items()}
    for s in ms.sims.values():
        s.vehicle.s.landed, s.vehicle.s.alt = False, 30
    assert ex["d1"].start("recharge", {"station": "pad_charge"})[0]
    ok, text = ex["d2"].start("recharge", {"station": "pad_charge"})
    assert not ok and "in use by d1" in text


def test_swarm_skills_refuse_with_one_drone(world, limits):
    from avians_sim import FastSim
    sim = FastSim(world, 0, limits)
    ex = Executor(sim)
    sim.vehicle.s.landed, sim.vehicle.s.alt = False, 30
    ok, text = ex.start("assign_tasks", {})
    assert not ok and "one drone" in text
