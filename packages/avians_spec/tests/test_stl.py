import math

import numpy as np
import pytest

from avians_spec import STLSyntaxError, Trace, parse, sar_registry, stl
from avians_spec.signals import Registry, Signal


def tr(dt=1.0, **sig):
    return Trace(dt, {k: np.asarray(v, float) for k, v in sig.items()})


def test_parse_round_trip_and_errors():
    reg = sar_registry()
    for text in ["G[0,10] battery >= 10", "(O released) & !help -> dist_person <= 30",
                 "F[0,5] found", "H[0,inf] !found", "G[0,900] ((!O found) -> F[0,600] found)"]:
        n = parse(text, reg)
        assert parse(str(n), reg) is not None
    with pytest.raises(STLSyntaxError, match="unknown signal"):
        parse("G[0,1] batery >= 3", reg)
    with pytest.raises(STLSyntaxError, match="not boolean"):
        parse("F[0,1] battery", reg)
    with pytest.raises(STLSyntaxError, match="bounded"):
        parse("G[0,inf] found", reg)
    assert parse("H found", reg).hi == math.inf


def brute(op, x, lo, hi, past=False):
    n, out = len(x), []
    for i in range(n):
        idx = [j for j in (range(i - hi, i - lo + 1) if past else range(i + lo, i + hi + 1)) if 0 <= j < n]
        vals = [x[j] for j in idx]
        out.append((min if op == "min" else max)(vals) if vals else (math.inf if op == "min" else -math.inf))
    return np.array(out)


@pytest.mark.parametrize("lo,hi", [(0, 0), (0, 3), (2, 5), (1, 17)])
def test_windows_match_brute_force(lo, hi):
    x = np.random.default_rng(lo * 31 + hi).normal(size=40)
    for how in ("min", "max"):
        assert np.allclose(stl._future(x, lo, hi, how), brute(how, x, lo, hi))
        assert np.allclose(stl._past(x, lo, hi, how), brute(how, x, lo, hi, past=True))


def test_past_operators():
    t = tr(found=[0, 0, 1, 0, 0])
    once = parse("O found")
    assert [stl._rho(once, t, None, 5)[k] > 0 for k in range(5)] == [False, False, True, True, True]
    hist = parse("H[0,1] !found")
    assert [stl._rho(hist, t, None, 5)[k] > 0 for k in range(5)] == [True, True, False, False, True]


def test_until():
    t = tr(a=[1, 1, 1, -1, -1], b=[-1, -1, 1, -1, -1])
    assert stl.robustness(parse("a U[0,4] b"), t) == pytest.approx(0.5)   # bare names: x - 0.5
    t2 = tr(a=[1, -1, 1, 1, 1], b=[-1, -1, 1, -1, -1])
    assert stl.robustness(parse("a U[0,4] b"), t2) < 0


def test_rosi_interval_contains_the_final_robustness():
    """Soundness: for any continuation within the registry's bounds, the
    finished flight's robustness lies in the interval computed from its prefix."""
    reg = Registry()
    reg.add(Signal("x", "", "", lo=0, hi=10, rate=1.0))
    reg.add(Signal("f", "", "", lo=0, hi=1, monotone="up", boolean=True))
    spec = parse("G[0,20] x >= 2 & F[0,30] f & G[0,30] (O f -> x <= 8)", reg)
    rng = np.random.default_rng(0)
    for trial in range(40):
        n = 31
        x = np.clip(5 + np.cumsum(rng.uniform(-1, 1, n)), 0, 10)
        k = int(rng.integers(1, n))
        f = (np.arange(n) >= rng.integers(0, 2 * n)).astype(float)
        full = tr(x=x, f=f)
        lo, hi = stl.interval(spec, tr(x=x[:k], f=f[:k]), registry=reg)
        r = stl.robustness(spec, full)
        assert lo - 1e-9 <= r <= hi + 1e-9, (trial, k, lo, r, hi)


def test_monitor_decides_violation_early(world):
    from avians_spec import MissionSpec, OnlineMonitor
    reg = sar_registry(world)
    spec = MissionSpec.from_dict({"dt": 0.5, "clauses": {
        "batt": "G[0,100] battery >= 10", "nf": "G[0,100] !in(nofly_power)"}}, reg, world.regions)
    mon = OnlineMonitor(spec, max_speed=15, every_s=1, policy="violated")
    for k in range(21):
        t = k * 0.5
        mon.add(t, {"battery": 50 - 5 * t, "east": 0, "north": 0, "alt": 30})
        mon.maybe_eval(t)
    assert mon.violated() == ["batt"] and mon.should_stop()
    assert spec.score(mon.trace())["clauses"][1]["satisfied"]
