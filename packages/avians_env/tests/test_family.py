from avians_env import WorldSpec, WorldState

from conftest import ROOT


def test_family_varies_by_seed_and_is_reproducible():
    w = WorldSpec.load(ROOT / "worlds/sar_valley_family")
    assert w.family is not None and w.name == "sar_valley_family"
    assert set(w.regions) == set(WorldSpec.load(ROOT / "worlds/sar_valley").regions)
    a, b = w.sample(1), w.sample(1)
    assert a.summary() == b.summary()
    draws = [w.sample(s) for s in range(20)]
    assert len({d.descriptions["r2"] for d in draws}) > 1
    assert len({(e.t, e.safe) for d in draws for e in d.timeline}) > 2
    assert any(d.motion == "random_walk" for d in draws) and any(d.motion == "static" for d in draws)
    assert any(not d.timeline for d in draws)            # p_close 0.8: some seeds never close
    assert any(d.news for d in draws)
    for d in draws:
        assert abs(sum(d.region_weights.values()) - 1) < 1e-6
        assert d.person_region in w.regions and w.regions[d.person_region].contains(*d.person_position(0))


def test_news_updates_descriptions_and_status():
    w = WorldSpec.load(ROOT / "worlds/sar_valley_family")
    sc = next(w.sample(s) for s in range(50) if any(n.description for n in w.sample(s).news))
    st = WorldState(sc)
    n = next(n for n in sc.news if n.description)
    before = st.describe()
    assert st.news(n.t - 1, n.t) and n.text in st.status(n.t)
    assert st.describe() != before and "phone ping" in st.describe() or "red jacket" in st.describe()


def test_fixed_world_has_one_member():
    w = WorldSpec.load(ROOT / "worlds/sar_valley")
    a, b = w.sample(1), w.sample(2)
    assert a.descriptions == b.descriptions and a.timeline == b.timeline and a.wind == b.wind
    assert a.person_position(0) != b.person_position(0)
