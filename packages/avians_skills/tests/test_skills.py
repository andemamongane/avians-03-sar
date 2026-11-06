import math

import pytest
from shapely.geometry import LineString, Polygon

from avians_skills import BENCHMARK_TOOLS, PathPlanner, SkillError, create, tools
from avians_skills.coverage import CoverageMap, expanding_square, lawnmower
from avians_skills.follow import PathFollower
from avians_skills.library.targets import intercept_point


def test_planner_goes_around_the_power_line(world):
    p = PathPlanner(world, margin_m=6)
    path = p.plan((300, 100), (380, 300), 30)
    assert len(path) > 2
    line = LineString(path)
    assert not line.intersects(world.regions["nofly_power"].shape.buffer(5.9))
    assert p.length(path) < 2.5 * 200


def test_planner_respects_closed_zones_and_rejects_forbidden_goals(world):
    p = PathPlanner(world)
    z = world.regions["zone_fire"]
    with pytest.raises(SkillError, match="zone_fire"):
        p.plan((100, 100), (0, 380), 30, unsafe=[z])
    path = p.plan((20, 250), (80, 470), 30, unsafe=[z])
    assert not LineString(path).intersects(z.shape)
    with pytest.raises(SkillError, match="tower"):
        p.plan((0, 0), (220, 110), 30)


def test_lawnmower_covers_the_polygon():
    poly = Polygon([(0, 0), (100, 0), (100, 60), (0, 60)])
    pts = lawnmower(poly, 20, (0, 0))
    assert math.dist(pts[0], (0, 0)) < 25
    cov = CoverageMap.__new__(CoverageMap)

    class R:
        id, shape = "a", poly
    cov.__init__([R])
    f = PathFollower([(*p, 30) for p in pts], 8, 4)
    for s in range(0, int(f.length), 2):
        e, n, _ = f.point(s)
        cov.mark(0, e, n, 30, 12)
    assert cov.fraction("a") > 0.95


def test_expanding_square_grows():
    pts = expanding_square((0, 0), 10, 40)
    assert pts[0] == (0, 0) and max(math.hypot(*p) for p in pts) >= 40


def test_intercept_point():
    (e, n), t = intercept_point((0, 0), (100, 0), (0, 1), 10)
    assert math.hypot(e, n) == pytest.approx(10 * t) and n == pytest.approx(t)


def test_create_validates_arguments():
    s = create("search_region", {"region": "r3"})
    assert s.args["alt"] == 30.0 and s.args["pattern"] == "lawnmower"
    with pytest.raises(SkillError, match="needs"):
        create("recharge", {})
    with pytest.raises(SkillError, match="one of"):
        create("search_region", {"region": "r1", "pattern": "spiral"})
    with pytest.raises(SkillError, match="no parameter"):
        create("goto", {"x": 1})


def test_tool_schemas():
    ts = {t["name"]: t for t in tools(BENCHMARK_TOOLS)}
    assert "escalate" not in ts and "handoff" not in ts and "search_region" in ts
    for t in ts.values():
        sch = t["input_schema"]
        assert sch["type"] == "object" and sch["additionalProperties"] is False
        assert set(sch["required"]) <= set(sch["properties"])
