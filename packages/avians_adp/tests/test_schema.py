"""schema.py mirrors ADP 1.3.1; legacy records upgrade; the validator catches what upstream would."""
import json
import os
import sys
from pathlib import Path

import pytest
from avians_adp import (
    SCHEMA_VERSION,
    APIAction,
    ApiAction,
    MessageAction,
    TextObservation,
    Trajectory,
    decisions,
    to_upstream,
    validate,
)
from pydantic import ValidationError

FIXTURES = Path(__file__).parent / "fixtures"
DETAILS = {"system": "s", "tools": [{"name": "takeoff"}, {"name": "keep"}], "meta": {}, "score": {}, "summary": {}}


def prompt(n, t=0.0):
    return TextObservation(source="environment", content=f"P{n}", details={"t": t, "decision": n})


def action(n, fn="keep", cid=None, **det):
    return ApiAction(function=fn, kwargs={}, description="why", tool_call_id=cid,
                     details={"t": 0.0, "decision": n, "t_obs": 0.0, "t_apply": 1.0, **det})


def result(n, cid):
    return TextObservation(source="environment", content="ok", tool_call_id=cid,
                           details={"t": 1.0, "decision": n, "accepted": True})


def traj(*content, apis=("takeoff", "keep")):
    return Trajectory(id="ep", content=list(content), available_apis=list(apis), details=dict(DETAILS))


def test_legacy_record_upgrades_on_load():
    tj = Trajectory.model_validate_json((FIXTURES / "legacy_trajectory.json").read_text())
    assert tj.schema_version == SCHEMA_VERSION
    kinds = [s.class_ for s in tj.content]
    assert kinds[:3] == ["text_observation", "api_action", "text_observation"]
    assert all(s.source == "environment" for s in tj.content if s.class_ == "text_observation")
    assert tj.content[1].tool_call_id == tj.content[2].tool_call_id == "call_000001"
    assert tj.content[4].tool_call_id == "call_000002"
    assert tj.content[1].details["decision"] == 1 and tj.content[4].details["decision"] == 2
    assert validate(tj) == []
    assert [(d.n, d.action.function, d.finished) for d in decisions(tj)] == \
        [(1, "takeoff", True), (2, "search_region", True), (3, "inspect", True)]


def test_legacy_thinking_moves_to_reasoning_content():
    raw = {"id": "e", "content": [
        {"class_": "text_observation", "source": "environment", "content": "P", "details": {"decision": 1}},
        {"class_": "message_action", "content": "hi", "details": {"thinking": "hmm", "decision": 1}},
        {"class_": "api_action", "function": "keep", "kwargs": {}, "details": {}},
        {"class_": "text_observation", "source": "tool", "content": "kept", "details": {"accepted": True}}]}
    tj = Trajectory.model_validate(raw)
    assert tj.content[1].reasoning_content == "hmm" and "thinking" not in tj.content[1].details
    assert tj.content[2].details["decision"] == 1


def test_optional_fields_absent_from_json_when_empty():
    raw = json.loads(traj(prompt(1), action(1, cid="c1"), result(1, "c1")).model_dump_json())
    assert "schema_version" in raw and raw["schema_version"] == SCHEMA_VERSION
    assert "reasoning_content" not in raw["content"][1] and "reward" not in raw["content"][1]
    assert raw["content"][1]["tool_call_id"] == "c1" and "name" not in raw["content"][0]


def test_api_action_alias_and_discriminator():
    assert APIAction is ApiAction
    with pytest.raises(ValidationError):
        Trajectory.model_validate({"id": "e", "schema_version": SCHEMA_VERSION,
                                   "content": [{"class_": "code_action", "content": "x"}]})
    with pytest.raises(ValidationError):
        Trajectory(id="e", schema_version="9.9.9")


@pytest.mark.parametrize("content, msg", [
    ([prompt(1), action(1), result(1, "c1")], "no tool_call_id"),
    ([prompt(1), action(1, cid="c1")], "no matching observation"),
    ([prompt(1), action(1, cid="c1"), result(1, "c1"), prompt(2), action(2, cid="c1"), result(2, "c1")], "duplicate"),
    ([prompt(1), action(1, cid="c1"), result(1, "c1"), result(1, "c1")], "second result"),
    ([prompt(1), MessageAction(content="m", tool_call_id="c1"), action(1, cid="c2"), result(1, "c2")], "MessageAction"),
])
def test_upstream_link_rules_reject(content, msg):
    with pytest.raises(ValidationError, match=msg):
        traj(*content)


def test_tool_result_must_not_be_from_user():
    with pytest.raises(ValidationError, match="source='user'"):
        TextObservation(source="user", content="x", tool_call_id="c1")


def test_unfinished_decision_only_at_the_end():
    ok = traj(prompt(1), action(1, cid="c1"), result(1, "c1"), prompt(2), action(2))
    assert validate(ok) == []
    ok2 = traj(prompt(1), action(1, cid="c1"), result(1, "c1"), prompt(2))
    assert validate(ok2) == [] and not decisions(ok2)[-1].finished
    bad = traj(prompt(1), prompt(2), action(2, cid="c1"), result(2, "c1"))
    assert any("before the last one" in p or "before the previous decision" in p for p in validate(bad))


def test_validator_avians_rules():
    tj = traj(prompt(1), action(1, "takeoff", cid="c1"), result(1, "c1"))
    assert validate(tj) == []
    tj.available_apis = ["keep"]
    assert any("available_apis" in p for p in validate(tj))
    tj.available_apis = None
    tj.content[1].details["t_apply"] = -1.0
    assert any("t_obs" in p for p in validate(tj))
    tj.content[1].details["t_apply"] = 1.0
    tj.details.pop("score")
    assert any("'score'" in p for p in validate(tj))
    tj.details["score"] = {}
    tj.content[2].tool_call_id = None            # the model object is now inconsistent: a result that is a prompt
    assert any("tool_call_id" in p for p in validate(tj))
    out_of_order = traj(prompt(2), action(2, cid="c1"), result(2, "c1"), prompt(1), action(1, cid="c2"), result(1, "c2"))
    assert any("does not increase" in p for p in validate(out_of_order))


def test_decisions_groups_by_position_and_tolerates_junk():
    tj = traj(prompt(1), MessageAction(content="m", reasoning_content="r"), action(1, cid="c1"), result(1, "c1"),
              prompt(2), action(2, cid="c2"), result(2, "c2"), prompt(3))
    ds = decisions(tj)
    assert [d.n for d in ds] == [1, 2, 3] and ds[0].message.reasoning_content == "r"
    assert ds[0].accepted is True and ds[2].accepted is None and ds[2].action is None
    with pytest.raises(ValueError):
        decisions(traj(MessageAction(content="orphan"), prompt(1), action(1, cid="c1"), result(1, "c1")), strict=True)


def test_to_upstream_strips_the_avians_extension():
    up = to_upstream(traj(prompt(1), MessageAction(content="m"), action(1, cid="c1"), result(1, "c1")))
    assert all("details" not in s for s in up["content"]) and up["details"] == DETAILS
    assert up["content"][2] == {"class_": "api_action", "function": "keep", "kwargs": {}, "description": "why",
                                "tool_call_id": "c1"}


@pytest.mark.skipif(not os.environ.get("ADP_UPSTREAM"),
                    reason="set ADP_UPSTREAM to a checkout of neulab/agent-data-protocol@274f935a")
def test_against_real_upstream_models():
    sys.path.insert(0, os.environ["ADP_UPSTREAM"])
    from schema.trajectory import Trajectory as Upstream  # type: ignore
    from schema.version import SCHEMA_VERSION as UP_VERSION  # type: ignore
    assert UP_VERSION == SCHEMA_VERSION
    legacy = Trajectory.model_validate_json((FIXTURES / "legacy_trajectory.json").read_text())
    ours = traj(prompt(1), MessageAction(content="m", reasoning_content="r"), action(1, "takeoff", cid="c1"),
                result(1, "c1"), prompt(2), action(2))
    for tj in (legacy, ours):
        up = Upstream(**to_upstream(tj))
        assert len(up.content) == len(tj.content)
    broken = to_upstream(ours)
    broken["content"][2]["tool_call_id"] = None
    with pytest.raises(ValidationError):
        Upstream(**broken)
    for p in Path("data/episodes").rglob("trajectory.json"):      # every real episode, when present
        Upstream(**to_upstream(Trajectory.model_validate_json(p.read_text())))
