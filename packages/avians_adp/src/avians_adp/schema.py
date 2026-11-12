"""Agent Data Protocol (ADP) records, as pydantic models.

ADP (Song et al., "Agent Data Protocol", arXiv 2510.24702) is an interlingua
for agent training data: a Trajectory is an id, an alternating list of actions
and observations, and free-form details. Converters turn ADP into each
training harness's SFT format.

These classes mirror the standardized ADP schema, version 1.3.1, as published
at neulab/agent-data-protocol@274f935a (UPSTREAM below). Upstream is not
imported: it is not on PyPI, needs Python >= 3.12, and installs top-level
packages named `schema`, `agents` and `scripts`. After that commit upstream
switched to Harbor's ATIF (`schema/atif.py`); an ATIF export can be added
later without touching the record.

Differences from upstream, all additive:
    *.details        avians' per-step extension (timing, model, usage, state).
                     Upstream forbids extra fields, so `to_upstream()` in
                     validation.py strips it before an upstream check.
    class_           a pydantic discriminator here; upstream validates the
                     value by hand. The JSON is the same.
    APIAction        alias of ApiAction, for existing imports.

avians conventions inside the standard types:
    TextObservation.source   "environment": the decision prompt, or (with a
                             tool_call_id) a skill's reply; "user": a brief
    ApiAction.function       the skill name; kwargs its arguments;
                             description the pilot's one-line reason
    MessageAction            the pilot's free text; reasoning_content its
                             thinking summary
    tool_call_id             links each ApiAction to the observation that
                             answers it (call_000001, ...)

Records written before this schema (no `schema_version`) are upgraded on
load: see Trajectory._upgrade_legacy.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator, model_serializer, model_validator

UPSTREAM = "neulab/agent-data-protocol@274f935a"
SCHEMA_VERSION = "1.3.1"
SUPPORTED_SCHEMA_VERSIONS = (SCHEMA_VERSION, "1.3.0", "1.2.0", "1.1.0", "1.0.0")

# Upstream marks these `exclude_if=None`; that needs a newer pydantic than our
# pin, so a serializer drops them instead.
_OPTIONAL = ("tool_call_id", "reasoning_content", "reward", "name", "available_apis")


class _Base(BaseModel):
    @model_serializer(mode="wrap")
    def _drop_empty_optionals(self, handler):
        d = handler(self)
        for k in _OPTIONAL:
            if k in d and d[k] is None:
                del d[k]
        return d


class _Step(_Base):
    details: dict[str, Any] = Field(default_factory=dict)   # avians extension


class Action(_Step):
    tool_call_id: str | None = None
    reasoning_content: str | None = None
    reward: float | None = None


class Observation(_Step):
    tool_call_id: str | None = None
    reward: float | None = None


class ApiAction(Action):
    class_: Literal["api_action"] = "api_action"
    function: str
    kwargs: dict[str, Any]
    description: str | None = None


APIAction = ApiAction


class MessageAction(Action):
    class_: Literal["message_action"] = "message_action"
    content: str
    description: str | None = None


class TextObservation(Observation):
    class_: Literal["text_observation"] = "text_observation"
    content: str
    name: str | None = None
    source: Literal["user", "agent", "environment"]

    @model_validator(mode="after")
    def _tool_result_source(self):
        if self.tool_call_id is not None and self.source == "user":
            raise ValueError("a TextObservation with tool_call_id is a tool result and must not use source='user'")
        return self


Step = Annotated[ApiAction | MessageAction | TextObservation, Field(discriminator="class_")]


def is_tool_action(step) -> bool:
    return isinstance(step, ApiAction)


class Trajectory(_Base):
    schema_version: str = SCHEMA_VERSION
    id: str
    content: list[Step] = Field(default_factory=list)
    available_apis: list[str] | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("schema_version")
    @classmethod
    def _supported_version(cls, v):
        if v not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(f"unsupported schema_version {v!r}; supported: {SUPPORTED_SCHEMA_VERSIONS}")
        return v

    @model_validator(mode="before")
    @classmethod
    def _upgrade_legacy(cls, data):
        """Records from before schema_version: source 'tool', no call ids,
        thinking in details, no decision number on the action."""
        if not isinstance(data, dict) or "schema_version" in data or not isinstance(data.get("content"), list):
            return data
        content, n_call, decision = [], 0, None
        prev: dict | None = None
        for raw in data["content"]:
            s = dict(raw) if isinstance(raw, dict) else raw
            if isinstance(s, dict):
                det = dict(s.get("details") or {})
                kind = s.get("class_")
                if kind == "text_observation":
                    if s.get("source") == "tool":
                        s["source"] = "environment"
                        if prev is not None and prev.get("class_") == "api_action" and not prev.get("tool_call_id"):
                            n_call += 1
                            prev["tool_call_id"] = s["tool_call_id"] = f"call_{n_call:06d}"
                    elif s.get("source") == "environment" and "decision" in det:
                        decision = det["decision"]
                if kind in ("api_action", "message_action"):
                    think = det.pop("thinking", None)
                    if think and not s.get("reasoning_content"):
                        s["reasoning_content"] = think
                    if kind == "api_action" and "decision" not in det and decision is not None:
                        det["decision"] = decision
                s["details"] = det
            content.append(s)
            prev = s if isinstance(s, dict) else None
        return {**data, "content": content, "schema_version": SCHEMA_VERSION}

    @model_validator(mode="after")
    def _tool_call_links(self):
        """Upstream's rule: a tool action followed by an observation carries an
        id; each id matches exactly one later observation, and every id is
        matched."""
        actions: dict[str, int] = {}
        matched: dict[str, int] = {}
        for i, item in enumerate(self.content):
            nxt = self.content[i + 1] if i + 1 < len(self.content) else None
            tid = item.tool_call_id
            if isinstance(item, ApiAction) and isinstance(nxt, Observation) and tid is None:
                raise ValueError(f"content[{i}]: tool action followed by an observation but no tool_call_id")
            if tid is None:
                continue
            if isinstance(item, MessageAction):
                raise ValueError(f"content[{i}]: MessageAction is not a tool call and must not carry tool_call_id")
            if isinstance(item, ApiAction):
                if tid in actions:
                    raise ValueError(f"content[{i}]: duplicate tool_call_id {tid!r} (also at {actions[tid]})")
                actions[tid] = i
            elif isinstance(item, Observation):
                if tid not in actions:
                    raise ValueError(f"content[{i}]: observation tool_call_id {tid!r} matches no earlier action")
                if tid in matched:
                    raise ValueError(f"content[{i}]: second result for tool_call_id {tid!r} (also at {matched[tid]})")
                matched[tid] = i
        for tid, i in actions.items():
            if tid not in matched:
                raise ValueError(f"content[{i}]: tool_call_id {tid!r} has no matching observation")
        return self
