"""Agent Data Protocol (ADP) records, as pydantic models.

ADP (Song et al., "Agent Data Protocol", arXiv 2510.24702) is an interlingua
for agent training data: a Trajectory is an id, an alternating list of actions
and observations, and free-form details. Converters turn ADP into each
training harness's SFT format.

STARTER (owner: McGlashan). These mirror the published schema's core types.
TODO(McGlashan): check field names and the type discriminator against the
upstream repository's schema (it is the source of truth) and replace these
with imports from it if it is pip-installable; keep the avians extensions in
`details` so upstream converters keep working.

avians conventions inside the standard types:
    TextObservation.source   "environment" (the decision prompt), "tool"
                             (a skill's reply), "user" (the mission brief)
    APIAction.function       the skill name; kwargs its arguments;
                             description the pilot's one-line reason
    MessageAction            the pilot's free text (and thinking summary)
    *.details                t_obs, t_apply, latency_s, model, usage, router
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field


class _Step(BaseModel):
    details: dict[str, Any] = Field(default_factory=dict)   # avians extension


class TextObservation(_Step):
    class_: Literal["text_observation"] = "text_observation"
    source: str
    content: str


class APIAction(_Step):
    class_: Literal["api_action"] = "api_action"
    function: str
    kwargs: dict[str, Any] = Field(default_factory=dict)
    description: str | None = None


class MessageAction(_Step):
    class_: Literal["message_action"] = "message_action"
    content: str
    description: str | None = None


Step = Annotated[Union[TextObservation, APIAction, MessageAction], Field(discriminator="class_")]


class Trajectory(BaseModel):
    id: str
    content: list[Step] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)
