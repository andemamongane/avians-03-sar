"""Talking to people: report to the rescue crew, ask the operator, escalate."""
from __future__ import annotations

from ..base import Skill, SkillError

NUM = {"type": "number"}


class Report(Skill):
    name, category, instant = "report", "comms", True
    summary = ("Radio a report to the ground station. kind person_found dispatches the rescue crew "
               "to the given position (default: the current estimate): a wrong position delays "
               "them. The acknowledgement (and the crew's ETA) comes back as a message.")
    params = {"kind": {"type": "string", "enum": ["person_found", "status", "hazard"]},
              "message": {"type": "string", "default": ""},
              "east": NUM, "north": NUM}
    required = ["kind"]

    def start(self, ctx):
        a = self.args
        e, n = a.get("east"), a.get("north")
        if a["kind"] == "person_found" and (e is None or n is None):
            b = ctx.belief() if ctx.belief else None
            if b is None or b.source == "prior":
                raise SkillError("no person estimate to report: give east and north")
            e, n = b.east, b.north
        reply = ctx.actuators.report(a["kind"], e, n, a["message"])
        self.succeed(ctx, reply)


class RequestHelp(Skill):
    name, category, instant = "request_help", "comms", True
    summary = ("Ask the operator at the ground station a question; the answer, if any, arrives "
               "later as a message (they may be slow, or not answer at all).")
    params = {"question": {"type": "string"}}
    required = ["question"]

    def start(self, ctx):
        self.succeed(ctx, ctx.actuators.ask_operator(self.args["question"]))


class Escalate(Skill):
    name, category, instant = "escalate", "comms", True
    summary = ("Hand this decision to the larger remote model (onboard pilots only): use when "
               "the situation is unfamiliar or the stakes are high.")
    params = {"reason": {"type": "string"}}
    required = ["reason"]

    def start(self, ctx):
        ctx.emit("escalate", self.args["reason"])
        self.succeed(ctx, "escalated")
