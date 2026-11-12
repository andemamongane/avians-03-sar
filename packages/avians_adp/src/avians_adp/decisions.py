"""One decision = the steps the pilot loop writes for one decision point:

    prompt     TextObservation(source="environment"), no tool_call_id
    message    MessageAction, only when the pilot wrote text or thinking
    action     ApiAction (the skill call, or keep)
    result     TextObservation(source="environment") with the action's tool_call_id

Steps are grouped by position, not by their `decision` number, so records
from any tier group the same way. The last decision may be unfinished: the
episode can end after the prompt (no action) or after the action (no
result). The SFT, filter and router builders all read decisions, never raw
steps.
"""
from __future__ import annotations

from dataclasses import dataclass

from .schema import ApiAction, MessageAction, TextObservation, Trajectory


@dataclass
class Decision:
    index: int                          # position in the episode, from 0
    prompt: TextObservation
    message: MessageAction | None = None
    action: ApiAction | None = None
    result: TextObservation | None = None

    @property
    def n(self) -> int | None:
        """The loop's decision number (1-based), from the prompt."""
        return self.prompt.details.get("decision")

    @property
    def finished(self) -> bool:
        return self.action is not None and self.result is not None

    @property
    def accepted(self) -> bool | None:
        return None if self.result is None else self.result.details.get("accepted")


def decisions(traj: Trajectory, *, strict: bool = False) -> list[Decision]:
    """Group a trajectory's steps into decisions.

    With strict=False, a step that does not fit the pattern (a result before
    any prompt, two actions in one decision) is skipped. With strict=True it
    raises ValueError; validation.py uses that to report it.
    """
    out: list[Decision] = []
    cur: Decision | None = None

    def bad(i, why):
        if strict:
            raise ValueError(f"content[{i}]: {why}")

    for i, step in enumerate(traj.content):
        if isinstance(step, TextObservation):
            if step.tool_call_id is None:
                if step.source != "environment":
                    bad(i, f"observation with source {step.source!r} is neither a prompt nor a result")
                    continue
                if cur is not None and cur.action is None:
                    bad(i, "a new prompt before the previous decision got an action")
                cur = Decision(index=len(out), prompt=step)
                out.append(cur)
            else:
                if cur is None or cur.action is None:
                    bad(i, "a tool result with no action before it")
                    continue
                if cur.result is not None:
                    bad(i, "a second result in one decision")
                    continue
                cur.result = step
        elif isinstance(step, MessageAction):
            if cur is None or cur.action is not None:
                bad(i, "a message outside a decision's prompt-action window")
                continue
            if cur.message is not None:
                bad(i, "a second message in one decision")
                continue
            cur.message = step
        elif isinstance(step, ApiAction):
            if cur is None:
                bad(i, "an action before any prompt")
                continue
            if cur.action is not None:
                bad(i, "a second action in one decision")
                continue
            cur.action = step
    return out
