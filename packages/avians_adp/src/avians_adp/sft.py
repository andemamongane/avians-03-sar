"""ADP -> training examples. TODO(McGlashan): everything below is the plan.

Two datasets come out of the same trajectories:

1. Pilot SFT (behaviour cloning the teacher). One example per decision:
       input  = system prompt + tools + the decision prompt at that point
                (it is self-contained: window + structured memory, storage
                option C, so no earlier turns are needed)
       target = the teacher's tool call (+ its reason), optionally its
                thinking summary
   Output formats: chat-messages JSONL for LLaMA-Factory / TRL (tool calls in
   the OpenAI/Hermes function-calling layout the student's chat template
   expects).

2. Router examples (System One: commit / local / escalate). One example per
   decision point, labelled by what happened (e.g. student and teacher agree
   -> local is enough; they disagree and the teacher's choice scored better
   -> escalate; nothing changed and the pilot kept its skill -> commit).
   Needs counterfactual student calls on the same prompts (run the student
   offline over the teacher's prompts).

Also: filters (by episode robustness, by clause), de-duplication of
near-identical decision prompts, train/val/test splits by seed (never split
one episode across sets), and a manifest (counts, sources, git commit).
"""
from __future__ import annotations

from .schema import Trajectory


def pilot_examples(traj: Trajectory) -> list[dict]:
    raise NotImplementedError("McGlashan: pilot SFT examples from a trajectory")


def router_examples(teacher: Trajectory, student_calls: list[dict]) -> list[dict]:
    raise NotImplementedError("McGlashan: router labels from teacher/student comparisons")
