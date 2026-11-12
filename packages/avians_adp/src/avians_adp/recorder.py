"""Writing an episode: the ADP trajectory plus the signals and events.

The pilot loop (avians_agent.loop) calls, in order:

    rec = EpisodeRecorder(out_dir, episode_id, meta)
    rec.brief(system_prompt, tools)             once
    rec.observation(text, t, **details)         each decision point: the prompt
    rec.message(text, t, **details)             the pilot's text / thinking summary
    rec.action(function, kwargs, reason, t, **details)
    rec.tool_result(text, t, **details)         what the skill said back
    rec.signals(t, dict)                        every simulation step (thinned)
    rec.event(t, kind, text)                    every event
    rec.finish(score, summary)

Files in out_dir:
    trajectory.json   the ADP Trajectory (the full growing record; storage
                      option A: everything, so any window can be cut later)
    signals.jsonl     t + every registry signal, at signals_hz
    events.jsonl      every event, waking or not
    meta.json         mission, world, seed, pilot, tier, versions
    score.json        the final STL score per clause

This is a cross-package contract (avians_agent writes through it, avians_viz
and `avians latency-fit` read trajectory.json). The method signatures are
the contract; the file follows ADP 1.3.1 (schema.py):
    - each action gets a tool_call_id (call_000001, ...) shared with the
      observation that answers it; a skill's reply is source "environment"
    - the thinking summary is the action's reasoning_content
    - the action carries the decision number of its prompt
    - available_apis lists the tool names from brief()
    - each decision prompt carries `events`: the kinds of the events
      recorded since the previous prompt, in order. The loop calls event()
      exactly when it adds to its own `since` list and observation() when it
      empties it, so this is the event list the router was given; a time
      window over events.jsonl cannot reproduce it (an event stamped at a
      decision's time may belong to either side).
finish() always writes the files first, then validates and warns: a flight's
record is never lost to a validation bug.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

from .schema import ApiAction, MessageAction, TextObservation, Trajectory


class EpisodeRecorder:
    def __init__(self, out_dir: str | Path, episode_id: str, meta: dict, signals_hz: float = 2.0):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.traj = Trajectory(id=episode_id, details={"meta": meta})
        self.meta = meta
        self._sig = open(self.dir / "signals.jsonl", "w")
        self._ev = open(self.dir / "events.jsonl", "w")
        self._sig_dt = 1.0 / signals_hz
        self._next_sig = 0.0
        self._n_calls = 0
        self._open_call: str | None = None     # the action still waiting for its result
        self._decision: int | None = None      # the latest prompt's decision number
        self._since: list[str] = []            # event kinds since the latest prompt
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    def brief(self, system: str, tools: list[dict]) -> None:
        self.traj.details["system"] = system
        self.traj.details["tools"] = tools
        self.traj.available_apis = [t["name"] for t in tools]

    def observation(self, text: str, t: float, source: str = "environment", **details) -> None:
        if source == "environment" and "decision" in details:
            self._decision = details["decision"]
            details.setdefault("events", self._since)
            self._since = []
        self.traj.content.append(TextObservation(source=source, content=text, details={"t": t, **details}))

    def message(self, text: str, t: float, **details) -> None:
        thinking = details.pop("thinking", None) or None
        if "decision" not in details and self._decision is not None:
            details["decision"] = self._decision
        self.traj.content.append(MessageAction(content=text, reasoning_content=thinking, details={"t": t, **details}))

    def action(self, function: str, kwargs: dict, reason: str | None, t: float, **details) -> None:
        thinking = details.pop("thinking", None) or None
        if "decision" not in details and self._decision is not None:
            details["decision"] = self._decision
        self._n_calls += 1
        self._open_call = f"call_{self._n_calls:06d}"
        self.traj.content.append(ApiAction(function=function, kwargs=dict(kwargs or {}), description=reason,
                                           tool_call_id=self._open_call, reasoning_content=thinking,
                                           details={"t": t, **details}))

    def tool_result(self, text: str, t: float, **details) -> None:
        if "decision" not in details and self._decision is not None:
            details["decision"] = self._decision
        self.traj.content.append(TextObservation(source="environment", content=text, tool_call_id=self._open_call,
                                                 details={"t": t, **details}))
        self._open_call = None

    def signals(self, t: float, values: dict, force: bool = False) -> None:
        if not force and t + 1e-9 < self._next_sig:
            return
        self._next_sig = t + self._sig_dt
        self._sig.write(json.dumps({"t": round(t, 2), **{k: _num(v) for k, v in values.items()}}) + "\n")

    def event(self, t: float, kind: str, text: str, **data) -> None:
        self._since.append(kind)
        self._ev.write(json.dumps({"t": round(t, 2), "kind": kind, "text": text, **data}, default=str) + "\n")

    def finish(self, score: dict, summary: dict) -> None:
        self.traj.details["score"] = score
        self.traj.details["summary"] = summary
        if self._open_call is not None:        # the episode ended before the last call was answered
            for step in reversed(self.traj.content):
                if getattr(step, "tool_call_id", None) == self._open_call:
                    step.tool_call_id = None
                    break
            self._open_call = None
        (self.dir / "trajectory.json").write_text(self.traj.model_dump_json(indent=1))
        (self.dir / "score.json").write_text(json.dumps(score, indent=2, default=str))
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        self._sig.close()
        self._ev.close()
        from .validation import validate
        problems = validate(self.traj)
        if problems:
            warnings.warn(f"{self.traj.id}: trajectory.json written but it fails validation: "
                          + "; ".join(problems[:5]) + (" ..." if len(problems) > 5 else ""), stacklevel=2)


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return round(f, 3)


def load(episode_dir: str | Path) -> Trajectory:
    return Trajectory.model_validate_json((Path(episode_dir) / "trajectory.json").read_text())
