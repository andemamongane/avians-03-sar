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

STARTER (owner: McGlashan): this works and the pilot loop uses it. TODO in
README.md: schema validation against upstream ADP, SFT/routing builders,
dataset manifests.
"""
from __future__ import annotations

import json
from pathlib import Path

from .schema import APIAction, MessageAction, TextObservation, Trajectory


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
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    def brief(self, system: str, tools: list[dict]) -> None:
        self.traj.details["system"] = system
        self.traj.details["tools"] = tools

    def observation(self, text: str, t: float, source: str = "environment", **details) -> None:
        self.traj.content.append(TextObservation(source=source, content=text, details={"t": t, **details}))

    def message(self, text: str, t: float, **details) -> None:
        self.traj.content.append(MessageAction(content=text, details={"t": t, **details}))

    def action(self, function: str, kwargs: dict, reason: str | None, t: float, **details) -> None:
        self.traj.content.append(APIAction(function=function, kwargs=kwargs, description=reason,
                                           details={"t": t, **details}))

    def tool_result(self, text: str, t: float, **details) -> None:
        self.observation(text, t, source="tool", **details)

    def signals(self, t: float, values: dict, force: bool = False) -> None:
        if not force and t + 1e-9 < self._next_sig:
            return
        self._next_sig = t + self._sig_dt
        self._sig.write(json.dumps({"t": round(t, 2), **{k: _num(v) for k, v in values.items()}}) + "\n")

    def event(self, t: float, kind: str, text: str, **data) -> None:
        self._ev.write(json.dumps({"t": round(t, 2), "kind": kind, "text": text, **data}, default=str) + "\n")

    def finish(self, score: dict, summary: dict) -> None:
        self.traj.details["score"] = score
        self.traj.details["summary"] = summary
        (self.dir / "trajectory.json").write_text(self.traj.model_dump_json(indent=1))
        (self.dir / "score.json").write_text(json.dumps(score, indent=2, default=str))
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        self._sig.close()
        self._ev.close()


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return round(f, 3)


def load(episode_dir: str | Path) -> Trajectory:
    return Trajectory.model_validate_json((Path(episode_dir) / "trajectory.json").read_text())
