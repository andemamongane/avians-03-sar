"""What the pilot remembers between decisions (storage option C).

Each decision prompt is self-contained: the model sees no earlier turns, only
  - structured mission memory, kept by the loop from facts (what has been
    searched and when, sightings, reports, delivery, help, zone changes,
    battery service, markers)
  - the pilot's own notes, which it rewrites through the `notes` field
  - the last k decisions and what they returned
The full growing history is still stored (option A) in the ADP trajectory,
for training on longer windows later.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Memory:
    history_k: int = 4
    max_notes: int = 1200
    notes: str = ""
    decisions: list[dict] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)

    def set_notes(self, text: str | None) -> None:
        if text is not None:
            self.notes = text.strip()[: self.max_notes]

    def add_decision(self, t: float, call: str, result: str, reason: str = "") -> None:
        self.decisions.append({"t": t, "call": call, "result": result, "reason": reason})

    def add_fact(self, t: float, text: str) -> None:
        self.facts.append(f"t={t:.0f}: {text}")

    def render_decisions(self) -> str:
        rows = self.decisions[-self.history_k:]
        if not rows:
            return "(none yet)"
        return "\n".join(f"- t={d['t']:.0f} {d['call']} -> {d['result']}" for d in rows)

    def render_facts(self, k: int = 12) -> str:
        return "\n".join(f"- {f}" for f in self.facts[-k:]) or "(none yet)"
