"""System One routing: for each decision point, should the drone
    commit    keep doing what it is doing without asking any model (cheapest),
    local     ask the onboard student pilot (fast, offline), or
    escalate  ask the remote teacher (slow, needs a link)?

The router is a small, fast classifier (a System One model: Laya ~0.4B, Kev
0.8B/4B, or Jev via API as a reference). It reads a compact feature string,
not the full prompt, and returns probabilities over the three choices; the
loop picks the argmax, or escalates when unsure (max probability below
`confidence`).

Interfaces here; the trained router comes from McGlashan's routing dataset
(avians_adp.sft.router_examples) fine-tuned onto Kev or Laya.

TODO: SystemOneRouter speaks a generic JSON-over-HTTP shape (below). Check it
against the System One API reference before pointing it at Jev, Kev (which
implements that API) or a local server, and adjust _payload/_parse.
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass

CHOICES = ("commit", "local", "escalate")


@dataclass
class Route:
    choice: str
    probs: dict
    router: str
    wall_s: float = 0.0


class FixedRouter:
    """Always the same answer: 'escalate' collects teacher data; 'local' runs the student alone."""

    def __init__(self, choice: str = "escalate"):
        assert choice in CHOICES
        self.choice = choice

    def route(self, features: str) -> Route:
        return Route(self.choice, {c: float(c == self.choice) for c in CHOICES}, f"fixed:{self.choice}")


class SystemOneRouter:
    def __init__(self, base_url: str, model: str, confidence: float = 0.6, timeout_s: float = 2.0,
                 api_key: str | None = None):
        self.url = base_url.rstrip("/") + "/v1/systemone"
        self.model, self.confidence, self.timeout, self.key = model, confidence, timeout_s, api_key

    def _payload(self, features: str) -> dict:
        return {"model": self.model,
                "question": {"type": "choice", "text": ROUTER_QUESTION, "context": features,
                             "options": list(CHOICES)}}

    def _parse(self, body: dict) -> dict:
        probs = body.get("probabilities") or body.get("probs") or {}
        return {c: float(probs.get(c, 0.0)) for c in CHOICES}

    def route(self, features: str) -> Route:
        import time
        t0 = time.monotonic()
        req = urllib.request.Request(self.url, json.dumps(self._payload(features)).encode(),
                                     {"Content-Type": "application/json",
                                      **({"Authorization": f"Bearer {self.key}"} if self.key else {})})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                probs = self._parse(json.loads(r.read()))
        except Exception as e:                      # no link, no router: be safe and ask the teacher
            return Route("escalate", {}, f"system_one:{self.model} error {e}", time.monotonic() - t0)
        best = max(probs, key=probs.get)
        choice = best if probs[best] >= self.confidence else "escalate"
        return Route(choice, probs, f"system_one:{self.model}", time.monotonic() - t0)


ROUTER_QUESTION = ("A drone's pilot must decide what to do next. Given the situation, should it keep "
                   "its current skill (commit), ask the small onboard model (local), or ask the large "
                   "remote model (escalate)?")


def features(state: dict, events: list) -> str:
    """The compact situation the router sees."""
    b = state["belief"]
    return (f"skill={state['skill']}:{state['skill_status']} battery={state['battery']:.0f} "
            f"found={b['found']} conf={b['conf']:.2f} payload={state['payload']} "
            f"released={state['released']} help={state['help_arrived']} "
            f"events={','.join(e.kind for e in events) or 'none'}")


def make_router(cfg: dict | None):
    cfg = cfg or {}
    kind = cfg.get("kind", "none")
    if kind == "none":
        return None
    if kind == "fixed":
        return FixedRouter(cfg.get("choice", "escalate"))
    if kind == "system_one":
        s = cfg["system_one"]
        return SystemOneRouter(s["base_url"], s["model"], float(s.get("confidence", 0.6)))
    raise ValueError(f"router kind {kind!r}")
