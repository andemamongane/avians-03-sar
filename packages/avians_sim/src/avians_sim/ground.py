"""The ground station: the other end of the drone's radio.

Everything the drone sends (reports, status, questions) and everything it
receives (acknowledgements, operator messages, tasking, news from the
scenario) goes through one GroundStation with a modelled link:

    latency_s      a report is acknowledged after this long (sim time)
    loss           probability a message is lost (the drone hears nothing)
    range_m        beyond this from the station there is no link at all

Messages to the drone are queued with a delivery time and handed out by
`inbox(t0, t1)`; the simulator turns them into "message" events, which wake
the pilot. Three sources feed the queue:

    scenario news        from the world family (avians_env.family.News)
    scripted operator    answers and tasking from the mission file
                         (`ground.operator: [{t, text}, ...]`, and
                         `ground.answers: {pattern: text}` for questions)
    live operator        a person at the console (ground_live.py), typing
                         while the episode runs in real time

A question (`request_help`) is answered asynchronously: the skill returns
"sent" at once; the answer, if any, arrives as a message later. In multi-drone
missions the same station relays each drone's shared state to the others
(`broadcast`, step 2), so comms and swarm use one mechanism.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Message:
    t: float                       # delivered at (sim time)
    sender: str                    # ground | operator | scenario | drone id
    to: str                        # drone id, or "all"
    kind: str                      # ack | answer | tasking | news | status | question | report
    text: str
    data: dict = field(default_factory=dict)
    sent_t: float | None = None    # when it was sent (for latency)


class GroundStation:
    def __init__(self, world, cfg: dict | None = None, rng=None):
        cfg = cfg or {}
        self.world = world
        self.latency = float(cfg.get("latency_s", 2.0))
        self.loss = float(cfg.get("loss", 0.0))
        self.range_m = float(cfg.get("range_m", 1e9))
        self.position = tuple(cfg.get("position", world.launch))
        self.name = str(cfg.get("name", "base"))
        self.rng = rng
        self.queue: list[Message] = []          # to drones, by delivery time
        self.log: list[Message] = []            # everything, both ways
        self.answers = {str(k): str(v) for k, v in (cfg.get("answers") or {}).items()}
        for item in cfg.get("operator", []) or []:
            self.queue.append(Message(float(item["t"]), "operator", item.get("to", "all"),
                                      item.get("kind", "tasking"), str(item["text"])))
        self.live = None                        # ground_live.Console, if any
        self.pending_questions: list[dict] = []

    # ------------------------------------------------------------ link
    def in_range(self, pos) -> bool:
        import math
        return math.dist((pos[0], pos[1]), self.position) <= self.range_m

    def _lost(self) -> bool:
        return self.rng is not None and self.loss > 0 and self.rng.random() < self.loss

    def _reply(self, t: float, to: str, kind: str, text: str, **data) -> None:
        self.queue.append(Message(t + self.latency, "ground", to, kind, text, data, sent_t=t))

    # ------------------------------------------------------------ from the drone
    def send(self, t: float, drone: str, kind: str, text: str, pos, world_state=None, **data) -> str:
        """A report/status/question from the drone. Returns what the drone knows
        at once (sent, or no link); the substance comes back as a message."""
        m = Message(t, drone, "ground", kind, text, data, sent_t=t)
        if not self.in_range(pos):
            m.data["delivered"] = False
            self.log.append(m)
            return f"no link to {self.name} from here ({self.range_m:.0f} m range): not sent"
        if self._lost():
            m.data["delivered"] = False
            self.log.append(m)
            return "sent (no acknowledgement yet)"
        m.data["delivered"] = True
        self.log.append(m)
        if self.live is not None:
            self.live.post(m)
        if kind == "person_found" and world_state is not None:
            reply = world_state.report_found(t, data["east"], data["north"])
            self._reply(t, drone, "ack", reply, east=data["east"], north=data["north"])
            return "sent; the crew will acknowledge"
        if kind == "question":
            self.pending_questions.append({"t": t, "drone": drone, "text": text, "answered": False})
            ans = self._scripted_answer(text)
            if ans:
                self._reply(t + 10.0, drone, "answer", ans)
                self.pending_questions[-1]["answered"] = True
            elif self.live is None:
                self._reply(t + 30.0, drone, "answer", "operator: no new information; use your judgement")
                self.pending_questions[-1]["answered"] = True
            return "sent; the operator will answer in a message when they can"
        self._reply(t, drone, "ack", f"{self.name}: copy your {kind}")
        return "sent"

    def _scripted_answer(self, question: str) -> str | None:
        for pattern, text in self.answers.items():
            if re.search(pattern, question, re.I):
                return text
        return None

    # ------------------------------------------------------------ to the drone
    def push(self, t: float, sender: str, kind: str, text: str, to: str = "all", **data) -> None:
        """Something to tell the drone(s) now (live operator, scenario news, a relay)."""
        self.queue.append(Message(t + (0.0 if sender == "scenario" else self.latency), sender, to, kind,
                                  text, data, sent_t=t))

    def inbox(self, t0: float, t1: float, drone: str = "all") -> list[Message]:
        """Messages for `drone` delivered in (t0, t1]."""
        due = [m for m in self.queue if t0 < m.t <= t1 and m.to in (drone, "all")]
        for m in due:
            self.queue.remove(m)
            self.log.append(m)
            if self.live is not None:
                self.live.post(m)
        return due

    def summary(self) -> list[dict]:
        return [{"t": round(m.t, 1), "sent_t": m.sent_t, "sender": m.sender, "to": m.to, "kind": m.kind,
                 "text": m.text, **({"data": m.data} if m.data else {})} for m in self.log]
