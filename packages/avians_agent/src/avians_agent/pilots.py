"""Pilots: who makes the decisions.

    AnthropicPilot   Claude through the Anthropic SDK (the teacher: Haiku 5.5)
    ScriptedPilot    a rule-based SAR policy: the baseline every learned pilot
                     must beat, the dry run (no API key), and the test fixture

All take a DecisionRequest and return a Decision: one tool call (or keep),
its reason, the pilot's notes, and what it cost.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field


@dataclass
class DecisionRequest:
    system: str
    prompt: str
    tools: list[dict]
    state: dict                 # structured snapshot (scripted pilot; logged in details)


@dataclass
class Decision:
    tool: str                   # a skill name, or "keep"
    args: dict = field(default_factory=dict)
    reason: str = ""
    notes: str | None = None
    text: str = ""              # the pilot's visible text
    thinking: str = ""          # summarized thinking, if any
    model: str = ""
    usage: dict = field(default_factory=dict)
    wall_s: float = 0.0         # real time the call took (to fit latency models)
    error: str = ""


def split_common(args: dict) -> tuple[dict, str, str | None]:
    args = dict(args or {})
    reason = str(args.pop("reason", "") or "")
    notes = args.pop("notes", None)
    return args, reason, notes


class AnthropicPilot:
    def __init__(self, model: str = "claude-haiku-5-5", effort: str = "medium",
                 thinking_display: str = "summarized", max_tokens: int = 4000, client=None):
        import anthropic
        self.anthropic = anthropic
        self.client = client or anthropic.Anthropic()
        self.model, self.effort = model, effort
        self.display, self.max_tokens = thinking_display, max_tokens
        self.name = model

    def decide(self, req: DecisionRequest) -> Decision:
        t0 = time.monotonic()
        try:
            r = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                # tools -> system -> messages: one breakpoint on the system prompt caches both
                system=[{"type": "text", "text": req.system, "cache_control": {"type": "ephemeral"}}],
                tools=req.tools,
                tool_choice={"type": "auto"},
                thinking={"type": "adaptive", "display": self.display},
                output_config={"effort": self.effort},
                messages=[{"role": "user", "content": req.prompt}],
            )
        except self.anthropic.APIStatusError as e:
            return Decision("keep", reason="API error", model=self.model, error=f"{e.status_code}: {e.message}",
                            wall_s=time.monotonic() - t0)
        except self.anthropic.APIConnectionError as e:
            return Decision("keep", reason="API unreachable", model=self.model, error=str(e),
                            wall_s=time.monotonic() - t0)
        wall = time.monotonic() - t0
        text = "".join(b.text for b in r.content if b.type == "text")
        thinking = "".join(getattr(b, "thinking", "") or "" for b in r.content if b.type == "thinking")
        calls = [b for b in r.content if b.type == "tool_use"]
        u = r.usage
        usage = {"input": u.input_tokens, "output": u.output_tokens,
                 "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
                 "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0,
                 "stop": r.stop_reason}
        if not calls:
            return Decision("keep", reason="no tool call", text=text, thinking=thinking, model=self.model,
                            usage=usage, wall_s=wall)
        args, reason, notes = split_common(calls[0].input)
        extra = f" (ignored {len(calls) - 1} more call(s))" if len(calls) > 1 else ""
        return Decision(calls[0].name, args, reason + extra, notes, text, thinking, self.model, usage, wall)


class ScriptedPilot:
    """A competent rule-based search-and-rescue policy, from the state snapshot.

    Search the regions in order of prior probability (skipping closed ones),
    report as soon as found, top the battery up while loading the kit, inspect
    if the estimate is loose, deliver, then orbit until help arrives; recharge
    whenever the battery would not cover the next leg and the way back."""

    name = "scripted"

    def __init__(self):
        self.searched: dict[str, float] = {}
        self.charged = False          # charged to full once, with the kit aboard
        self.moving = False           # the person was seen walking: don't wait on the ground
        self.watched = False          # watched them for a while after finding (to see if they move)

    def decide(self, req: DecisionRequest) -> Decision:
        s = req.state
        d = self._policy(s)
        d.model = "scripted"
        return d

    def _policy(self, s: dict) -> Decision:
        skill, status = s["skill"], s["skill_status"]
        running = status == "running"
        b, bat = s["belief"], s["battery"]
        others = s.get("others") or []
        if others and s["me"] != min([s["me"]] + [o["id"] for o in others]):
            return self._wingman(s)          # not the lowest id: the supporting drone
        if s["help_arrived"]:
            if s["landed"]:
                return Decision("keep", reason="mission complete")
            return Decision("keep") if skill in ("rtl",) and running else Decision("rtl", reason="help is here")
        if s["released"]:
            if skill in ("track_person", "search_around") and running:
                return Decision("keep")
            if b["sigma"] > 15 or (skill == "track_person" and status == "failed"):
                return Decision("search_around", {"around": "person", "alt": 25}, reason="lost them: datum search")
            return Decision("track_person", {"mode": "orbit", "until": "help_arrived"},
                            reason="stay with the person until help arrives")
        if b["found"]:
            if not s["help_dispatched"]:
                return Decision("report", {"kind": "person_found"}, reason="dispatch the helicopter")
            if not s["payload"]:
                if skill in ("recharge", "pick_up", "inspect") and running:
                    return Decision("keep")
                if not self.watched and not s["released"]:
                    self.watched = True
                    return Decision("inspect", {"alt": 20, "seconds": 12},
                                    reason="watch them briefly: are they moving?")
                if math.hypot(b.get("ve", 0.0), b.get("vn", 0.0)) > 0.12:
                    self.moving = True
                return Decision("pick_up", {"depot": "base"}, reason="get the kit")
            # with the kit: charge to full at base (a recharge the reflex interrupted is retried),
            # then, since the battery cannot hover through a 10-minute wait, wait on the ground
            # until help is about 5 minutes out, then fly, drop and stay
            if skill == "recharge" and running:
                return Decision("keep")
            if skill == "recharge" and status == "succeeded":
                self.charged = True
            if bat < 90 and not self.charged:
                return Decision("recharge", {"station": "pad_charge", "until_pct": 100},
                                reason="charge to full before the wait with the person")
            if math.hypot(b.get("ve", 0.0), b.get("vn", 0.0)) > 0.12 and b["sigma"] < 5:
                self.moving = True
            eta = s["help_eta_t"]
            if eta is not None and eta - s["t"] > 300 and not s["released"] and not self.moving:
                if skill in ("precision_land", "wait_until") and running:
                    return Decision("keep")
                if not s["landed"]:
                    return Decision("precision_land", {"place": "base"},
                                    reason=f"help is {(eta - s['t']) / 60:.0f} min out: wait on the ground to save battery")
                return Decision("wait_until", {"condition": "time", "value": eta - 300, "timeout_s": 1500},
                                reason="wait on the pad until help is 5 minutes out")
            if s["landed"]:
                return Decision("takeoff", {"alt": 30}, reason="help is close: go deliver")
            if skill in ("deliver", "inspect", "takeoff", "search_around") and running:
                return Decision("keep")
            if "deliver" in s["rejected"] and b["sigma"] <= 6:
                return Decision("track_person", {"until": "help_arrived"},
                                reason="cannot drop there; stay with the person instead")
            if b["sigma"] > 15 or (skill == "inspect" and status == "succeeded" and b["conf"] < 0.6):
                return Decision("search_around", {"around": "person", "alt": 25},
                                reason=f"the estimate has drifted (+/- {b['sigma']:.0f} m): datum search")
            if b["sigma"] > 6 and "inspect" not in s["rejected"]:
                return Decision("inspect", {"alt": 15, "seconds": 8}, reason="tighten the estimate before the drop")
            return Decision("deliver", {"target": "person", "drop_alt": 8}, reason="drop the kit")
        # not found
        if s["landed"] and skill != "recharge":
            return Decision("takeoff", {"alt": 30}, reason="start the search")
        if skill in ("recharge", "swap_battery", "takeoff") and running:
            return Decision("keep")
        if bat < s["battery_needed_home"] + 15:
            return Decision("recharge", {"station": "pad_charge", "until_pct": 95}, reason="battery low")
        if skill == "inspect" and running:
            return Decision("keep")
        if b["source"] != "prior" and b["conf"] > 0.4 and skill != "inspect":
            return Decision("inspect", {"alt": 15, "seconds": 8}, reason="check the candidate")
        if skill == "search_region" and running:
            return Decision("keep")
        return self._next_search(s)

    def _next_search(self, s: dict) -> Decision:
        """The most overdue open region that no other drone has claimed."""
        skill, status = s["skill"], s["skill_status"]
        if skill == "search_region" and status in ("succeeded", "failed"):
            self.searched[s["skill_args"].get("region", "")] = s["t"]
        taken = {r for r, who in (s.get("claims") or {}).items() if who != s["me"]}
        order = sorted(s["regions"], key=lambda r: (self.searched.get(r["id"], -1e9), -r["prior"]))
        for r in order:
            if not r["closed"] and r["id"] not in taken:
                return Decision("search_region", {"region": r["id"], "alt": 30},
                                reason=f"search {r['id']} (prior {r['prior']:.0%})")
        for r in order:
            if not r["closed"]:
                return Decision("search_region", {"region": r["id"], "alt": 30},
                                reason=f"search {r['id']} again (all regions claimed)")
        return Decision("hover", {"seconds": 0}, reason="nothing open to search")

    def _wingman(self, s: dict) -> Decision:
        """The supporting drone: searches its own regions until the person is
        found, then stays on them (orbit) so the estimate stays sharp while the
        lead drone fetches the kit; recharges when low and comes back."""
        skill, status = s["skill"], s["skill_status"]
        running = status == "running"
        b, bat = s["belief"], s["battery"]
        if s["help_arrived"]:
            if s["landed"]:
                return Decision("keep", reason="mission complete")
            return Decision("keep") if skill == "rtl" and running else Decision("rtl", reason="help is here")
        if s["landed"] and skill not in ("recharge", "takeoff") or (skill == "takeoff" and not running and s["landed"]):
            return Decision("takeoff", {"alt": 30}, reason="start")
        if skill in ("recharge", "takeoff") and running:
            return Decision("keep")
        if bat < s["battery_needed_home"] + 25:
            if "recharge" in s["rejected"]:        # the charging pad is taken: swap at the forward station
                return Decision("swap_battery", {"station": "pad_swap"}, reason="battery low; charging pad in use")
            return Decision("recharge", {"station": "pad_charge", "until_pct": 95}, reason="battery low")
        if b["found"]:
            if skill in ("track_person", "search_around") and running:
                return Decision("keep")
            if b["sigma"] > 15 or (skill == "track_person" and status == "failed"):
                return Decision("search_around", {"around": "person", "alt": 25}, reason="lost them: datum search")
            return Decision("track_person", {"mode": "orbit", "alt": 30, "standoff": 15, "until": "help_arrived"},
                            reason="keep eyes on the person while the lead drone fetches the kit")
        if skill == "inspect" and running:
            return Decision("keep")
        if b["source"] != "prior" and b["conf"] > 0.4 and skill != "inspect":
            return Decision("inspect", {"alt": 15, "seconds": 8}, reason="check the candidate")
        if skill == "search_region" and running:
            return Decision("keep")
        return self._next_search(s)


def make_pilot(name: str, cfg: dict):
    p = cfg["pilots"][name]
    if p["provider"] == "scripted":
        return ScriptedPilot()
    if p["provider"] == "anthropic":
        return AnthropicPilot(p["model"], p.get("effort", "medium"), p.get("thinking_display", "summarized"),
                              int(p.get("max_tokens", 4000)))
    raise ValueError(f"unknown provider {p['provider']!r}")


def snapshot(sim, executor) -> dict:
    """The structured state the scripted pilot reads (and every decision logs)."""
    v, w = sim.vehicle.s, sim.world
    b = sim.tracker.belief()
    sk = executor.skill
    home = min((s.position for s in w.stations.values() if s.kind == "charge"),
               key=lambda p: math.dist(p, (v.east, v.north)), default=w.launch)
    weights = w.person.region_weights
    sw = getattr(sim.actuators, "swarm", None)
    return {
        "me": sim.drone_id, "others": sw.others(sim.drone_id) if sw else [],
        "claims": dict(sw.claims) if sw else {},
        "t": round(sim.t, 2), "east": v.east, "north": v.north, "alt": v.alt, "battery": v.battery,
        "landed": v.landed, "payload": v.payload, "released": bool(sim.kits),
        "skill": sk.name if sk else None, "skill_status": sk.status if sk else None,
        "skill_args": dict(sk.args) if sk else {},
        "rejected": [h["skill"] for h in executor.history[-3:] if not h["accepted"]],
        "belief": {"east": b.east, "north": b.north, "sigma": b.sigma, "conf": b.conf,
                   "found": b.found, "source": b.source, "ve": b.ve, "vn": b.vn},
        "help_dispatched": sim.world_state.help.dispatched_t is not None,
        "help_eta_t": sim.world_state.help.arrival_t,
        "help_arrived": sim.world_state.help_arrived(sim.t),
        "battery_needed_home": sim.energy.cost(math.dist(home, (v.east, v.north)),
                                               sim.ctx.limits.cruise_speed, v.payload),
        "regions": [{"id": r.id, "prior": weights.get(r.id, 0.0),
                     "closed": any(z.shape.intersection(r.shape).area > 0.5 * r.shape.area
                                   for z in sim.world_state.unsafe_zones(sim.t))}
                    for r in w.search_regions()],
    }
