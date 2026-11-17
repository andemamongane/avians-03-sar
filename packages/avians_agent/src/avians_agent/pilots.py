"""Pilots: who makes the decisions.

    AnthropicPilot   Claude through the Anthropic SDK (the teacher: Haiku 5.5)
    OpenRouterPilot  any tool-calling model on OpenRouter (OpenAI-compatible
                     chat completions), e.g. NVIDIA Nemotron 3 Ultra (free)
    ScriptedPilot    a rule-based SAR policy: the baseline every learned pilot
                     must beat, the dry run (no API key), and the test fixture

All take a DecisionRequest and return a Decision: one tool call (or keep),
its reason, the pilot's notes, and what it cost.
"""
from __future__ import annotations

import http.client
import json
import math
import os
import time
import urllib.error
import urllib.request
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


class OpenRouterPilot:
    """A model on OpenRouter, through its OpenAI-compatible chat completions API (stdlib HTTP only).

    Same contract as AnthropicPilot: one request per decision (system prompt, decision prompt, the
    tools in the OpenAI layout), the first tool call becomes the Decision, `reason` and `notes` are
    split out of its arguments, the model's reasoning text is `thinking` (raw reasoning, not a
    Claude-style summary). A reply without a usable tool call (none, no name, arguments that are not a
    JSON object) is `keep` with reason "no tool call" (avians_adp: no_tool_call). Anything that says
    nothing about the model -- an HTTP or provider error, a dropped connection, a malformed body -- is
    `keep` with `error` set (avians_adp: api_error), and never raises: a live episode must not die.

    Rate limits (OpenRouter free models, ids ending ":free"): 20 requests a minute; 50 a day, or 1,000
    a day once $10 of credits have been bought. 408/429/500/502/503/504/524 and connection failures are
    retried, after the wait the server asks for (Retry-After, or X-RateLimit-Reset in ms or s), else
    with backoff, up to `retries` times; a timeout or dropped connection is retried once. A wait longer than `max_wait_s`, or a message naming a
    per-day limit, means the daily cap: the pilot then stops calling (`exhausted`), and every later
    decision fails fast as an API error. `wall_s` times the successful attempt only (no retry sleeps),
    so `avians latency-fit` measures the model, not the rate limiter.
    """

    URL = "https://openrouter.ai/api/v1/chat/completions"
    RETRY = (408, 429, 500, 502, 503, 504, 524)

    def __init__(self, model: str, name: str | None = None, effort: str | None = "medium",
                 max_tokens: int = 8000, tool_choice: str = "auto", retries: int = 4, max_wait_s: float = 90.0,
                 timeout_s: float = 120.0, api_key: str | None = None, post=None, sleep=time.sleep,
                 url: str | None = None):
        self.model, self.name = model, name or model
        self.effort, self.max_tokens, self.tool_choice = effort, max_tokens, tool_choice
        self.retries, self.max_wait_s, self.timeout_s = retries, max_wait_s, timeout_s
        self.url = url or self.URL
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if post is None and not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set: add it to .env at the repository root")
        self._post, self._sleep = post or self._http_post, sleep
        self.exhausted = ""

    def _http_post(self, body: dict) -> tuple[int, dict, object]:
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), method="POST", headers={
            "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json", "X-Title": "avians"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                return r.status, dict(r.headers), json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                raw = e.read() or b""
            except Exception:                               # the error body itself can fail mid-read
                raw = b""
            try:
                data = json.loads(raw)
            except ValueError:
                data = {"error": {"message": raw.decode(errors="replace")[:500] or str(e)}}
            return e.code, dict(e.headers or {}), data

    @staticmethod
    def _wait_s(headers: dict, attempt: int) -> float:
        h = {k.lower(): v for k, v in (headers or {}).items()}
        try:
            if "retry-after" in h:
                return max(0.0, float(h["retry-after"]))
            if "x-ratelimit-reset" in h:                       # epoch ms (OpenRouter); accept epoch s too
                reset = float(h["x-ratelimit-reset"])
                return max(0.0, (reset / 1000.0 if reset > 1e11 else reset) - time.time())
        except ValueError:
            pass
        return min(60.0, 5.0 * 2 ** attempt)

    @staticmethod
    def _code(err, status: int) -> int:
        try:
            return int((err or {}).get("code", status)) if isinstance(err, dict) else int(status)
        except (TypeError, ValueError):
            return int(status)

    def body(self, req: DecisionRequest) -> dict:
        from avians_adp.sft import openai_tools
        b = {"model": self.model, "max_tokens": self.max_tokens,
             "messages": [{"role": "system", "content": req.system}, {"role": "user", "content": req.prompt}],
             "tools": openai_tools(req.tools), "tool_choice": self.tool_choice}
        if self.effort:
            b["reasoning"] = {"effort": self.effort}
        return b

    def _fail(self, reason: str, error: str, t0: float) -> Decision:
        return Decision("keep", reason=reason, model=self.model, error=error, wall_s=time.monotonic() - t0)

    def decide(self, req: DecisionRequest) -> Decision:
        t0 = time.monotonic()
        if self.exhausted:
            return self._fail("API error", self.exhausted, t0)
        try:
            return self._decide(req, t0)
        except Exception as e:                               # last line of defence: never kill the episode
            return self._fail("API error", f"{type(e).__name__}: {e}", t0)

    def _decide(self, req: DecisionRequest, t0: float) -> Decision:
        body = self.body(req)
        data: object = None
        for attempt in range(self.retries + 1):
            t_try = time.monotonic()
            try:
                status, headers, data = self._post(body)
            except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError, ValueError) as e:
                if attempt == 0 and self.retries > 0:        # a dropped connection or timeout: once only
                    self._sleep(5.0)                          # (a timed-out request may still count to quota)
                    continue
                return self._fail("API unreachable", f"{type(e).__name__}: {e}", t0)
            if not isinstance(data, dict):
                return self._fail("API error", f"{status}: reply is not a JSON object: {str(data)[:200]}", t0)
            err = data.get("error")
            if status == 200 and not err:
                break
            code = self._code(err, status)
            msg = (err.get("message", "") if isinstance(err, dict) else str(err or "")) or f"HTTP {status}"
            if status in self.RETRY or code in self.RETRY:
                wait = self._wait_s(headers, attempt)
                if wait > self.max_wait_s or "per-day" in msg.lower() or "per day" in msg.lower():
                    self.exhausted = f"{code}: rate limit, wait {wait / 3600:.1f} h (daily cap): {msg}"
                    return self._fail("API error", self.exhausted, t0)
                if attempt < self.retries:
                    self._sleep(wait)
                    continue
            return self._fail("API error", f"{code}: {msg}", t0)
        wall = time.monotonic() - t_try                      # the successful attempt only
        choices = data.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            return self._fail("API error", f"no choices in reply: {str(data)[:300]}", t0)
        ch = choices[0]
        if ch.get("error") or ch.get("finish_reason") == "error":    # the upstream provider failed mid-reply
            e = ch.get("error")
            return self._fail("API error", f"provider error: {e.get('message', e) if isinstance(e, dict) else e}", t0)
        msg = ch.get("message") or {}
        text, thinking = msg.get("content") or "", msg.get("reasoning") or ""
        u = data.get("usage") or {}
        usage = {"input": int(u.get("prompt_tokens") or 0), "output": int(u.get("completion_tokens") or 0),
                 "reasoning": int((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0),
                 "stop": ch.get("finish_reason"), "provider": data.get("provider")}
        calls = msg.get("tool_calls") or []
        if not calls:
            return Decision("keep", reason="no tool call", text=text, thinking=thinking, model=self.model,
                            usage=usage, wall_s=wall)
        fn = calls[0].get("function") or {}
        name, raw = fn.get("name"), fn.get("arguments") or "{}"
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else dict(raw)
            if not isinstance(parsed, dict) or not isinstance(name, str) or not name:
                raise ValueError("no tool name, or arguments that are not an object")
        except (ValueError, TypeError):
            return Decision("keep", reason="no tool call", model=self.model, usage=usage, wall_s=wall,
                            thinking=thinking,
                            text=(text + "\n" if text else "") + f"[unusable call to {name!r}: {str(raw)[:300]}]")
        args, reason, notes = split_common(parsed)
        extra = f" (ignored {len(calls) - 1} more call(s))" if len(calls) > 1 else ""
        return Decision(name, args, reason + extra, notes, text, thinking, self.model, usage, wall)


class ScriptedPilot:
    """A competent rule-based search-and-rescue policy, from the state snapshot.

    Search the regions in order of prior probability (skipping closed ones),
    report as soon as found, top the battery up while loading the kit, inspect
    if the estimate is loose, deliver, then orbit until help arrives; recharge
    whenever the battery would not cover the next leg and the way back."""

    name = "scripted"

    def __init__(self):
        self.searched: dict[str, float] = {}

    def decide(self, req: DecisionRequest) -> Decision:
        s = req.state
        d = self._policy(s)
        d.model = "scripted"
        return d

    def _policy(self, s: dict) -> Decision:
        skill, status = s["skill"], s["skill_status"]
        running = status == "running"
        b, bat = s["belief"], s["battery"]
        if s["help_arrived"]:
            if s["landed"]:
                return Decision("keep", reason="mission complete")
            return Decision("keep") if skill in ("rtl",) and running else Decision("rtl", reason="help is here")
        if s["released"]:
            if skill == "track_person" and running:
                return Decision("keep")
            return Decision("track_person", {"mode": "orbit", "until": "help_arrived"},
                            reason="stay with the person until help arrives")
        if b["found"]:
            if not s["help_dispatched"]:
                return Decision("report", {"kind": "person_found"}, reason="dispatch the helicopter")
            if not s["payload"]:
                if skill == "recharge" and running:
                    return Decision("keep")
                if skill == "pick_up" and running:
                    return Decision("keep")
                if bat < 85 and skill != "recharge":
                    return Decision("recharge", {"station": "pad_charge", "until_pct": 98},
                                    reason="top up before the long wait with the person")
                return Decision("pick_up", {"depot": "base"}, reason="get the kit")
            if skill in ("deliver", "inspect") and running:
                return Decision("keep")
            if "deliver" in s["rejected"]:
                return Decision("track_person", {"until": "help_arrived"},
                                reason="cannot drop there; stay with the person instead")
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
        if skill == "search_region" and status in ("succeeded", "failed"):
            self.searched[s["skill_args"].get("region", "")] = s["t"]
        order = sorted(s["regions"], key=lambda r: (self.searched.get(r["id"], -1e9), -r["prior"]))
        for r in order:
            if not r["closed"]:
                return Decision("search_region", {"region": r["id"], "alt": 30},
                                reason=f"search {r['id']} (prior {r['prior']:.0%})")
        return Decision("hover", {"seconds": 0}, reason="nothing open to search")


def make_pilot(name: str, cfg: dict):
    p = cfg["pilots"][name]
    if p["provider"] == "scripted":
        return ScriptedPilot()
    from .env import load_env
    load_env()                      # API keys from .env, for every caller (avians_adp.student_calls too)
    if p["provider"] == "openrouter":
        return OpenRouterPilot(p["model"], name=name, effort=p.get("effort", "medium"),
                               max_tokens=int(p.get("max_tokens", 8000)), tool_choice=p.get("tool_choice", "auto"),
                               retries=int(p.get("retries", 4)), max_wait_s=float(p.get("max_wait_s", 90)),
                               timeout_s=float(p.get("timeout_s", 120)), url=p.get("url"))
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
    return {
        "t": round(sim.t, 2), "east": v.east, "north": v.north, "alt": v.alt, "battery": v.battery,
        "landed": v.landed, "payload": v.payload, "released": bool(sim.kits),
        "skill": sk.name if sk else None, "skill_status": sk.status if sk else None,
        "skill_args": dict(sk.args) if sk else {},
        "rejected": [h["skill"] for h in executor.history[-3:] if not h["accepted"]],
        "belief": {"east": b.east, "north": b.north, "sigma": b.sigma, "conf": b.conf,
                   "found": b.found, "source": b.source},
        "help_dispatched": sim.world_state.help.dispatched_t is not None,
        "help_arrived": sim.world_state.help_arrived(sim.t),
        "battery_needed_home": sim.energy.cost(math.dist(home, (v.east, v.north)),
                                               sim.ctx.limits.cruise_speed, v.payload),
        "regions": [{"id": r.id, "prior": weights.get(r.id, 0.0),
                     "closed": any(z.shape.intersection(r.shape).area > 0.5 * r.shape.area
                                   for z in sim.world_state.unsafe_zones(sim.t))}
                    for r in w.search_regions()],
    }
