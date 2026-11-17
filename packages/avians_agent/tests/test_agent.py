import json
from types import SimpleNamespace

import numpy as np

from avians_agent.latency import LatencyModel, fit
from avians_agent.loop import run_episode
from avians_agent.mission import Mission, load_config
from avians_agent.pilots import AnthropicPilot, DecisionRequest, ScriptedPilot


def test_scripted_episode_satisfies_the_benchmark(tmp_path):
    m = Mission.load("missions/sar_supply.yaml")
    s = run_episode(m, ScriptedPilot(), 0, tmp_path, load_config("agent.yaml"), load_config("latency.yaml"))
    assert s["satisfied"], s["violated"]
    assert s["end"] == "help arrived and landed"
    tj = json.loads((tmp_path / "trajectory.json").read_text())
    acts = [c for c in tj["content"] if c["class_"] == "api_action"]
    assert len(acts) == s["decisions"] and acts[0]["function"] == "takeoff"
    assert (tmp_path / "signals.jsonl").exists() and (tmp_path / "system_prompt.txt").exists()


def test_latency_model_and_fit():
    rng = np.random.default_rng(0)
    m = LatencyModel(2.0, 0.4, 0.0)
    xs = [m.sample(rng) for _ in range(4000)]
    f = fit(xs)
    assert abs(f.median_s - 2.0) < 0.1 and abs(f.sigma - 0.4) < 0.03


class FakeClient:
    """Stands in for anthropic.Anthropic: returns one tool call."""

    def __init__(self):
        self.calls = []

        def create(**kw):
            self.calls.append(kw)
            return SimpleNamespace(
                content=[SimpleNamespace(type="thinking", thinking="r3 first"),
                         SimpleNamespace(type="text", text="Searching r3."),
                         SimpleNamespace(type="tool_use", name="search_region",
                                         input={"region": "r3", "reason": "highest prior", "notes": "r3 then r2"})],
                usage=SimpleNamespace(input_tokens=100, output_tokens=20, cache_read_input_tokens=2000,
                                      cache_creation_input_tokens=0), stop_reason="tool_use")
        self.messages = SimpleNamespace(create=create)


def test_anthropic_pilot_parses_one_call():
    fc = FakeClient()
    p = AnthropicPilot(client=fc)
    d = p.decide(DecisionRequest("sys", "prompt", [{"name": "search_region"}], {}))
    assert d.tool == "search_region" and d.args == {"region": "r3"}
    assert d.reason == "highest prior" and d.notes == "r3 then r2" and d.thinking == "r3 first"
    kw = fc.calls[0]
    assert kw["model"] == "claude-haiku-5-5" and kw["thinking"]["type"] == "adaptive"
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}


# ------------------------------------------------------------------ OpenRouter pilot (no network: a fake transport)

def _reply(tool=None, args=None, text="", reasoning="", raw_args=None, n_calls=1):
    msg = {"role": "assistant", "content": text, "reasoning": reasoning}
    if tool:
        a = raw_args if raw_args is not None else json.dumps(args or {})
        msg["tool_calls"] = [{"id": f"c{i}", "type": "function", "function": {"name": tool, "arguments": a}}
                             for i in range(n_calls)]
    return 200, {}, {"provider": "Nvidia", "choices": [{"message": msg, "finish_reason": "tool_calls" if tool else "stop"}],
                     "usage": {"prompt_tokens": 6000, "completion_tokens": 300,
                               "completion_tokens_details": {"reasoning_tokens": 250}}}


class FakeTransport:
    def __init__(self, *replies):
        self.replies, self.bodies, self.slept = list(replies), [], []

    def post(self, body):
        self.bodies.append(body)
        return self.replies.pop(0)


def _or_pilot(ft, **kw):
    from avians_agent.pilots import OpenRouterPilot
    return OpenRouterPilot("nvidia/nemotron-3-ultra-550b-a55b:free", name="nemotron", post=ft.post,
                           sleep=ft.slept.append, **kw)


REQ = DecisionRequest("sys", "prompt", [{"name": "search_region", "description": "Search",
                                         "input_schema": {"type": "object", "properties": {"region": {"type": "string"}}}}], {})


def test_openrouter_pilot_parses_one_call():
    ft = FakeTransport(_reply("search_region", {"region": "r3", "reason": "highest prior", "notes": "r3 then r2"},
                              text="Searching r3.", reasoning="r3 has the highest prior", n_calls=2))
    d = _or_pilot(ft).decide(REQ)
    assert (d.tool, d.args, d.notes) == ("search_region", {"region": "r3"}, "r3 then r2")
    assert d.reason == "highest prior (ignored 1 more call(s))"
    assert d.thinking == "r3 has the highest prior" and d.text == "Searching r3." and not d.error
    assert d.model == "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert d.usage["input"] == 6000 and d.usage["output"] == 300 and d.usage["reasoning"] == 250
    b = ft.bodies[0]
    assert b["model"] == "nvidia/nemotron-3-ultra-550b-a55b:free" and b["tool_choice"] == "auto"
    assert b["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "prompt"}]
    assert b["tools"][0]["type"] == "function" and b["tools"][0]["function"]["parameters"]["properties"] == {
        "region": {"type": "string"}}
    assert b["reasoning"] == {"effort": "medium"}


def test_openrouter_pilot_no_call_and_bad_arguments_are_no_tool_call():
    from avians_adp.sft import decision_kind
    for reply in (_reply(text="I will search r3."), _reply("search_region", raw_args="{not json")):
        d = _or_pilot(FakeTransport(reply)).decide(REQ)
        assert (d.tool, d.reason, d.error) == ("keep", "no tool call", "")
        assert decision_kind(d.tool, d.reason, {"error": d.error}) == "no_tool_call"


def test_openrouter_pilot_errors_are_api_errors():
    from avians_adp.sft import decision_kind
    for reply in ((400, {}, {"error": {"code": 400, "message": "bad request"}}),
                  (200, {}, {"error": {"code": 500, "message": "provider failed"}}),
                  (200, {}, {"choices": []})):
        d = _or_pilot(FakeTransport(reply)).decide(REQ)
        assert d.tool == "keep" and d.error and decision_kind(d.tool, d.reason, {"error": d.error}) == "api_error"


def test_openrouter_pilot_retries_a_short_rate_limit():
    ft = FakeTransport((429, {"Retry-After": "7"}, {"error": {"code": 429, "message": "slow down"}}),
                       (503, {}, {"error": {"code": 503, "message": "overloaded"}}),
                       _reply("keep", {"reason": "searching"}))
    d = _or_pilot(ft).decide(REQ)
    assert d.tool == "keep" and not d.error and ft.slept == [7.0, 10.0] and len(ft.bodies) == 3


def test_openrouter_pilot_stops_calling_after_the_daily_cap():
    import time as _t
    reset_ms = str(int((_t.time() + 6 * 3600) * 1000))
    ft = FakeTransport((429, {"X-RateLimit-Reset": reset_ms}, {"error": {"code": 429, "message": "free-models-per-day"}}))
    p = _or_pilot(ft)
    d1, d2 = p.decide(REQ), p.decide(REQ)
    assert "daily cap" in d1.error and d2.error == d1.error and len(ft.bodies) == 1 and ft.slept == []


def test_openrouter_pilot_needs_a_key(monkeypatch):
    import pytest
    from avians_agent.pilots import OpenRouterPilot
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        OpenRouterPilot("nvidia/nemotron-3-ultra-550b-a55b:free")


def test_nemotron_is_configured():
    p = load_config("pilots.yaml")["pilots"]["nemotron"]
    assert p["provider"] == "openrouter" and p["model"].endswith(":free")
    from avians_agent.loop import latency_key
    lat = load_config("latency.yaml")
    assert p["model"] in lat["models"]
    pilot = _or_pilot(FakeTransport())
    assert latency_key(lat, pilot) == p["model"]                     # what latency-fit prints
    assert latency_key(lat, ScriptedPilot()) == "scripted"
    assert latency_key(lat, AnthropicPilot(client=FakeClient())) == "claude-haiku-5-5"


def test_an_openrouter_episode_records_and_builds(tmp_path):
    """A whole episode through OpenRouterPilot (the scripted policy behind a fake transport): the record
    validates and every decision becomes a training example, as with Claude."""
    from avians_adp import load, validate
    from avians_adp.build import BuildConfig, build

    class ScriptedServer:
        def __init__(self):
            self.policy, self.state = ScriptedPilot(), None

        def post(self, body):
            d = self.policy.decide(DecisionRequest("", "", [], self.state))
            return _reply(d.tool, {**d.args, "reason": d.reason or "continue"}, reasoning=f"thinking about {d.tool}")

    server = ScriptedServer()
    pilot = _or_pilot(FakeTransport())
    pilot._post = server.post
    real_decide = pilot.decide

    def decide(req):
        server.state = req.state
        return real_decide(req)
    pilot.decide = decide
    m = Mission.load("missions/sar_supply.yaml")
    ep = tmp_path / "eps" / "s0000"
    s = run_episode(m, pilot, 0, ep, load_config("agent.yaml"), load_config("latency.yaml"))
    assert s["satisfied"], s["violated"]
    tj = load(ep)
    assert validate(tj) == [] and tj.details["meta"]["pilot"] == "nemotron"
    assert tj.id == "sar_supply_nemotron_s0000"
    r = build(tmp_path / "eps", tmp_path / "out", BuildConfig(ratios=(1.0, 0.0, 0.0), keep_every=1))
    assert r["examples"]["train"] == s["decisions"] and r["removed"]["decisions"] == {}
    row = json.loads((tmp_path / "out" / "pilot" / "train.meta.jsonl").read_text().splitlines()[0])
    assert row["model"] == "nvidia/nemotron-3-ultra-550b-a55b:free"


def test_openrouter_pilot_never_raises():
    """Review A1: a dropped connection mid-body, a non-object body or any surprise is an API error, not a crash."""
    import http.client
    from avians_adp.sft import decision_kind

    def incomplete(body):
        raise http.client.IncompleteRead(b"partial")

    def boom(body):
        raise KeyError("surprise")
    for post in (incomplete, boom, lambda body: (200, {}, [])):
        ft = FakeTransport()
        p = _or_pilot(ft, retries=1)
        p._post = post
        d = p.decide(REQ)
        assert d.tool == "keep" and d.error and decision_kind(d.tool, d.reason, {"error": d.error}) == "api_error"


def test_openrouter_pilot_provider_error_inside_a_choice_is_an_api_error():
    """Review A2: HTTP 200 with choices[0].error / finish_reason "error" is the provider failing, not the model."""
    from avians_adp.sft import decision_kind
    reply = (200, {}, {"choices": [{"message": {"content": ""}, "finish_reason": "error",
                                    "error": {"code": 502, "message": "upstream died"}}]})
    d = _or_pilot(FakeTransport(reply)).decide(REQ)
    assert "upstream died" in d.error and decision_kind(d.tool, d.reason, {"error": d.error}) == "api_error"


def test_openrouter_pilot_call_without_a_name_is_no_tool_call():
    """Review A3: a nameless call must not become a deliberate teacher keep."""
    from avians_adp.sft import decision_kind
    status, headers, data = _reply("x", {"reason": "r"})
    data["choices"][0]["message"]["tool_calls"][0]["function"]["name"] = None
    d = _or_pilot(FakeTransport((status, headers, data))).decide(REQ)
    assert (d.tool, d.reason) == ("keep", "no tool call")
    assert decision_kind(d.tool, d.reason, {"error": d.error}) == "no_tool_call"


def test_openrouter_wall_time_excludes_retries():
    """Review A4: wall_s (what latency-fit reads) times the successful attempt, not the rate-limit waits."""
    import time as _t
    ft = FakeTransport((429, {"Retry-After": "0"}, {"error": {"code": 429, "message": "slow"}}), _reply("keep", {}))
    p = _or_pilot(ft)
    first = [True]

    def slow_first(body):
        if first[0]:
            first[0] = False
            _t.sleep(0.3)
        return ft.post(body)
    p._post = slow_first
    d = p.decide(REQ)
    assert d.tool == "keep" and not d.error and d.wall_s < 0.2


def test_openrouter_daily_cap_without_headers_and_reset_in_seconds():
    """Review A5: a per-day 429 with no wait header, and X-RateLimit-Reset in seconds."""
    import time as _t
    ft = FakeTransport((429, {}, {"error": {"code": 429, "message": "Rate limit exceeded: free-models-per-day"}}))
    p = _or_pilot(ft)
    assert "daily cap" in p.decide(REQ).error and len(ft.bodies) == 1 and ft.slept == []
    ft = FakeTransport((429, {"X-RateLimit-Reset": str(int(_t.time() + 7200))}, {"error": {"code": 429, "message": "x"}}))
    p = _or_pilot(ft)
    assert "daily cap" in p.decide(REQ).error and ft.slept == []


def test_openrouter_retries_more_codes_and_connection_errors():
    """Review A6: 524 / 500 / a string code / a refused connection are retried."""
    import urllib.error
    ft = FakeTransport((524, {}, {"error": {"code": 524, "message": "timeout"}}),
                       (200, {}, {"error": {"code": "500", "message": "internal"}}),
                       _reply("keep", {}))
    calls = [0]

    def post(body):
        calls[0] += 1
        if calls[0] == 1:
            raise urllib.error.URLError("connection refused")
        return ft.post(body)
    p = _or_pilot(ft)
    p._post = post
    d = p.decide(REQ)
    assert d.tool == "keep" and not d.error and len(ft.slept) == 3 and calls[0] == 4


def test_openrouter_timeout_is_retried_once_only():
    """Re-review: a timeout retried at full backoff could block one decision for ~26 min and spend quota."""
    ft = FakeTransport()
    tries = [0]

    def timeout(body):
        tries[0] += 1
        raise TimeoutError("timed out")
    p = _or_pilot(ft)
    p._post = timeout
    d = p.decide(REQ)
    assert d.reason == "API unreachable" and tries[0] == 2 and ft.slept == [5.0]


def test_daily_cap_shows_in_the_summary_and_stops_a_sweep(tmp_path, monkeypatch):
    """Review A7 guard: the summary counts API errors and flags the daily cap; a 1-worker sweep stops there."""
    from avians_agent import rollouts
    capped = (429, {}, {"error": {"code": 429, "message": "Rate limit exceeded: free-models-per-day"}})
    p = _or_pilot(FakeTransport(capped))
    s = run_episode(Mission.load("missions/sar_supply.yaml"), p, 0, tmp_path / "ep",
                    load_config("agent.yaml"), load_config("latency.yaml"))
    assert s["daily_cap"] is True and s["api_errors"] == s["decisions"] > 0
    assert "| API errors |" in rollouts.table([s]) and "(daily cap)" in rollouts.table([s])
    flown = []

    def fake_one(job):
        flown.append(job[2])
        return {**s, "seed": job[2]}
    monkeypatch.setattr(rollouts, "_one", fake_one)
    res = rollouts.run_many("missions/sar_supply.yaml", "nemotron", [0, 1, 2], tmp_path / "sweep", workers=1)
    assert flown == [0] and len(res) == 1
