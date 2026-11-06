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
