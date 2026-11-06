# avians_agent: the pilot loop and the `avians` command

- **`loop.py`:** `run_episode`. The simulator runs continuously. Decision
  points come from events and from a `wake_s` timer. Each decision applies
  after a simulated latency (fast tier) or its real latency (Gazebo tier,
  where the call runs in a thread). The online monitor runs alongside, and
  everything is recorded through `avians_adp`.
- **`prompts.py`:** the system prompt (mission brief, STL spec, signals,
  world; cacheable) and the self-contained decision prompt. The tools are
  the skills plus `keep`, each with optional `reason` and `notes`.
- **`pilots.py`:** three pilots.
  - `AnthropicPilot` (the teacher, `claude-haiku-5-5`): adaptive thinking with
    summarized display, the system prompt cached, and one tool call per
    decision.
  - `ScriptedPilot`: the rule-based baseline, and the no-key dry run.
  - `snapshot()`: the structured state logged with every decision.
- **`memory.py`:** structured mission memory, the pilot's notes, and recent
  decisions.
- **`latency.py`:** lognormal latency models per model, and fitting them
  from logged wall times.
- **`router.py`:** System One routing (commit / local / escalate). There is a
  fixed router, and a `SystemOneRouter` HTTP client whose request format is
  still to be checked against the API reference. Kev, Laya or Jev can sit
  behind it.
- **`rollouts.py`:** parallel sweeps with a results table.
- **`cli.py`:** `avians check | fly | rollouts | score | video | qgc |
  gazebo-replay | latency-fit | doctor`.
- **`mission.py`:** a mission file becomes its world, spec, limits and
  horizon.
