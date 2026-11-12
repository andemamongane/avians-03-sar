# avians_adp: episodes as training data (owner: McGlashan)

Every episode is recorded as an **Agent Data Protocol** trajectory (Song et
al., *Agent Data Protocol*, arXiv 2510.24702). ADP is a standard
interlingua: actions (API calls, messages) alternating with observations.
Converters exist from it to the formats training frameworks expect. This
package owns that record and turns it into datasets: one to distil the
teacher into an onboard model, and one to train the router.

```python
from avians_adp import EpisodeRecorder, load
traj = load("data/episodes/sar_supply/haiku/s0003")   # pydantic Trajectory
for step in traj.content:
    step.class_    # text_observation | message_action | api_action
```

## What an episode contains (written by `avians_agent.loop`)

The record follows the standardized ADP schema 1.3.1 (`schema.py` mirrors
`neulab/agent-data-protocol@274f935a`; upstream has since moved to ATIF, an
export for which can be added later). Each decision point is four steps:

| ADP step | avians meaning | `details` |
|---|---|---|
| `TextObservation(source="environment")` | the decision prompt: self-contained (state, events, memory, notes, last decisions) | `t`, `decision` |
| `MessageAction` | the pilot's visible text; `reasoning_content` = its thinking summary | `t`, `decision` |
| `ApiAction(function, kwargs, description, tool_call_id)` | the tool call: a skill or `keep`; description = its reason | `t_obs`, `t_apply`, `latency_s`, `wall_s`, `model`, `usage`, `by` (local/escalate), `route`, `state` (structured snapshot), `notes`, `decision` |
| `TextObservation(source="environment", tool_call_id)` | what the skill replied (accepted / rejected and why); the id links it to its call | `accepted`, `decision` |

`Trajectory.available_apis` lists the tool names. `decisions(traj)` groups
the steps per decision point; `validate(traj)` checks a record (every
`finish()` runs it and warns), and `python -m avians_adp validate <dir>`
checks a folder. Records written before `schema_version` existed
(`source="tool"`, no call ids) are upgraded on load.

`Trajectory.details` holds the system prompt, the tool definitions, the meta
(seed, pilot, scenario truth), the final STL score and the summary. Next to
it, `signals.jsonl`, `events.jsonl` and `score.json` hold the per-step data
for rewards.

This is storage option A: the full growing history. The pilot itself sees
option C: a window plus structured memory. Each prompt already contains
everything needed, so one decision makes one training example.

## Building datasets

```bash
python -m avians_adp.build data/episodes/sar_supply/haiku data/datasets/pilot_v1 --satisfied
```

Each decision with an action becomes one example: the system prompt, the
tools and the decision prompt in; the teacher's tool call out, with
`reason` (and `notes`) put back into its arguments. The pipeline, in order:

1. **Validate.** Every `trajectory.json` under the folder; invalid or
   unreadable ones are skipped and reported.
2. **Episodes.** `--satisfied`, `--min-robustness R`,
   `--require-clauses battery,deliver` (each satisfied, with robustness at
   least `--clause-margin`), `--pilot haiku`, `--tier fast`. A clause name
   no episode has is an error.
3. **Split by seed.** `--split 80/10/10` (the default). A seed's split comes
   from a hash of the seed alone: every episode of a seed (any pilot) lands
   in the same split, and adding episodes never moves an old seed.
4. **Decisions.** Dropped by default, each counted: keeps the teacher never
   chose (`api_error`, `no_tool_call`, `router_commit`), `rejected` calls and
   `unanswered` ones (the episode ended first). `--allow rejected,...` keeps
   some anyway; `--model claude-haiku-5-5` keeps only that model's calls.
5. **Thin (train only).** In a run of consecutive `keep` decisions with the
   same skill, skill status, mission facts, closed regions and battery band
   (`--battery-band`, default 25 %; 0 ignores battery), keep 1 in
   `--keep-every` (default 3). Val and test are not thinned.

| File (per split: `train`, `val`, `test`) | Layout |
|---|---|
| `pilot/<split>.jsonl` | TRL: `messages` (system, user, assistant + `tool_calls`) and `tools` |
| `pilot/<split>.meta.jsonl` | `id` + `meta` per row (episode, seed, kind, model, accepted, robustness, ...), same order |
| `pilot_lf/<split>.jsonl`, `pilot_lf/dataset_info.json` | LLaMA-Factory sharegpt: `dataset_dir: pilot_lf`, `dataset: avians_pilot_train`, `eval_dataset: avians_pilot_val`, `template: qwen3` |

The command prints what each stage removed and warns about an empty split
(likely with only a few seeds). `--text` and `--thinking` add the pilot's
visible text and thinking summary to the TRL layout (the sharegpt
`function_call` turn has no place for them). The output folder must not
contain the episodes, and must be empty unless `--force` is given, which
replaces only `pilot/` and `pilot_lf/`.

**Sequence length.** With the Qwen3 tokenizer an example is 6.1k–7.1k
tokens (system prompt about 1.8k, the 23 tools about 3.9k, the decision
prompt about 0.9k), and the target is only about 25 tokens at the very end.
Set `cutoff_len` (LLaMA-Factory) or `max_length` (TRL) to at least 8192: the
common 2048 truncates the target away and trains on nothing.

## Your tasks

1. **Schema:** check `schema.py` against the upstream ADP repository (field
   names, the type discriminator, the required fields). Import from it if it
   is pip-installable, keeping the avians fields in `details`. Add a
   validator that every saved trajectory passes.
2. **Pilot SFT dataset** (`sft.pilot_examples`): one example per decision:
   system prompt, tools and decision prompt in; the teacher's tool call (with
   its reason) out. Write chat-format JSONL that LLaMA-Factory and TRL accept,
   in the tool-calling layout of the student model's chat template (Qwen3.x,
   if the student is Qwen-based like Kev).
3. **Filters and splits:** by episode robustness or by clause (train on
   decisions from flights that kept the spec), de-duplicate near-identical
   `keep` decisions, and split train/val/test by seed (never one episode
   across splits).
4. **Router dataset** (`sft.router_examples`): label each decision point
   commit / local / escalate. Commit means the teacher kept going. Local
   means the student's call (run offline on the same prompt) matches the
   teacher's. Escalate means it doesn't. Later, label from outcomes too (which
   choice led to higher robustness).
5. **Manifests:** each dataset build writes counts, sources (episode ids),
   the git commit and the filter settings.
6. **Scale:** `avians rollouts ... --pilot haiku --seeds 0-999 --workers 16`
   costs about $0.02 per episode. Coordinate with Andema before large runs.

The skipped tests in `tests/test_todo.py` mark tasks 2 and 4. Turn them into
real tests as you implement them.
