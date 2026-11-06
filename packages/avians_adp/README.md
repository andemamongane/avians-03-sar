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

| ADP step | avians meaning | `details` |
|---|---|---|
| `TextObservation(source="environment")` | the decision prompt: self-contained (state, events, memory, notes, last decisions) | `t`, `decision` |
| `MessageAction` | the pilot's visible text | `thinking` (summary), `decision` |
| `APIAction(function, kwargs, description)` | the tool call: a skill or `keep`; description = its reason | `t_obs`, `t_apply`, `latency_s`, `wall_s`, `model`, `usage`, `by` (local/escalate), `route`, `state` (structured snapshot), `notes` |
| `TextObservation(source="tool")` | what the skill replied (accepted / rejected and why) | `accepted` |

`Trajectory.details` holds the system prompt, the tool definitions, the meta
(seed, pilot, scenario truth), the final STL score and the summary. Next to
it, `signals.jsonl`, `events.jsonl` and `score.json` hold the per-step data
for rewards.

This is storage option A: the full growing history. The pilot itself sees
option C: a window plus structured memory. Each prompt already contains
everything needed, so one decision makes one training example.

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
