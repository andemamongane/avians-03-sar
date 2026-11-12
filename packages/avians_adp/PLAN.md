# avians_adp: implementation plan (McGlashan), revision 2

Branch: `mcglashan/adp-datasets` → pull requests into `main` (Andema merges).
Source of tasks: "AVIANS stage 3: notebook for Taylor and McGlashan", section "McGlashan: the data".
This file was the working plan and is kept as the record of what was planned and what was executed
(Austin's decision at Phase 5, replacing step 5.6c "remove this file"). The plan sections are as written
before coding; the **Status** section at the end records what happened, phase by phase.

**Done means** (from the notebook):
- `python -m avians_adp.build` turns a folder of episodes into train/val/test JSONL files plus a manifest.
- LLaMA-Factory loads those files without edits.
- The two skipped tests in `tests/test_todo.py` are real, passing tests.
- `python -m pytest` passes at the repo root.

Rationale notes (*Changed*, *Added*, *Removed*) say how a step differs from revision 1 and why.

## Facts the plan rests on (all checked)

**F1. Upstream ADP changed after the paper.** The repo is
[neulab/agent-data-protocol](https://github.com/neulab/agent-data-protocol).
- On 2026-06-14 (#256) its main schema became ATIF-v1.7 (`schema/atif.py`).
- The paper-style classes were deleted. Their last version is ADP 1.3.1, at commit `274f935a`.
- It cannot be imported: not on PyPI, needs Python ≥ 3.12 (we support 3.10), and installs top-level
  packages named `schema`, `agents` and `scripts`.

**F2. Our record differs from ADP 1.3.1 in these ways:**

| Ours today | ADP 1.3.1 |
|---|---|
| `TextObservation.source="tool"` for skill replies | `source` is `user`, `agent` or `environment`. A tool result is `environment` plus a `tool_call_id`. |
| No `tool_call_id` | Required on an `ApiAction` directly followed by an observation. Each id must match exactly one later observation. |
| Thinking summary in `details.thinking` | `reasoning_content` on the action |
| No `schema_version`, `available_apis`, `reward` | `schema_version="1.3.1"`, optional `available_apis`, optional `reward` |
| Class `APIAction` | Class `ApiAction` |
| `details` on every step | Steps forbid extra fields |

**F3. Three readers outside this package parse `trajectory.json` as raw JSON:**
- `avians_viz/episode.py` reads `step["details"]`, `function`, `kwargs`, `description` on each `api_action`.
- `avians_agent/cli.py` (`latency-fit`) reads `details.wall_s` and `details.model`.
- `avians_agent/tests/test_agent.py` counts `api_action` steps.

None reads `source` or `details.thinking`. So per-step `details` must stay where it is.

**F4. How the pilot loop writes a decision** (`avians_agent/loop.py`):
- Order: prompt observation, then a `MessageAction` only if there is text or thinking, then the
  `APIAction`, then the result observation.
- The `APIAction` has no `decision` number in its `details`. The prompt and the result do.
- An episode can end mid-decision. The record can then end with a prompt that has no action, or an
  action that has no result.
- `reason` and `notes` are split out of the tool arguments before recording.
- Three kinds of recorded action are not real teacher calls:
  - API errors: `keep` with `details.error` set.
  - No tool call in the reply: `keep` with reason `"no tool call"` and an empty `error`.
  - Router commits: `keep` with `model == "router"`.
- If the teacher made several calls, only the first is kept and the reason gains
  `" (ignored N more call(s))"`.
- Without a router, `details.by` is always `"local"`, even for the teacher.

**F5. `events.jsonl` does not record which events woke the pilot.** It has `t`, `kind` and `text` only.

**F6. Dependencies run one way.** `avians_agent` depends on `avians_adp`. So `avians_adp` must not import
`avians_agent` at module level.

**F7. Measured on 6 scripted episodes (seeds 0–5):**
- 19–35 decisions per episode; 55–62 % are `keep`.
- The step pattern is prompt, action, result. There are no `MessageAction` steps and no rejected calls,
  and `usage` is empty.
- So scripted episodes do not exercise the message, thinking, rejection or stop-reason paths.

**F8. Training formats:**
- LLaMA-Factory's sharegpt format takes `conversations`, `system` and `tools` columns. `human` and
  `observation` sit at odd positions; `gpt` and `function_call` at even positions. Its OpenAI-style
  format does not document tool calls, so we use sharegpt.
- Qwen3's chat template prints each tool as whole JSON, and accepts tool-call arguments as a dict or as
  a JSON string.

**F9. This machine** has Python 3.14 only (no 3.10, no `uv`). `.venv` is installed; the suite gives
36 passed, 2 skipped.

## Decisions

Settled by the facts:

- **D1. Per-step `details` stays on the steps.** *Changed:* revision 1 offered moving it into
  `Trajectory.details`. F3 rules that out: it would break `avians_viz` and `latency-fit`.
- **D2. Mirror ADP 1.3.1 locally; do not import upstream.** Pin the commit in a comment (F1).
- **D3. The validator never blocks saving an episode.** *Changed:* revision 1 had `finish()` raise "in
  tests" and warn "in production", with no way to tell the two apart. A raise could also lose a paid
  Haiku episode. Now `finish()` always writes, then validates and warns.
- **D4. `avians_adp` takes `avians_agent` functions as arguments.** *Changed:* revision 1 both used
  `avians_agent.router.features` and promised not to import `avians_agent` (F6).

Working assumptions, to confirm with Andema (step 0.1). None blocks the start:

- **A1. Schema target is ADP 1.3.1, with ATIF export later.** The notebook's wording (field names, type
  discriminator) describes the paper-style schema. If Andema prefers ATIF, only step 1.7 grows.
- **A2. Student chat template is Qwen3.** The notebook lists the student model as open. Examples are
  stored in a neutral form, so a different template means a new writer, not a rebuild.

## Risks

| Risk | Where | Safeguard |
|---|---|---|
| **Highest: changing the on-disk record.** It is a cross-package contract, and Haiku episodes cost money. | Phase 1 | Legacy fixture saved first (1.1). Old records still load (1.3). Recorder signatures unchanged. Other readers re-tested (1.8). Own pull request, so it can be reverted alone. |
| Training files fail to load, or render wrongly, in the real tools. | Phases 2, 5 | Format spike before any builder code (0.2). Real-loader gates (2.6, 5.4). |
| Mixed tool arguments get padded with nulls by the `datasets` loader. | 0.2 | The spike measures it and picks the encoding. |
| Rebuilt router features differ from what the live router saw. | 4.2 | A test compares them against a live run. |
| Small episode counts leave a split empty. | 3.3 | The build warns and records it in the manifest. |
| Code is untested on Python 3.10 (F9). | All | Avoid features newer than 3.10 and newer than the `pydantic>=2` pin. Said in the pull request. |
| Gazebo-tier records are untested here. | 1.5 | Decisions are grouped by position, not by timing. Said in the pull request. |

**External dependencies:**

| Need | From | Blocks |
|---|---|---|
| Anthropic API key, and approval above 100 episodes | Andema | Gates 2.6c and 5.5 (real teacher data). Nothing before. |
| Student model choice | Andema (open in the notebook log) | Final chat template (A2); real router labels (4.5). |
| A machine that can install LLaMA-Factory | Linux GPU machine, if this one cannot | Gate 5.4 |
| Network access to GitHub and Hugging Face | — | Gates 1.7 and 0.2 |

**Steps that need your go-ahead each time:**
- Any commit, push or pull request.
- Any Haiku run (it spends money).
- Overwriting an existing dataset folder (the build refuses without `--force`).
- Any edit outside `packages/avians_adp`. None is planned.

## Order and parallel work

Phases run 0 → 5, following the notebook's task order. Within that:

- 0.1 (questions) goes out first: it has the longest wait and blocks nothing.
- 0.2 (format spike) can run alongside Phase 1. It must finish before 2.3.
- 3.1–3.3 (filters, thinning, splits) are pure functions. They can be written alongside Phase 2.
- 4.1–4.3 (router labelling) needs no real student. Only 4.5 waits on one.
- `build.py` grows phase by phase (2.5, 3.5, 4.4, 5.1), so there is an end-to-end run to check from
  Phase 2 on. *Changed:* revision 1 built it only at the end, leaving nothing runnable until then.

---

## Phase 0: before coding

**0.1 Send Andema the open questions.**
- a. Confirm A1 (ADP 1.3.1 now, ATIF later).
- b. Confirm A2 (Qwen3 template), or name the student model.
- c. Ask for an API key and a budget of about 10 Haiku episodes (about $0.20) for gates 2.6c and 5.5.

*Added:* these were listed in revision 1 but never sent or tied to a step.

**0.2 Format spike** (scratch virtualenv; nothing committed).
- a. Hand-write 3 examples with different tools, in both layouts (step 2.3).
- b. Load the TRL file with `datasets.load_dataset("json")`. Render each row with the Qwen3 tokenizer's
  `apply_chat_template(messages, tools=...)`.
- c. Check the rendered text: one `<tool_call>` block; no `null` padding in arguments or tool schemas.
- d. If padding appears, switch the affected column to a JSON string and re-check.
- e. Try installing LLaMA-Factory here. If it installs, load the sharegpt file with its loader now. If
  not, record that gate 5.4 must run on the Linux machine.

Done when: the encoding of `arguments` and `tools` is chosen and written into step 2.3.
*Added:* revision 1 deferred all format checks to an optional step at the end, after the code that
depends on them.

## Phase 1: schema and validator (notebook task 1; own pull request)

**1.1 Save a legacy fixture first.** Copy one current episode's `trajectory.json`, cut to 3 decisions,
into `tests/fixtures/legacy_trajectory.json`.
*Added:* after 1.4 the recorder can no longer produce the old format, so this must come first.

**1.2 Rewrite `schema.py` to mirror ADP 1.3.1.**
- a. Classes `ApiAction`, `MessageAction`, `TextObservation`, `Trajectory`; keep `APIAction` as an alias.
- b. New fields: `tool_call_id`, `reasoning_content`, `reward`, `schema_version`, `available_apis`.
- c. Keep `details` on steps (D1) and the `class_` discriminator.
- d. Leave optional upstream fields out of the JSON when empty. Use a serializer, not `exclude_if`
  (it needs a newer pydantic than our pin).
- e. Pin `neulab/agent-data-protocol@274f935a` in the module docstring.

**1.3 Upgrade old records on load.** A before-validator on `Trajectory` detects a record with no
`schema_version` and:
- a. maps `source="tool"` to `environment`;
- b. gives each action and the result right after it a shared `tool_call_id`;
- c. moves `details.thinking` to `reasoning_content`;
- d. copies the `decision` number onto each action.

*Changed:* revision 1 mapped `"tool"` only "when there is a `tool_call_id`", but old records have none.

**1.4 Update `recorder.py`.** Method signatures stay the same.
- a. `action()` creates `call_000001`, ...; `tool_result()` reuses the id and writes `source="environment"`.
- b. `observation()` remembers the decision number; `action()` stamps it (F4).
- c. `message()` writes thinking to `reasoning_content`.
- d. `brief()` sets `available_apis` from the tool names.
- e. `finish()` clears the id on a final action that never got a result (F4), writes all files, then
  runs the validator and prints a warning on failure (D3).

**1.5 Add `decisions.py`: `decisions(traj)`.** It groups steps by position into records of prompt,
optional message, action and optional result. It tolerates an unfinished last decision.
*Added:* the SFT builder, the filters and the router builder all need this grouping. Revision 1 left
each to derive it, and assumed actions carried a decision number.

**1.6 Add `validation.py`: `validate(traj) -> list[str]` and `validate_dir(root)`.**
- a. Upstream rules: source values; id links; `available_apis` covers every function called.
- b. Avians rules: step order per decision; decision numbers increase; `t_obs ≤ t_apply`; trajectory
  `details` has `system`, `tools`, `meta`, `score`, `summary`.
- c. An unfinished decision is allowed only at the very end.

**1.7 Check against the real upstream code.**
- a. Add `to_upstream(traj)`: the record without per-step `details`.
- b. Add a test that validates `to_upstream` output with upstream's own `Trajectory`, from a checkout at
  the pinned commit named by `ADP_UPSTREAM`. It skips when the variable is unset.
- c. Run it once and record the result in the pull request.

*Changed:* revision 1 "re-validated against 1.3.1 rules" using our own copy of the rules, which proves
nothing about upstream.

**1.8 Gate.**
- a. Tests: round trip; validator rejects a broken link and a bad source; legacy fixture loads and
  validates; unfinished last decision validates.
- b. Re-record seeds 0–5 (free); `validate_dir` reports no errors.
- c. Other readers still work: `python -m pytest` at the root, and
  `avians fly missions/sar_supply.yaml --seed 3 --video` produces a replay.

Done when: a–c pass and the pull request notes the contract change (format changed, signatures not).
Rollback: revert this pull request. Old episodes stay readable either way (1.3).

## Phase 2: pilot SFT set (notebook task 2)

**2.1 `sft.pilot_examples(traj, *, include_thinking=False) -> list[dict]`.** One neutral example per
decision that has an action:
- a. `system` from `details.system`.
- b. `tools` from `details.tools`, converted to the OpenAI shape
  (`{"type": "function", "function": {name, description, parameters}}`).
- c. `prompt`: the decision prompt.
- d. `call`: the tool name, and arguments with `reason` (and `notes`, if set) put back (F4).
- e. `text` if the pilot wrote any; `thinking` only with the flag.
- f. `id` (`<episode>#d<n>`) and `meta`: episode, seed, pilot, model, `t`, decision, `by`, `error`,
  stop reason, `accepted`, episode robustness.

It does not filter. Phase 3 does, using `meta`.
*Changed:* revision 1 mixed exclusions into this function, which conflicts with "one per decision".

**2.2 Clean the reason.** Strip the `" (ignored N more call(s))"` suffix and set `meta.extra_calls`.
*Added:* otherwise the student learns to write that suffix.

**2.3 `formats.py`: two writers.**
- a. TRL: `messages` (system, user, assistant with `tool_calls`) and `tools`.
- b. LLaMA-Factory sharegpt: `conversations` (`human`, then `function_call`), `system`, `tools`; plus a
  `dataset_info.json` with one entry per split.
- c. Use the `arguments` and `tools` encoding chosen in 0.2.
- d. `meta` goes to a side file `<split>.meta.jsonl`, in the same row order, so the training files hold
  only the columns the tools expect.

**2.4 Replace the first skipped test** with a real `test_pilot_examples_one_per_decision`. Use a
hand-built trajectory with a message, thinking, a rejected call and an unfinished last decision (F7).

**2.5 `build.py`, first version.** `python -m avians_adp.build <episodes> <out>`: find every
`trajectory.json` under the folder, validate, write unsplit files in both layouts.

**2.6 Gate.**
- a. The target parses back to the recorded tool, arguments, reason and notes, for every decision in
  seeds 0–5.
- b. The build output loads with `datasets` and renders with the Qwen3 template (the 0.2 script).
- c. When a key is available: one Haiku episode (about $0.02) builds and renders, covering text and
  thinking.

Done when: a and b pass. Item c is tracked as open until a key arrives.

## Phase 3: filters and splits (notebook task 3)

**3.1 Episode filters** (`filters.py`): `satisfied`; `min_robustness`; `require_clauses` (named clauses
satisfied, or above a margin); `pilot`; `model`; `tier`.

**3.2 Decision filters**, each on by default and counted separately:
- a. drop API errors;
- b. drop "no tool call" keeps;
- c. drop router commits;
- d. drop rejected calls (`accepted` is false) and calls with no result;
- e. select the teacher by `model`, not by `by` (F4).

*Added:* b and d were missing in revision 1. A "no tool call" keep has an empty `error`, so the error
filter alone lets it through.

**3.3 Splits** (`splits.py`).
- a. Assign by seed with a stable hash against the ratios (default 80/10/10). All episodes of one seed
  share a split. Adding episodes never moves a seed.
- b. Fail on duplicate episode ids.
- c. Warn when a split is empty, and record it.

**3.4 Thin repeated keeps** (train split only).
- a. A decision's signature is: skill, skill status, found, payload, released, help arrived, closed
  regions, battery in 10 % bands.
- b. Within a run of consecutive `keep` decisions with one signature, keep the first and every k-th
  after it (`--keep-every`, default 3).
- c. Val and test are never thinned, so they reflect what the pilot really sees.

*Changed:* revision 1 said "never drop a keep right after an event", which F5 makes impossible to
check. The signature rule covers the same intent. Thinning before splitting, or thinning val and test,
was also left open.

**3.5 Wire into the build**, in this order: find → validate → episode filters → split → decision
filters → thin train → write.

**3.6 Gate.** Tests:
- no seed in two splits;
- ratios hold over 1000 made-up seeds;
- thinning keeps every non-keep and the first keep of each run;
- each filter drops exactly its hand-built case.

Then a build on seeds 0–5 with `--satisfied` uses only the satisfied episodes.

## Phase 4: router set (notebook task 4)

**4.1 `sft.router_examples(teacher, student_calls, *, features=None, tolerances=None) -> list[dict]`.**
- a. `student_calls` is one `{decision, tool, args}` per decision.
- b. Label `commit` if the teacher called `keep`. Otherwise `local` if the student's call matches,
  else `escalate`. A student calling `escalate`, or missing for that decision, gives `escalate`.
- c. A match is the same tool, and equal arguments ignoring `reason` and `notes`. Numbers match within
  a per-argument tolerance; the defaults live in one table and are copied into the manifest.
- d. Skip the non-teacher decisions of 3.2 a–c. Never thin: every keep is a `commit` example.
- e. Each example holds the label, the choices, the structured state, the event kinds since the last
  decision, and `text` when a `features` function is given.

**4.2 Rebuild the router's input.**
- a. Event kinds come from `events.jsonl`, for times after the previous decision up to this one.
- b. The build imports `avians_agent.router.features` lazily and passes it in (D4).
- c. Test: run a scripted episode with a router that records the features it is given. The rebuilt
  strings must equal the recorded ones. Skip if `avians_agent` is absent.

*Added:* revision 1 assumed the rebuild would match without checking it.

**4.3 `student_calls.py`.** Run a `decide(system, prompt, tools, state)` function over a trajectory's
decisions and write `student_calls.jsonl` beside the episode.

**4.4 Replace the second skipped test** with a real `test_router_examples` (hand-built keep, match,
near-match within tolerance, mismatch). Wire the build: write `router/<split>.jsonl` for episodes that
have `student_calls.jsonl`, with the same seed splits.

**4.5 Gate.** Tests pass. A build over seeds 0–5, using the scripted pilot as a stand-in student,
writes router files with all three labels counted in the output.

Done when: 4.5 passes. Real labels wait on the student model; say so in the pull request.

## Phase 5: manifests and finish (notebook task 5)

**5.1 `manifest.json`**, written by every build:
- a. counts per split, per tool and per pilot, and the count each filter removed;
- b. episode ids per split, and episodes skipped with the reason (invalid, unfinished, filtered);
- c. the git commit of the build and whether the tree had changes (null outside a git checkout);
- d. the full filter, split, thinning and tolerance settings;
- e. the ADP schema version and the format encoding from 0.2;
- f. a sha256 and line count per output file; the creation time.

**5.2 Safe writes.** Build into a temporary folder and rename at the end. Refuse a non-empty output
folder without `--force`. The default output is under `data/datasets/`, which git ignores.
*Added:* revision 1 had no protection against overwriting a dataset or leaving half of one.

**5.3 Tests.** Build 6 scripted episodes in a temporary folder. Manifest counts equal file line counts;
hashes match; a second build with the same settings gives identical data files.

**5.4 LLaMA-Factory gate.** Load the built `pilot_lf` set with LLaMA-Factory's own loader, using only
`dataset_dir` and the dataset names, with the Qwen3 template. Run it where 0.2e found it installs.
*Changed:* revision 1 marked this optional, but it is part of "done".

**5.5 Real-data gate** (needs the key and your go-ahead). About 10 Haiku episodes; full build; read
5 rendered examples by eye.

**5.6 Finish.**
- a. Delete `tests/test_todo.py` (its tests now live in the real test files).
- b. Update the package `README.md`: the record changes, a "Building datasets" section, the task list.
- c. Remove this `PLAN.md`.
- d. `python -m pytest` at the root.

Done when: every item under "Done means" holds and gates 5.4 and 5.5 are passed or listed as open in
the pull request.

## Not in this plan

- **ATIF export.** Later, per A1.
- **Labelling the router from outcomes** (which choice scored higher). The package README marks it "later".
- **Large Haiku rollouts** (README task 6). Needs Andema's approval.
- **Saving the record during a flight.** Today a crash mid-episode loses `trajectory.json`. Worth
  raising with Andema, but it is not one of the five tasks.

## Status (2026-11-11)

**Phase 0:** A1 and A2 settled by Austin (ADP 1.3.1 now, ATIF later; Qwen3 template). 0.1c (API key,
Haiku budget) still to ask Andema. 0.2 (format spike) run on 2026-11-11 in `~/.cache/avians_adp_spike` (the session scratchpad is a
3.8 GB tmpfs; transformers overflowed it). Results, with `datasets` 4.x, `transformers` and the
`Qwen/Qwen3-4B` tokenizer:
- TRL layout (`messages` + `tools`) loads with no null padding, whether `arguments` is a dict or a JSON
  string; both render the identical `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` block.
- LLaMA-Factory sharegpt layout (`conversations`, `system`, `tools` as a JSON string) loads clean.
- **Encoding chosen for step 2.3:** `arguments` as a JSON string (the Qwen3 template has an
  `is string` branch, and a string cannot hit Arrow type conflicts across 23 tools with different
  argument types); `tools` as a list of dicts, re-checked at gate 2.6b with the real 23 tools, with a
  JSON-string fallback if padding appears there.
- 0.2e: see the LLaMA-Factory line in the status below.

- 0.2e: `pip install --dry-run llamafactory` resolves in the spike venv (Python 3.14, real disk), so
  gate 5.4 can run on this machine; the real install (torch, several GB) is left until Phase 5.

**Phase 1: done, gate passed.** Nothing committed yet.
- 1.1 `tests/fixtures/legacy_trajectory.json` (3 decisions of scripted seed 0, old format).
- 1.2–1.3 `schema.py`: ADP 1.3.1 classes, `APIAction` alias, legacy upgrade in a before-validator,
  upstream's tool-call-link rules in an after-validator, a serializer that drops empty optionals.
- 1.4 `recorder.py`: same signatures; ids, `source="environment"`, `reasoning_content`, decision number
  on the action, `available_apis`; `finish()` writes, then validates and warns.
- 1.5 `decisions.py`, 1.6 `validation.py` (+ `python -m avians_adp validate <dir>` in `__main__.py`;
  the CLI lives there because a module imported by `__init__` cannot be run with `-m` cleanly).
- 1.7 `test_against_real_upstream_models` ran against a checkout of `neulab/agent-data-protocol@274f935a`
  (`ADP_UPSTREAM=<checkout>`): our fixture, a hand-built record and all six real episodes pass upstream's
  own `Trajectory`; a broken link is rejected by both sides. Skipped when the variable is unset.
- 1.8 Seeds 0–5 re-recorded and validated; root suite 54 passed, 2 skipped (`test_todo.py`);
  `avians fly --seed 3 --video` renders `replay.mp4` from the new record; `avians latency-fit` runs.
- README table updated to the new record.
- Pull request notes to write: the record format changed (ids, source, reasoning_content,
  schema_version, available_apis), signatures did not; old records upgrade on load; untested on
  Python 3.10 and on Gazebo-tier records.

**Phase 1 committed** as `1633daa` on `mcglashan/adp-datasets` (PLAN.md left out of the commit).

**Phase 2: gates a and b passed; c open (needs an API key).**
- 2.1–2.2 `sft.py`: `pilot_examples` (layout-neutral, unfiltered, `meta` for Phase 3), `openai_tools`,
  `clean_reason`, `target_arguments`. Thinking comes from the action, else from the message.
- 2.3 `formats.py`: `to_trl` (arguments as a JSON string), `to_sharegpt`, `dataset_info.json`, meta side
  file. Text and thinking are TRL-only flags: a sharegpt `function_call` turn has no place for them.
- 2.4 `tests/test_sft.py`: the real `test_pilot_examples_one_per_decision` (message, thinking, notes,
  rejected call, extra-calls suffix, API-error keep, unfinished last decision); `test_todo.py` keeps only
  the router test.
- 2.5 `build.py` (`python -m avians_adp.build <episodes> <out> [--force] [--text] [--thinking]`). It
  already refuses a non-empty output and duplicate episode ids (pulled forward from 3.3b / 5.2).
- 2.6a: 156 examples from seeds 0–5 (= 156 decisions); every target round-trips in both layouts. Also a
  test that flies a fresh scripted episode and checks the round trip.
- 2.6b (`~/.cache/avians_adp_spike/gate_2_6b.py <build>`): both layouts load with `datasets`, no null
  padding with the real 23 tools, loaded rows equal written rows (156/156); the Qwen3 render has exactly
  one `<tool_call>` in the assistant turn, the right call, and a tools block equal to the written tools
  (156/156). So `tools` stays a list (no JSON-string fallback needed).
- Independent review (Fable subagent) and fixes:
  - `--force` removed the whole output folder, deleting episodes stored under it (confirmed). Now the
    build refuses an output folder that contains the episodes, and `--force` removes only `pilot/` and
    `pilot_lf/`.
  - `meta.kind` (`call` / `api_error` / `no_tool_call` / `router_commit`) marks keeps the teacher never
    chose; Phase 3.2 filters on it instead of re-deriving.
  - An empty `reason` is no longer written into the target (the teacher did not send one).
  - Unreadable trajectories are skipped, not fatal; an output path that is a file is a clean error.
  - Rejected: "datasets pads `tools` with nulls". Gate 2.6b measured 0 null paths on the real 23 tools
    (datasets 4.x). Added `test_files_load_unchanged_with_datasets`, which runs where `datasets` is
    installed (it passes in the spike venv; skipped in `.venv`).
- **Finding:** an example is 6.1k–7.1k Qwen3 tokens (system ~1.8k, tools ~3.9k, prompt ~0.9k); the
  target is ~25. Training needs `cutoff_len`/`max_length` ≥ 8192. In the README; add token stats to the
  manifest in 5.1 if a tokenizer is available.

**Phase 2 committed** as `c3a85a0`.

**Phase 3: done, gate passed.**
- 3.1–3.2 `filters.py`: `EpisodeFilter` (satisfied, min robustness, required clauses with margin, pilot,
  tier; an unknown clause name is an error), `DecisionFilter` (drops `meta.kind` api_error /
  no_tool_call / router_commit, rejected, unanswered; `allow` keeps some; `models`), `Counts`.
- 3.3 `splits.py`: sha256 of the seed against the ratios; seeds must be integers; all pilots of a seed
  share a split; pinned values in a test so a hash change cannot silently reshuffle datasets.
- 3.4 `thin_keeps`: train only, runs of identical keeps per episode, 1 in `keep_every`.
  **Changed from the plan:** battery band 25 %, not 10 % (`--battery-band`, 0 = ignore). Measured on
  seeds 0–5: with 10 % bands the draining battery cuts keep runs into pieces of 2–3, so k barely mattered
  (k=3 and k=10 removed 28 and 29 of 91 keeps); with 25 % bands, 38 and 43.
- 3.5 `build.py`: find → validate → episode filters → split → decision filters → thin train →
  write `train/val/test` in both layouts + `dataset_info.json` with three entries; report of what each
  stage removed; empty-split warning only for splits with a non-zero ratio.
- 3.6 tests (`tests/test_filters.py`, 46 tests). Gate on seeds 0–5 with `--satisfied`: exactly the four
  satisfied episodes (0, 2, 3 in train; 4 in test), train 73 decisions → 55 after thinning, test 35
  unthinned; both split files load unchanged with `datasets` and render with Qwen3 (55/55, 35/35).
- Independent review (Fable subagent) and fixes:
  - Infinite robustness is written as the string "inf"; comparing it crashed `--min-robustness` and
    `--require-clauses`. Values now go through `number()`.
  - `--force` deleted the previous build before validating the options, so a typo cost the old build.
    Deletion now happens only after every input check.
  - NaN/inf ratios crashed `split_of`; non-integer seeds were silently coerced (1.9, True, "07"). Both are
    now errors (an episode with a non-integer seed is counted, not split).
  - A region without an `id` no longer raises; zeroed splits no longer warn; the thinning docstring says
    runs are judged after the decision filters.

**Phase 3 committed** as `ed20f21`.

**Phase 4: done, gate passed (with a stand-in student; real labels wait on the student model).**
- 4.1 `routing.py`: `args_match` (tolerance table `TOLERANCES`, strings exact, bool is not a number),
  `label`, `event_kinds`, `router_examples`; `sft.router_examples` delegates.
- 4.2 **Changed from the plan:** the router's event list is recorded, not rebuilt from a time window.
  The recorder stores on each prompt the event kinds recorded since the previous one; this equals the
  loop's `since` list. Measured: recorded lists reproduce the live router's features 81/81 (seeds 0, 1,
  4); a time window gets ~90 % (an event stamped at the decision time lands on the wrong side). Old
  records fall back to the window and are flagged `events_exact=False`. Test:
  `test_rebuilt_features_equal_what_the_live_router_saw` (recording router, seed 4).
- 4.3 `student_calls.py` (+ CLI `--pilot`): fresh pilot per episode; asks only on the teacher's real
  non-keep calls; adds the `escalate` tool; rows carry the episode id; API errors recorded and skipped.
- 4.4 `test_todo.py` deleted; `tests/test_routing.py` (36 tests). Build writes `router/<split>.jsonl`
  (id, text, label, options) + `.meta.jsonl`, same episodes and splits as the pilot set.
- 4.5 Seeds 0–5 re-recorded (now with event lists). Scripted stand-in student: 91 commit, 65 local,
  0 escalate. That is correct, because the student is a copy of the teacher. A test with a student
  whose search altitude is off by 20 m escalates every search (3 m stays local), so all three labels
  are exercised.
- Found while gating: one shared `ScriptedPilot` carried its search memory across episodes and
  produced 7 false escalates. Fixed (fresh pilot per episode) and covered by a test.
- Independent review (Fable subagent) and fixes:
  - Episodes flown with a router hold the student's own decisions (`by="local"`); both the router and
    the pilot sets treated them as teacher calls. New decision kind `student`, dropped by default.
  - `AnthropicPilot` returns keep + error on an API failure; student calls recorded it as a real keep
    (an all-escalate dataset with no warning). Now error rows are skipped and counted; "no tool call"
    replies are labelled escalate; the CLI stops if every call fails.
  - The student was asked on every decision; now only where the label depends on it (65 of 156 on
    seeds 0–5).
  - The offline student lacked the `escalate` tool it has in flight; added.
  - `student_calls.jsonl` had no provenance; rows now carry the episode id and the build ignores a
    stale file with a warning.
  - Also fixed: `True == 1` counted as an argument match.

**Phase 4 committed** as `7696f57`.

**Phase 5: done; gate 5.4 passed; gate 5.5 open (needs an API key).**
- 5.1 `manifest.py`, written by every build as `manifest.json`. It records:
  - source: the episodes folder, the number found and loaded, skipped episodes with reasons;
  - per split: episode ids, counts by tool, pilot and model, and what each filter removed;
  - the router labels per split;
  - the full config (filters, ratios, keep_every, battery band, tolerances), ADP schema version and
    upstream pin, file encoding plus `chat_template: qwen3`;
  - git commit, branch and dirty paths (scoped to `packages/`, untracked included), the command line,
    the creation time;
  - sha256, bytes and rows per file, and a `data_digest` over all files.
- 5.2 Safe writes, **changed after review**. The first version staged beside `<out>`, deleted the old
  build, then moved. The Fable review reproduced that a failure during the moves, or `--force` into an
  unwritable folder, lost both builds, and that a symlinked or mounted `<out>` failed with a
  cross-device error. Now:
  - staging happens inside `<out>`;
  - the swap renames the old build aside, moves the new one in (manifest last), and rolls back on any
    failure;
  - leftovers of a killed build are swept (stage) or reported (aside).
  - `<out>` may not contain, or lie inside, the episodes.
  - The default `<out>` is `data/datasets/build-<UTC>` under the repository root.
- 5.3 `tests/test_manifest.py` (19 tests):
  - the files match the manifest (sha256, rows, every file listed);
  - every decision is accounted for (examples + removed + thinned);
  - the same inputs give the same `data_digest`;
  - a failure at any of four points of the swap restores the previous build byte for byte;
  - an unwritable `<out>` touches nothing;
  - a symlinked `<out>` works;
  - git dirtiness counts untracked code but not edits outside `packages/`.
- 5.4 **Passed** with LLaMA-Factory 0.9.5 (`~/.cache/avians_adp_spike/gate_5_4.py <build> <dataset>`):
  - Ran with `get_dataset`, setting only `dataset_dir`, the dataset name, `template: qwen3` and
    `cutoff_len: 8192`.
  - Loaded 91/91 train rows and 35/35 test rows. In every row the supervised tokens are exactly the
    teacher's `<tool_call>` (behind Qwen3's empty `<think></think>`), and the longest example is
    7,073 tokens.
  - The final code's build has the same `data_digest` as the gated one.
  - Needs Python ≤ 3.12: `datasets` 4.0.0 (pinned by LLaMA-Factory) cannot pickle under Python 3.14.
    The gate used a uv-installed Python 3.12 venv (`venv312`) with CPU torch; torchvision and
    torchaudio had to come from the same CPU index.
- 5.5 Open: no API key yet (`.env` absent).
- 5.6 Done:
  - `test_todo.py` deleted in Phase 4;
  - README updated (record, building, router, manifest);
  - this file kept (see top);
  - root suite: 178 passed, 2 skipped (the `ADP_UPSTREAM` check, and the `datasets` loader test that
    passes in the spike venv).
- Other review fixes: `--force` help text; manifest written as UTF-8; git info is None for an installed
  (site-packages) copy.

**Done means:**
- `python -m avians_adp.build` turns episodes into train/val/test JSONL plus a manifest: **yes**.
- LLaMA-Factory loads them without edits: **yes** (gate 5.4).
- The two skipped tests are real and passing: **yes** (`test_sft.py::test_pilot_examples_one_per_decision`,
  `test_routing.py::test_router_examples`).
- `python -m pytest` passes at the root: **yes**.
- Open: gate 2.6c / 5.5 (real Haiku episodes); real router labels (need the student model).

**Final review (task 3 of the Phase 5 request).** A Fable reviewer checked the whole package against
the notebook, and edited REPORT.md only. Verdict: the goal is accomplished as the notebook defines it.
Findings and what was done:
- D1: the TRL `tools` list is null-padded by `datasets` 4.0.0 (667 nulls/row) but not by 5.2.0 (0;
  measured both). The earlier gate ran with 5.x before LLaMA-Factory downgraded that venv. Documented in
  the README and REPORT; the layout choice is left to Austin.
- D2: `ruff --fix` had rewritten `timezone.utc` as 3.11-only `datetime.UTC`. Fixed; the full root suite
  now passes on Python 3.10.22 (178 passed, 2 skipped; `venv310`).
- D3: "2048 trains on nothing" was wrong (LLaMA-Factory cuts the prompt, keeps the target). README
  corrected.
- D4 (a missing student row becomes escalate, uncounted) and D5 (the loader test is skipped in `.venv`)
  are open; see REPORT.md section 7.

## Setup state

- Branch `mcglashan/adp-datasets` exists; nothing is committed.
- `.venv` (Python 3.14.4) has every package installed.
- Seeds 0–5 of the scripted pilot are in `data/episodes/sar_supply/scripted/`, in the new format.
