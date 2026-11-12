# avians_adp: status report (McGlashan)

**As of 2026-11-12. Branch `mcglashan/adp-datasets`, 6 commits on `main` (`1633daa`..the report
commit), not yet pushed or opened as a pull request.**

## Independent review (final)

*2026-11-12, by a reviewer given the notebook task, the code and the data, not the author's conclusions.
Only this file was edited; corrections to the author's text below are marked "(corrected in review)".*

**Verdict: the package goal is accomplished as the notebook defines it.** `python -m avians_adp.build`
turns a folder of episodes into train/val/test JSONL plus a manifest; LLaMA-Factory 0.9.5 loads the
sharegpt files without edits and supervises exactly the teacher's call; the two placeholder tests are
real and pass; the root suite passes. All five tasks are met on the code I read and the builds I made.
**One claim in this report was wrong and matters for training:** the TRL layout (`pilot/*.jsonl`) is
*not* read back "exactly as written" by `datasets`. Arrow pads every tool's parameter schema with the
other tools' property names as nulls (665 nulls a row with the real 23 tools; the rendered prompt
grows from 6.1k to 9.7k Qwen3 tokens). The LLaMA-Factory files are unaffected. Details, with the
other findings, are in section 7; the fixes are small (section 5, steps 0a–0b).

What I ran (everything under `~/.cache/avians_adp_final/`, nothing written into `data/`):
- `python -m pytest` at the root: **178 passed, 2 skipped**; `packages/avians_adp` alone: 143 passed,
  2 skipped (both skips are opt-in: `ADP_UPSTREAM`, `datasets`).
- `avians rollouts missions/sar_supply.yaml --seeds 0-7 --pilot scripted --out ...` (8 free episodes),
  `python -m avians_adp validate` (8/8 ok), `python -m avians_adp.student_calls ... --pilot scripted`
  (8 files, 83 calls), then three builds: defaults; `--satisfied --keep-every 5 --battery-band 0
  --split 60/20/20 --text --thinking`; and `--require-clauses nosuchclause` (a clean error, nothing
  written). `--force` rebuild gave the same `data_digest`; a non-empty folder without `--force` refused.
- A spot check of every row of both builds against its trajectory (296 pilot rows, 354 router rows:
  system prompt, prompt, tools, call name and arguments incl. `reason`/`notes`, sharegpt = TRL, meta
  fields, seed-split membership, no episode in two splits, only keeps thinned and only in train, router
  labels recomputed, router `text` = `avians_agent.router.features(state, recorded events)`): 0 problems.
- LLaMA-Factory gate (`venv312`, `gate_5_4.py`): train 110/110, val 21/21 rows supervise exactly the
  teacher's `<tool_call>`, longest 7,073 tokens at `cutoff_len` 8192.
- `datasets` + Qwen3 gate (`gate_2_6b.py`): crashes in the Python 3.14 venv (dill vs 3.14, an
  environment fault, not the data); in `venv312` (same `datasets` 4.0.0, pyarrow 26.0.1) it reports
  0/110 rows loaded equal to written, for the reason above.
- Upstream check: cloned `neulab/agent-data-protocol` at `274f935a`; `schema.py` matches its field
  names, types, discriminator values and validators; the opt-in test passes against it (7 passed).

### Author's follow-up to the final review (2026-11-12)

- **D1 depends on the `datasets` version.** Both the review and the earlier gate measured correctly.
  Measured on the same TRL file (`pilot/train.jsonl`, 91 rows, 23 tools):

  | `datasets` version | Null paths per row after `load_dataset("json")` | Rows loaded == written |
  |---|---|---|
  | 4.0.0 (LLaMA-Factory 0.9.5 pins this) | 667 | 0/91 |
  | 5.2.0 (latest) | 0 | 91/91 |

  Gate 2.6b first ran in the spike venv with the latest `datasets`. Installing LLaMA-Factory there later
  downgraded it to 4.0.0, which then crashed on Python 3.14. The review only had 4.0.0 environments.
  - **Consequence:** the TRL layout is correct, and loads unchanged, with `datasets` ≥ 5. With
    `datasets` 4.x it is silently padded.
  - **The sharegpt / LLaMA-Factory path is unaffected either way.**
  - **The fix is a decision for Austin** (section 5, step 0a):
    - keep `tools` as a list, which TRL takes natively, and require `datasets` ≥ 5; or
    - write a JSON string, which works on any version but needs a parse step in the TRL script.
- **D2 is fixed.**
  - The cause: `ruff check --fix`, run with a ruff config from outside the repo that targets Python ≥
    3.11, had rewritten `timezone.utc` as `datetime.UTC` in the Phase 5 commit. `build.py` and
    `manifest.py` use `timezone.utc` again.
  - **Verified on Python 3.10.22**, a uv-installed interpreter with the whole avians stack plus
    `anthropic` in `~/.cache/avians_adp_spike/venv310`: the full root suite gives **178 passed,
    2 skipped**, the same as on 3.14. So the package now runs on 3.10, as its `pyproject.toml` says.
- **D3 is accepted.** The README's "2048 trains on nothing" sentence is corrected to the measured
  behaviour.
- **D4 and D5 are open**, as described in section 7.2.

All five notebook tasks for `packages/avians_adp` are implemented and tested, and the notebook's
definition of done holds:
- `python -m avians_adp.build <episodes>` turns a folder of episodes into train/val/test JSONL plus a
  manifest.
- LLaMA-Factory loads the files without edits.
- The two placeholder tests are now real, passing tests.
- The full suite passes: 178 passed, 2 opt-in skips.

So far everything has been verified on scripted-pilot episodes only. Two things wait on other people:
- **Real teacher data:** an Anthropic API key from Andema.
- **Real router labels:** a choice of student model (open in the notebook log).

Companion documents:
- `PLAN.md`: the plan as written, and a phase-by-phase record of what was executed and why it changed.
- `README.md`: how to use the package.

---

## 1. The original tasks and how the code meets them

| # | Notebook task | Status | Where | Verified by |
|---|---|---|---|---|
| 1 | **Schema.** Check against upstream ADP, import if pip-installable, add a validator every saved trajectory passes | Done | `schema.py`, `validation.py`, `decisions.py`, `recorder.py` | `test_schema.py`, `test_recorder.py`; an opt-in test against upstream's own models; all real episodes validate |
| 2 | **Pilot SFT set.** System prompt, tools and decision prompt in; the teacher's tool call with its reason out; chat JSONL for LLaMA-Factory and TRL in the student's tool-calling layout | Done | `sft.py`, `formats.py` | `test_sft.py`; `datasets` load check; Qwen3 render check; LLaMA-Factory load (gate 5.4) |
| 3 | **Filters and splits.** By episode or clause robustness; thin near-identical keeps; split by seed, never one episode across splits | Done | `filters.py`, `splits.py`, `build.py` | `test_filters.py`; gate on seeds 0–5 |
| 4 | **Router set.** Label each decision commit / local / escalate | Done, but real labels need the student model | `routing.py`, `student_calls.py`, `build.py` | `test_routing.py`, including against the live router |
| 5 | **Manifests.** Every build writes counts, episode ids, the git commit and the filter settings | Done | `manifest.py`, `build.py` | `test_manifest.py` |

### Task 1: schema
- **Upstream has moved.** ADP's repository ([neulab/agent-data-protocol](https://github.com/neulab/agent-data-protocol))
  switched its canonical schema to Harbor's **ATIF-v1.7** on 2026-06-13 (commit `3824726`, "Add ATIF
  conversion pipeline"; date corrected in review). The paper-style classes our
  starter mirrored were deleted. Their last version is **ADP 1.3.1** (commit `274f935a`).
- **It cannot be imported.** It is not on PyPI, needs Python ≥ 3.12 (we support 3.10), and installs
  top-level packages called `schema`, `agents` and `scripts`. So `schema.py` mirrors 1.3.1 and pins
  that commit. This follows Austin's decision: ADP 1.3.1 now, an ATIF export later.
- **What changed in our record to match upstream:**
  - Each tool call and its result share a `tool_call_id`.
  - A skill's reply is `source="environment"` (`"tool"` is not an ADP value).
  - The thinking summary lives in `reasoning_content`.
  - The record has `schema_version` and `available_apis`.
  - `ApiAction` replaces `APIAction`, with the old name kept as an alias.
  - Per-step `details` stays as our extension; `to_upstream()` strips it.
- **Old records still load.** Records written before this change are upgraded on load.
- **Validation:**
  - `validate()` checks upstream's rules plus the avians decision pattern.
  - Every `finish()` writes the record first, then validates and warns, so a paid episode is never
    lost to a validation bug.
  - `python -m avians_adp validate <dir>` checks a folder.
- **Contract:** `EpisodeRecorder`'s method signatures are unchanged, so `avians_agent` needed no edit.
  The file format did change (see section 3).

### Task 2: pilot SFT set
- **One example per decision.**
  - The input is the system prompt, the tools (converted to the OpenAI function layout) and the decision
    prompt.
  - The target is the teacher's tool call, with `reason` and `notes` **put back into the arguments**.
    The loop strips them before recording, but the student's tool schema has them as arguments.
  - The loop's `" (ignored N more call(s))"` suffix is removed from the target.
- **Two layouts, chosen for a Qwen3 student:**
  - TRL: `messages` plus `tools`, with `arguments` as a JSON string.
  - LLaMA-Factory sharegpt: `conversations`, `system` and `tools`, plus `dataset_info.json`.
  - A meta side file carries the episode, seed, model, decision kind and so on.
- **Verified:**
  - The sharegpt layout loads with `datasets` exactly as written. **The TRL layout does not**
    (corrected in review): `datasets` 4.0.0 / pyarrow 26.0.1 unify the `tools[*].function.parameters.
    properties` structs across the 23 tools and pad each tool with the others' property names as
    nulls (665 null paths a row), and add `tool_calls: null` to the system and user messages. The
    earlier text said "no null padding with the real 23 tools"; that could not be reproduced in any
    environment on this machine (section 7, D1).
  - The Qwen3-4B chat template renders each *written* row with exactly one correct `<tool_call>`;
    a row *loaded through `datasets`* renders the same call but a tools block with the nulls in it.
  - LLaMA-Factory 0.9.5's own pipeline loads the sharegpt files unmodified, and the supervised tokens
    are exactly the teacher's call (re-verified in review on a fresh build: 110/110 train, 21/21 val).

### Task 3: filters and splits
- **Episode filters:** `--satisfied`, `--min-robustness`, `--require-clauses` (with `--clause-margin`),
  `--pilot`, `--tier`. A misspelled clause name is an error. Infinite robustness, which the scorer
  writes as the string `"inf"`, is handled.
- **Decision filters,** on by default and each counted:
  - keeps the teacher never chose (API error, no tool call, router commit);
  - the student's own decisions in an episode flown with a router;
  - rejected calls and unanswered calls.

  `--allow` keeps any of them, and `--model` restricts to one model.
- **Splits:** a sha256 of the seed against the ratios (default 80/10/10).
  - Every episode of a seed, from any pilot, is in one split.
  - Adding episodes never moves a seed.
  - Pinned values in a test guard against an accidental reshuffle.
- **Keep-thinning (train only):** a run of consecutive keeps with the same skill, status, mission facts,
  closed regions and battery band keeps 1 in `--keep-every` (default 3). Val and test are never thinned.

### Task 4: router set
- **Labels:**
  - `commit`: the teacher kept going.
  - `local`: a student run offline on the same prompt makes the same call (same tool; same arguments
    ignoring reason/notes; numbers within `routing.TOLERANCES`, e.g. 10 m for positions).
  - `escalate`: anything else.
- **Only the teacher's decisions are labelled.** An API failure by the student is skipped, not
  labelled.
- **The router's input is recorded, not reconstructed.** The recorder now stores, on each decision
  prompt, the event kinds since the previous one. Rebuilt router text equals what the live router read
  in flight on 81 of 81 decisions; a test checks this. Reconstructing from `events.jsonl` timestamps is
  wrong about 10 % of the time.
- `python -m avians_adp.student_calls <episodes> --pilot <name>` runs a student offline:
  - a fresh pilot per episode;
  - only where the label depends on the student (about 40 % of decisions);
  - with the `escalate` tool it has in flight.

  The build then writes `router/<split>.jsonl` (`id`, `text`, `label`, `options`) with the same splits
  as the pilot set.

### Task 5: manifests
- **`manifest.json` records:**
  - the source folder, and the skipped episodes with reasons;
  - the episode ids and counts per split, by tool, pilot and model;
  - what each filter removed, and the router labels;
  - the full settings, including tolerances;
  - the ADP version and the file encoding;
  - the git commit, branch and uncommitted changes, scoped to `packages/` (untracked files included);
  - the command line;
  - a sha256 and row count per file, and a `data_digest` over all of them.
- **Reproducible:** identical inputs and settings give an identical `data_digest`.
- **Builds are transactional.** Files are staged inside the output folder, the previous build is renamed
  aside, and any failure restores it byte for byte. A failed build never leaves half a dataset.

---

## 2. What works today

```bash
source .venv/bin/activate
avians rollouts missions/sar_supply.yaml --seeds 0-99 --workers 8        # free scripted episodes
python -m avians_adp validate data/episodes/sar_supply/scripted          # check the records
python -m avians_adp.student_calls data/episodes/sar_supply/scripted --pilot scripted   # optional: router set
python -m avians_adp.build data/episodes/sar_supply/scripted --satisfied # -> data/datasets/build-<UTC>/
```

| Check | Result |
|---|---|
| Root test suite (`python -m pytest`) | 178 passed, 2 skipped (145 tests in `avians_adp`) |
| Real episodes validate (scripted seeds 0–5) | 6/6 |
| Every SFT target round-trips to the recorded call (seeds 0–5) | 156/156 decisions |
| `datasets` loads the files unchanged; Qwen3 template renders the right call | sharegpt: yes. TRL: **no**, the tools column is null-padded on load (corrected in review; section 7, D1). The written TRL rows render the right call |
| LLaMA-Factory 0.9.5 `get_dataset` (template `qwen3`, `cutoff_len` 8192) | 91/91 train, 35/35 test; supervised tokens = the teacher's call |
| Rebuilt router text = the live router's input | 81/81 decisions (3 seeds) |
| Upstream ADP 1.3.1's own models accept our records (`ADP_UPSTREAM=...`) | yes (fixture, hand-built and all real episodes) |
| Build on seeds 0–5 with `--satisfied` | the 4 satisfied episodes; train 73 → 55 after thinning; test 35 |

**Independent reviews.** A separate reviewer (a Fable subagent, given the requirements and the code but
not my conclusions) checked Phases 2–5. Every confirmed defect was fixed and covered by a test. The
serious ones:
- `--force` could delete episodes stored under the output folder.
- Stored infinite robustness crashed the episode filters.
- Episodes flown with a router fed the student's own decisions into both training sets as the
  teacher's.
- A student's API failures were recorded as normal `keep` calls.
- A failure mid-swap could lose both the old and the new build.

The details are in `PLAN.md`, Status. One finding of that review, "`datasets` pads `tools` with
nulls", was rejected on the strength of gate 2.6b; the final review reproduced the padding (section 7,
D1), so that rejection was wrong (noted in review).

---

## 3. Changes that affect other packages (read this, Andema and Taylor)

- **The `trajectory.json` format changed; the recorder's API did not.** New fields:
  `schema_version`, `available_apis`, `tool_call_id`, `reasoning_content`, `decision` on each action,
  and `events` on each decision prompt. Skill replies are now `source="environment"`.
  - `avians_viz` and `avians latency-fit` read the record directly; both were re-checked and still work.
  - Old episodes load and upgrade automatically.
- **No file outside `packages/avians_adp` was edited.**
- **Training sequence length: set `cutoff_len` / `max_length` to 8192.**
  - An example is 6.1k–7.1k Qwen3 tokens: system prompt about 1.8k, the 23 tools about 3.9k, the
    decision prompt about 0.9k.
  - The target is about 25 tokens at the very end. With a 2048 cutoff LLaMA-Factory 0.9.5 keeps the
    target and cuts the *prompt* to fit (`infer_seqlen` truncates the source first), so the student
    would train on the right call after a prompt missing its last two thirds, including the state,
    memory and "Decide" lines, with no warning (corrected in review: the earlier text said "trains on
    nothing"; measured on the test split, 35/35 rows hit the cutoff with the call intact). Either way
    a 2048 cutoff is wrong.
  - This is for the sharegpt files. A TRL run that loads `pilot/*.jsonl` with `datasets` sees ~9.7k
    tokens an example because of the null padding (section 7, D1), so it needs the fix there first.
  - The student needs an ~8k context window. Shrinking the tool descriptions or the system prompt would
    cut cost a lot.
- **LLaMA-Factory needs Python ≤ 3.12.** The `datasets` 4.0.0 it pins cannot run on Python 3.14.
  Install torch, torchvision and torchaudio from the same index.
- **Qwen3 target format.** The supervised text is
  `<think>\n\n</think>\n\n<tool_call>{...}</tool_call>`: Qwen3's non-thinking form. With `--thinking`,
  the TRL layout fills the think block with the teacher's summary.

---

## 4. What still needs to be done

| Item | Blocked on | Notes |
|---|---|---|
| **Real teacher data** (gates 2.6c / 5.5): about 10 Haiku episodes, a build, and 5 rendered examples read by eye | API key from Andema (`.env` is absent) | Exercises paths scripted episodes never show: visible text, thinking summaries, rejected calls, `stop` reasons, `no_tool_call` |
| **Real router labels** | Student model choice (notebook log: open; working assumption Qwen ~1–4B) | A scripted student is a copy of the teacher: 91 commit, 65 local, 0 escalate. Correct, but not informative |
| **Router training format** for Kev / Laya | System One API reference (notebook log: open) | Rows are plain classification (`text`, `label`, `options`), easy to reshape |
| ~~**Python 3.10**~~ | Done | The review found `from datetime import UTC` (3.11+) in `build.py` and `manifest.py` (section 7, D2). Fixed after the review; the full root suite passes on Python 3.10.22 (178 passed, 2 skipped) |
| **Gazebo-tier records** | The Linux machine | The recorder and decision grouping are tier-independent; the reviewer simulated the realtime path (37/37 router features equal) but no real Gazebo record exists |
| **ATIF export** | Nothing | Decided for later; upstream's converters now consume ATIF |
| **Router labels from outcomes** | More data | README task 4: "later, label from outcomes too (which choice led to higher robustness)" |
| **Scale** (README task 6) | Andema's approval | `avians rollouts ... --pilot haiku --seeds 0-999` is about $20; ask before more than 100 episodes |
| **Pull request** | Austin | Push the branch and open the PR into `main`; the description should carry section 3 |

---

## 5. Suggested next steps (in order)

0. **Two small fixes before the pull request** (added in review; section 7):
   - **0a.** In `formats.py` `to_trl`, write `tools` as a JSON string (as `to_sharegpt` already does),
     set `ENCODING["tools_trl"] = "json_string"`, and say in the README that a TRL training script
     must `json.loads(row["tools"])` before `apply_chat_template`. Then make
     `test_files_load_unchanged_with_datasets` run somewhere it is not skipped (it is skipped in
     `.venv`, and the fixture it uses already pads on load). Alternative: keep the list and document
     the padding with a loader that strips nulls, but the string is simpler and what LLaMA-Factory does.
     *Author's note:* this depends on the `datasets` version (see "Author's follow-up" at the top):
     5.x loads the list unchanged, and TRL takes a list natively. So the choice is between "list +
     require `datasets` ≥ 5" (no TRL script change) and "JSON string" (any version, one parse step).
     Austin to decide. Either way, make the loader test run in CI with a pinned `datasets`.
   - ~~**0b.**~~ Done after the review: `timezone.utc` restored and the suite run on Python 3.10.
1. **Open the pull request now.** The work is complete against the notebook, and the record-format
   change (section 3) should land before others write many episodes in the old format.
2. **Get the API key; fly ~10 Haiku episodes (about $0.20).** Run `python -m avians_adp validate`,
   build, and read five rendered examples. This closes gates 2.6c and 5.5.
3. **Decide the student model,** then run it with `student_calls` over the Haiku episodes for real
   router labels. A local model needs a `decide()` adapter, a small function. Check the label balance
   in the manifest before training.
4. **Shrink the prompt** (team decision). About 85 % of every example is the same system prompt and tool
   list. Options:
   - shorter tool descriptions;
   - training without the tool list (the student learns the tools);
   - prefix caching at inference.
5. **Agree the router format** with whoever owns System One, and add a writer.
6. **Scale up** with Andema's approval: 1000 Haiku seeds is about $20 and gives a real val split (seeds
   0–5 put nothing in val at 80/10/10).
7. **Housekeeping:**
   - ~~test on Python 3.10~~ (done, see the follow-up at the top);
   - pin `target-version = "py310"` for ruff in the repo, so `ruff --fix` cannot reintroduce 3.11-only
     code (D2's cause);
   - move the two gate scripts (below) into the repo, e.g. `packages/avians_adp/scripts/`, with an
     optional extra for their dependencies;
   - consider an ATIF exporter.

---

## 6. Handoff for the next coding session

**State:**
- Branch `mcglashan/adp-datasets`; commits `1633daa` (Phase 1) … `200684c` (Phase 5). Clean apart from
  an untracked `.vscode/` that is not ours.
- `PLAN.md` has the plan and the per-phase status.
- The facts F1–F9 in `PLAN.md` are the checked assumptions behind the design.

**Environments:**
- `.venv`: the repo, Python 3.14.4, installed by `./scripts/install.sh`.
- `~/.cache/avians_adp_spike/venv`: Python 3.14 with `datasets`, `transformers`, `jinja2`. Was meant to
  run `gate_2_6b.py <build> <split>` (load and Qwen3 render), but as of the final review
  `datasets.load_dataset` crashes there (`dill` 0.4.1 vs Python 3.14 pickling). Run that gate with
  `venv312` instead (it has `datasets`, `transformers` 5.6.0 and the Qwen3-4B tokenizer); expect it to
  fail on the TRL split until step 0a in section 5 is done.
- `~/.cache/avians_adp_spike/venv312`: Python 3.12 (uv-installed) with LLaMA-Factory 0.9.5 and CPU
  torch. Runs `gate_5_4.py <build> avians_pilot_train` (LLaMA-Factory pipeline). It has no `pip` or
  `pytest`; `uv` is not on PATH on this machine (the interpreter is under `~/.local/share/uv/python/`).
- `~/.cache/avians_adp_spike/venv312new`: Python 3.12 with `datasets` 5.2.0 only (the D1 comparison;
  `padcheck.py <jsonl>` in the same folder).
- `~/.cache/avians_adp_spike/venv310`: Python 3.10.22 with the whole avians stack (editable) plus
  `pytest` and `anthropic`. Run the root suite with `venv310/bin/python -m pytest -p no:cacheprovider`.
- `~/.cache/avians_adp_final/`: the final review's scratch: 8 scripted episodes (`episodes/`), two
  builds (`build_default`, `build_filtered`), the upstream checkout (`upstream/`), the check scripts
  (`spotcheck.py <build> <episodes>`, `diffload.py <jsonl>`, `render_loaded.py <trl jsonl>`,
  `three_tools.py`) and the gate logs (`gates.txt`, `gates2.txt`).
- Neither gate script is in the repo yet (next step 7).
- `ADP_UPSTREAM=<checkout of neulab/agent-data-protocol@274f935a>` enables the upstream test. A copy of
  the pre-ATIF schema was in the session scratchpad, which is gone. Re-clone, then `git checkout 274f935a`.

**Code map** (`src/avians_adp/`, about 1,860 lines):

| Module | Role |
|---|---|
| `schema.py` | ADP 1.3.1 models + legacy upgrade + tool-call link rules |
| `recorder.py` | `EpisodeRecorder` (the contract with `avians_agent.loop`) and `load()` |
| `decisions.py` | groups steps into decisions (prompt, message, action, result) |
| `validation.py`, `__main__.py` | `validate`, `validate_dir`, `to_upstream`; `python -m avians_adp validate` |
| `sft.py` | `pilot_examples`, `decision_kind`, `router_examples` (delegates) |
| `formats.py` | TRL and sharegpt writers, `dataset_info.json` |
| `filters.py`, `splits.py` | episode/decision filters, keep-thinning, seed splits |
| `routing.py`, `student_calls.py` | router labels and matching; offline student runs |
| `manifest.py`, `build.py` | the manifest; the build pipeline, CLI and transactional install |

**Gotchas learned the hard way:**
- **`by` does not identify the teacher.** Without a router it is always `"local"`. Use
  `decision_kind()`, which reads the episode's `meta.router`.
- **`ScriptedPilot` is stateful** (it remembers regions searched). Use one per episode.
- **The repo's `addopts` already has `-q`.** An extra `-q` hides the summary line.
- **Ruff picks up a stricter config from outside the repo.** The four remaining findings
  (`SIM115` ×2, `TRY004` ×2) are deliberate.
- **Never spend API money without asking.** Scripted rollouts are free; Haiku is about $0.02 an episode.
- **`datasets` 4.x is not a byte-for-byte loader.** Any column of dicts whose keys differ across rows
  or list items (tool schemas, tool arguments) comes back with the union of keys, missing ones as null.
  `datasets` 5.x keeps them as written. Record the `datasets` version with any training run.
- **Do not run `ruff check --fix` without checking the diff.** It follows a config outside the repo that
  targets newer Python, and it rewrote `timezone.utc` into the 3.11-only `datetime.UTC` once (D2).

---

## 7. Final review: results and defects (added in review, 2026-11-12)

### 7.1 The notebook's tasks, checked against the code and the builds

| # | Requirement | Finding |
|---|---|---|
| 1 | Schema matches upstream; import if pip-installable; validator every trajectory passes | **Met.** `schema.py` matches `neulab/agent-data-protocol@274f935a` field for field (`Action`/`Observation` optionals, `class_` values, `source` literal, `validate_tool_result_source`, `validate_tool_call_links`, supported versions). Not importable for the reasons given in section 1; mirroring is the right call and was Austin's decision. `validate()` passes on all 8 fresh episodes and the 6 in `data/`; `finish()` writes before validating. The per-step `details` extension is additive and `to_upstream()` strips it (upstream models accept the result: 7/7 opt-in tests) |
| 2 | One example per decision; system prompt, tools, decision prompt in; teacher's call with reason out; chat JSONL that LLaMA-Factory and TRL load, in the student's tool-calling layout | **Met for LLaMA-Factory; partly met for TRL.** 296 rows spot-checked against their trajectories: 0 differences. `reason` and `notes` go back into the arguments and match the student's tool schema. Qwen3 renders a written row as `<think>\n\n</think>\n\n<tool_call>{...}</tool_call>`, and with `--thinking` as `<think>\n<summary>\n</think>` (rendered in review). The TRL file is well-formed but see D1 |
| 3 | Filter by episode or clause robustness; thin near-identical keeps; split by seed, never one episode across splits | **Met.** `--satisfied` removed exactly the 2 unsatisfied episodes of 8 (seeds 1, 5); a bad clause name is an error before anything is written. Thinning removed only `keep` rows, only in train, in decision order (35 of 145 train decisions at `--keep-every 3`; 23 at 5 with `--battery-band 0`). Every row's split equals `split_of(seed, ratios)`; no episode in two splits in either build |
| 4 | Router set: commit / local / escalate as defined | **Met, with the stated caveat.** Labels recomputed independently for all 354 router rows: 0 differences; `text` equals the live router's `features()` over the recorded event kinds for all of them. With the scripted stand-in student the labels are 207 commit / 147 local / 0 escalate across the two builds, which says nothing about a real student (the scripted policy reads the state, not the prompt). Real labels wait on the student model, as section 4 says |
| 5 | Every build writes counts, episode ids, the git commit and the filter settings | **Met.** `manifest.json` has all four plus file hashes, row counts, the command line and the skipped episodes; identical inputs gave identical `data_digest` across a `--force` rebuild |

"Done means": build entry point, yes; LLaMA-Factory loads without edits, yes (110/110, 21/21,
supervised tokens = the call); the two placeholder tests are real and pass, yes
(`test_sft.py::test_pilot_examples_one_per_decision`, `test_routing.py::test_router_examples`,
`test_todo.py` deleted); `python -m pytest` at the root, 178 passed, 2 skipped.

### 7.2 Defects found (most severe first)

**D1. Confirmed with `datasets` 4.0.0; not reproduced with 5.2.0 (author's follow-up at the top). The TRL layout is null-padded when loaded with `datasets` 4.x; the report said it was not.**
`src/avians_adp/formats.py:45` (`"tools": ex["tools"]`, a list of dicts). Arrow unifies the
`parameters.properties` struct across the 23 tools, so after `load_dataset("json", ...)` every tool
carries every other tool's property names with `None` values (665 null paths a row, measured on
`build_default/pilot/test.jsonl`; the author's own earlier build `~/.cache/avians_adp_spike/build_s0-5`
pads the same way; even the 3-tool fixture of `tests/test_sft.py:260` pads, so that test would fail
anywhere `datasets` works, and it is skipped in `.venv`). `messages[0..1].tool_calls` also become
`null` (harmless: Qwen3's template treats it as absent).
*Failure scenario:* a TRL `SFTTrainer` run that loads `pilot/train.jsonl` with `datasets` and passes
`row["tools"]` to the chat template trains the student on tool schemas with ~30 spurious `"x": null`
properties per tool that it will never see at inference, and the example grows from 6,134 to 9,657
Qwen3 tokens, so at `max_length` 8192 the target at the end is at risk of being cut. Nothing warns.
*Why the author's gate said otherwise:* the gate script now crashes in the Python 3.14 venv before
loading anything (dill vs 3.14); run in `venv312` with the same `datasets` 4.0.0 and pyarrow 26.0.1
it reports 0/110 rows equal. I could not find an environment where the "0 null paths" result occurs.
*Fix:* section 5, step 0a. The LLaMA-Factory path (`pilot_lf/`, tools as a JSON string) is correct
and verified end to end; training with LLaMA-Factory is unaffected.

**D2. (Fixed after the review; see the follow-up at the top.) Confirmed by the language reference (no 3.10 interpreter on this machine). The package does not run on
Python 3.10.** `src/avians_adp/build.py:45` and `src/avians_adp/manifest.py:19`: `from datetime import UTC`
(added in Python 3.11). `pyproject.toml` declares `requires-python = ">=3.10"`.
*Failure scenario:* on 3.10, `python -m avians_adp.build` and `import avians_adp.manifest` fail with
`ImportError` at import time. `import avians_adp` (the recorder, which `avians_agent.loop` uses) still
works because `__init__.py` does not import those modules. *Fix:* section 5, step 0b.

**D3. Confirmed. A wrong statement in this report about `cutoff_len` 2048.** LLaMA-Factory 0.9.5
truncates the prompt and keeps the target (`llamafactory/data/processor/processor_utils.py`,
`infer_seqlen`): at 2048 all 35 test rows train on the full call after a prompt cut to ~2,000 tokens.
That is silently wrong training data, not "nothing". Corrected in section 3. No code change needed;
keep 8192.

**D4. Suspected, low. A missing student row is labelled `escalate`, not skipped.**
`src/avians_adp/routing.py:93` (`label()`, "no student call"). `student_calls.py` always writes a row
for every decision it is asked about, so a complete `student_calls.jsonl` never triggers this. A
partial file (an older `needs_student` rule, a hand-edited file, a file written by a future student
runner that stops early) would quietly turn every uncovered non-keep decision into `escalate`, with
only `meta.why` saying so and no count in the manifest. *Suggested:* count "no student call" under
`router.skipped` in the manifest, or skip those decisions like student errors.

**D5. Low. The `datasets` loader test cannot catch D1.** `tests/test_sft.py:260` is skipped in `.venv`
(no `datasets`) and would fail where `datasets` is installed. Either install `datasets` in `.venv` as
a test extra, or drop the test and keep the gate script in the repo (section 5, step 7).

Not defects, but worth knowing: scripted episodes never produce rejected calls, API errors, text or
thinking, so the `rejected`/`unanswered`/`api_error`/`no_tool_call` paths and the `--text`/`--thinking`
flags are covered by unit fixtures only, not by real data (the author says this in section 4; it holds).
With seeds 0–7 at 80/10/10 the val split holds one episode (seed 7); at 0–5 it is empty.

### 7.3 Risks to carry into the pull request

- Anyone training with TRL from `pilot/*.jsonl` before step 0a gets D1 without a warning. If the PR
  opens before the fix, say so in the description.
- The manifest records `tools_trl: "list"`; builds made before step 0a can be told apart by that field.
- Ask anyone training with TRL to record their `datasets` version: 4.x pads (D1).
- `git.dirty` in the manifest counts untracked files under `packages/`, so this `REPORT.md` made the
  review builds "dirty" until it is committed. Harmless, but do not read `dirty: true` as a code change.
