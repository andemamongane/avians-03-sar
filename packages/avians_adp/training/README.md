# Training the student pilot (LLaMA-Factory)

`pilot_sft_qwen3_4b_lora.yaml` fine-tunes a Qwen3 student on a build's `pilot_lf/` files with
LLaMA-Factory 0.9.5 (LoRA, supervised). `check_lf.py` checks a build against that config without a
GPU. Nothing here is imported by `avians_adp`; these files are for the training machine.

## 1. Build the dataset (in the avians `.venv`)

```bash
python -m avians_adp.build data/episodes/sar_supply/haiku data/datasets/pilot_v1 --satisfied
```

Read the counts it prints, and `manifest.json`. With few seeds the val split is empty (the build warns);
see "No val split" below.

## 2. Set up LLaMA-Factory (separate environment, Python ≤ 3.12)

LLaMA-Factory 0.9.5 pins `datasets` 4.0.0, which cannot run on Python 3.14. Use a Python 3.12 venv:

```bash
python3.12 -m venv ~/venvs/lf && source ~/venvs/lf/bin/activate
pip install torch torchvision torchaudio          # from one index (the CUDA one on a GPU machine)
pip install "llamafactory==0.9.5"
```

## 3. Check the build against the config (no GPU needed)

```bash
python -I packages/avians_adp/training/check_lf.py \
    packages/avians_adp/training/pilot_sft_qwen3_4b_lora.yaml data/datasets/pilot_v1/pilot_lf
```

It runs LLaMA-Factory's own parser and data pipeline, downloads only the tokenizer, and fails unless
every train and val row loads, has its loss on exactly one `<tool_call>` equal to the teacher's call,
and fits in `cutoff_len`. Expect `N/N rows supervise exactly the teacher's call; longest ~7,100 tokens`.

## 4. Train (GPU machine)

```bash
llamafactory-cli train packages/avians_adp/training/pilot_sft_qwen3_4b_lora.yaml \
    dataset_dir=data/datasets/pilot_v1/pilot_lf output_dir=data/models/pilot_v1_qwen3-4b_lora
```

`key=value` after the YAML overrides it, for both `llamafactory-cli` and `check_lf.py`.

## What the config fixes, and why

| Key | Value | Why |
|---|---|---|
| `template` | `qwen3` | The build writes Qwen3's tool-calling layout. Another student family needs another writer in `formats.py`. |
| `enable_thinking` | `true` | The supervised text is `<think>\n\n</think>\n\n<tool_call>{...}</tool_call>`: the student learns to write an empty think block, then the call. Serve it with Qwen3's default (thinking on); with `enable_thinking=False` at inference the empty block is in the prompt instead, which also works. |
| `cutoff_len` | `8192` | An example is 6.1k–7.1k tokens (system prompt ~1.8k, 23 tools ~3.9k, decision prompt ~0.9k). Below that, LLaMA-Factory silently cuts the *prompt* and keeps the target. `check_lf.py` fails if any row reaches the cutoff. |
| `train_on_prompt` | `false` | The loss is on the ~25-token call only. |
| `dataset`, `eval_dataset` | `avians_pilot_train`, `avians_pilot_val` | The names in the build's `pilot_lf/dataset_info.json`. |
| `warmup_ratio` | `0.05` | LLaMA-Factory 0.9.5 accepts transformers 4.55–5.6. `warmup_ratio` works on both (5.x warns it is deprecated); `warmup_steps: 0.05` would be a fraction only on 5.x and 0 warm-up steps on 4.x. |

Starting points, to tune on val: `Qwen/Qwen3-4B` (the student model is still open; any Qwen3 size uses
this file unchanged), LoRA rank 16 on all linear layers, learning rate 1e-4, 2 epochs, effective batch
16, bf16. 1,000 Haiku episodes give roughly 25k decisions before filtering; at ~6.5k tokens each, one
epoch is about 160M tokens, nearly all of them prompt.

## Small builds

`eval_steps` and `save_steps` are 200. A smoke run on a few seeds has only a handful of steps (91 rows
at effective batch 16 is 6 steps an epoch), so neither fires before the end. Add
`eval_strategy=epoch save_strategy=epoch` on the command line for such runs.

## No val split

Seeds are split by a hash of the seed (80/10/10 by default). With a handful of seeds val can be empty,
and `check_lf.py` then stops with a message. Either build with more seeds, build with another ratio
(`--split 70/15/15`), or for a smoke run drop `eval_dataset` and the `eval_*` keys
(`eval_dataset=null eval_strategy=no`).

## Not here yet

- **Evaluating the student.** Validation loss only. Tool-call accuracy on the test split and, above all,
  closed-loop flights scored by STL robustness need a student pilot in `avians_agent` (a `decide()`
  adapter that calls the served model and parses its `<tool_call>`), which does not exist yet.
- **The router (System One).** Its training layout waits on the System One API reference; the
  `router/` rows (`text`, `label`, `options`) are plain classification data.
- **TRL.** The `pilot/` files are the TRL layout. `tools` is a JSON string: call
  `json.loads(row["tools"])` before `apply_chat_template` (see the package README). Nothing enforces
  it: passed unparsed, Qwen3's template renders one "tool" per character of the string, without an
  error. A rendered prompt with thousands of one-character tools is that mistake.
