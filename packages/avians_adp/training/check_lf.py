"""Check a dataset build against a LLaMA-Factory training config, without a GPU or model weights.

usage (in the LLaMA-Factory environment, from the repository root):
    python -I packages/avians_adp/training/check_lf.py <config.yaml> <build>/pilot_lf [key=value ...]

Runs LLaMA-Factory's own argument parser and SFT data pipeline (get_dataset) with the config, its
dataset_dir replaced by the given folder, and checks every row of the train and eval datasets:
  - LLaMA-Factory loads every row the build wrote;
  - the supervised tokens (labels != -100) hold exactly one <tool_call>, equal to the teacher's call;
  - no row reaches cutoff_len (LLaMA-Factory would cut the prompt, silently).
Only the tokenizer is downloaded. Exit status 0 when every row passes.

Extra key=value arguments override the config, as with llamafactory-cli (e.g. cutoff_len=4096).
Formerly gate_5_4.py of the format spike (REPORT.md).
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


def _value(text: str):
    try:
        return json.loads(text)
    except ValueError:
        return text


def check(config: str, dataset_dir: str, overrides: dict | None = None) -> int:
    import yaml
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer

    lf_dir = Path(dataset_dir)
    info = json.loads((lf_dir / "dataset_info.json").read_text())
    args = yaml.safe_load(Path(config).read_text())
    args.update(overrides or {})
    args.update(dataset_dir=str(lf_dir), output_dir=tempfile.mkdtemp(), overwrite_cache=True,
                bf16=False, fp16=False, report_to="none")   # data pipeline only: no mixed precision on CPU
    names = {"train": args["dataset"], "eval": args.get("eval_dataset")}
    raws = {}
    for role, name in names.items():
        if not name:
            continue
        if "," in str(name):
            sys.exit(f"{role}: one dataset at a time, got {name!r}")
        if name not in info:
            sys.exit(f"{role}: {name!r} is not in {lf_dir / 'dataset_info.json'} ({sorted(info)})")
        raws[role] = [json.loads(x) for x in (lf_dir / info[name]["file_name"]).read_text().splitlines()]
        if not raws[role]:
            sys.exit(f"{role}: {name} is empty. Build with more seeds, or drop eval_dataset and the eval_* keys")

    model_args, data_args, training_args, _, _ = get_train_args(args)
    tok_mod = load_tokenizer(model_args)
    tokenizer = tok_mod["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tok_mod)
    cutoff = data_args.cutoff_len
    failed = 0
    for role, key in (("train", "train_dataset"), ("eval", "eval_dataset")):
        if role not in raws:
            continue
        ds, raw = module[key], raws[role]
        bad, longest = 0, 0
        if len(ds) != len(raw):
            print(f"{role}: written {len(raw)} rows, LLaMA-Factory loaded {len(ds)}")
            failed += 1
        for i in range(min(len(ds), len(raw))):
            ids, labels = ds[i]["input_ids"], ds[i]["labels"]
            longest = max(longest, len(ids))
            target = tokenizer.decode([t for t, lab in zip(ids, labels) if lab != -100], skip_special_tokens=False)
            want = json.loads(raw[i]["conversations"][1]["value"])
            try:
                ok = (target.count("<tool_call>") == 1 and len(ids) < cutoff
                      and json.loads(target[target.index("<tool_call>") + 11:target.index("</tool_call>")]) == want)
            except ValueError:
                ok = False
            if not ok:
                bad += 1
                if bad <= 3:
                    print(f"  {role} row {i} BAD: {len(ids)} tokens; supervised text: {target[:300]!r}")
            if i == 0 and role == "train":
                print(f"supervised tokens of train row 0: {target!r}")
        print(f"{role} ({names[role]}): {len(ds) - bad}/{len(raw)} rows supervise exactly the teacher's call; "
              f"longest {longest} tokens (cutoff_len {cutoff})")
        failed += bad
    return 0 if failed == 0 else 1


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 2
    overrides = {}
    for kv in argv[2:]:
        k, sep, v = kv.partition("=")
        if not sep:
            sys.exit(f"expected key=value, got {kv!r}")
        overrides[k] = _value(v)
    return check(argv[0], argv[1], overrides)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
