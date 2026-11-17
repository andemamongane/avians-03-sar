"""Training-file layouts for pilot examples (sft.pilot_examples).

Two layouts, both for a Qwen3 student (its chat template prints each tool as
JSON inside <tools>, and each call as <tool_call>{"name", "arguments"}):

  TRL         {"messages": [system, user, assistant + tool_calls], "tools": "<json>"}
              Loads with datasets.load_dataset("json"); parse `tools` with
              json.loads before apply_chat_template / trl.SFTTrainer.
  sharegpt    {"conversations": [human, function_call], "system": ..., "tools": "<json>"}
              LLaMA-Factory's sharegpt layout, described by dataset_info.json.

Encoding (format spike, PLAN.md 0.2):
  - tool-call `arguments` is a JSON string. Qwen3's template prints a string
    as is and a dict through tojson, so both render the same text, but a
    string cannot hit Arrow type conflicts across tools whose arguments
    differ. (A template that always applies tojson would double-encode it:
    another student family means another writer, not a rebuild.)
  - `tools` is a JSON string in both layouts. TRL: a list of OpenAI function
    dicts ({"type": "function", "function": {...}}); sharegpt: the bare
    function dicts, as LLaMA-Factory's examples have. A list of dicts is not
    safe: `datasets` 4.x (Arrow) merges the parameter names of all tools into
    one struct and pads every tool with the others' names as nulls (REPORT.md,
    D1). A string loads unchanged on any `datasets` version.

The target is the same in both: the teacher's tool call with its reason.
The pilot's visible text and thinking summary go into the TRL layout only,
and only when asked for: a sharegpt function_call turn has no place for them.

`meta` never enters a training file; writers put it in <name>.meta.jsonl, in
the same row order.
"""
from __future__ import annotations

import json
from pathlib import Path

ENCODING = {"arguments": "json_string", "tools_trl": "json_string", "tools_sharegpt": "json_string"}


def to_trl(ex: dict, *, include_text: bool = False, include_thinking: bool = False) -> dict:
    assistant: dict = {"role": "assistant", "content": ex["text"] if include_text else "",
                       "tool_calls": [{"type": "function", "function": {
                           "name": ex["call"]["name"],
                           "arguments": json.dumps(ex["call"]["arguments"], ensure_ascii=False)}}]}
    if include_thinking and ex.get("thinking"):
        assistant["reasoning_content"] = ex["thinking"]
    return {"messages": [{"role": "system", "content": ex["system"]},
                         {"role": "user", "content": ex["prompt"]},
                         assistant],
            "tools": json.dumps(ex["tools"], ensure_ascii=False)}


def to_sharegpt(ex: dict) -> dict:
    call = {"name": ex["call"]["name"], "arguments": ex["call"]["arguments"]}
    return {"conversations": [{"from": "human", "value": ex["prompt"]},
                              {"from": "function_call", "value": json.dumps(call, ensure_ascii=False)}],
            "system": ex["system"],
            "tools": json.dumps([t["function"] for t in ex["tools"]], ensure_ascii=False)}


def sharegpt_dataset_info(files: dict[str, str]) -> dict:
    """dataset_info.json for LLaMA-Factory: {dataset name: file name}."""
    return {name: {"file_name": fname, "formatting": "sharegpt",
                   "columns": {"messages": "conversations", "system": "system", "tools": "tools"},
                   "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human",
                            "assistant_tag": "gpt", "observation_tag": "observation",
                            "function_tag": "function_call"}}
            for name, fname in files.items()}


def write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_pilot(out_dir: Path, name: str, examples: list[dict], *, include_text: bool = False,
                include_thinking: bool = False) -> dict:
    """Write one dataset (e.g. name="all", later "train") in both layouts.

        out_dir/pilot/<name>.jsonl         TRL
        out_dir/pilot/<name>.meta.jsonl    id + meta, same row order
        out_dir/pilot_lf/<name>.jsonl      sharegpt
    Returns {path: rows written}.
    """
    out_dir = Path(out_dir)
    written = {}
    p = out_dir / "pilot" / f"{name}.jsonl"
    written[str(p)] = write_jsonl(p, (to_trl(e, include_text=include_text, include_thinking=include_thinking)
                                      for e in examples))
    p = out_dir / "pilot" / f"{name}.meta.jsonl"
    written[str(p)] = write_jsonl(p, ({"id": e["id"], **e["meta"]} for e in examples))
    p = out_dir / "pilot_lf" / f"{name}.jsonl"
    written[str(p)] = write_jsonl(p, (to_sharegpt(e) for e in examples))
    return written


def write_sharegpt_info(out_dir: Path, names: list[str], prefix: str = "avians_pilot") -> Path:
    p = Path(out_dir) / "pilot_lf" / "dataset_info.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(sharegpt_dataset_info({f"{prefix}_{n}": f"{n}.jsonl" for n in names}), indent=2) + "\n")
    return p
