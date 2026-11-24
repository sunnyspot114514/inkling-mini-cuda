"""Regrade prefixes of saved greedy outputs to isolate output-budget sensitivity."""

import argparse
import hashlib
import json
from pathlib import Path

from transformers import PreTrainedTokenizerFast

from evaluate import check_code, completion_code, final_text, load_cases, number, sandbox_command, sandbox_selftest, write_json


def audit(result_dir, tokenizer, data_dir, cap, command):
    suites = {}
    for split in ("gsm8k", "humaneval"):
        _, tasks = load_cases(data_dir, split)
        folder = result_dir / split
        protocol = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
        if not 0 < cap < protocol["max_new_tokens"]:
            raise ValueError("Requested cap must be below the recorded generation budget")
        summary = {"items": len(tasks), "full_correct": 0, "prefix_correct": 0, "outputs_longer_than_cap": 0,
                   "longer_outputs_without_final_text_at_cap": 0, "prefix_only_correct": 0, "full_only_correct": 0}
        for i, task in enumerate(tasks):
            case = json.loads((folder / f"{i:04}.json").read_text(encoding="utf-8"))
            ids = case["output_ids"]
            if case["index"] != i or case["id"] != task.get("task_id", str(i)):
                raise ValueError(f"Task identity mismatch at {split}/{i}")
            decoded = tokenizer.decode(ids, skip_special_tokens=False)
            if decoded != case["completion"] or len(ids) != case["output_tokens"]:
                raise ValueError(f"Token round trip failed for {split}/{i}")
            full = case["passed"]
            if len(ids) > cap:
                prefix = tokenizer.decode(ids[:cap], skip_special_tokens=False)
                summary["outputs_longer_than_cap"] += 1
                summary["longer_outputs_without_final_text_at_cap"] += int("<|content_text|>" not in prefix)
                answer = final_text(prefix)
                if split == "gsm8k":
                    pred, gold = number(answer), number(task["answer"])
                    passed = pred is not None and pred == gold
                else:
                    passed, _ = check_code(command, completion_code(answer, task), task)
            else:
                passed = full
            summary["full_correct"] += int(full)
            summary["prefix_correct"] += int(passed)
            summary["prefix_only_correct"] += int(passed and not full)
            summary["full_only_correct"] += int(full and not passed)
        summary["input_protocol_sha256"] = hashlib.sha256((folder / "protocol.json").read_bytes()).hexdigest()
        suites[split] = summary
    return suites


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("baseline-results", "candidate-results", "tokenizer", "data", "out"):
        p.add_argument("--"+k, type=Path, required=True)
    p.add_argument("--cap", type=int, default=384)
    p.add_argument("--bwrap", default="/usr/bin/bwrap")
    p.add_argument("--wsl-distro", default="Ubuntu-22.04")
    a = p.parse_args()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(a.tokenizer, local_files_only=True)
    command = sandbox_command(a.bwrap, a.wsl_distro)
    sandbox_selftest(command)
    result = {"method": "Prefix-only regrading of stored token IDs with the original local grader",
              "prefix_cap": a.cap, "reference_output_budget": None}
    original_caps = set()
    for name, folder in (("baseline", a.baseline_results), ("candidate", a.candidate_results)):
        result[name] = audit(folder, tokenizer, a.data, a.cap, command)
        for split in ("gsm8k", "humaneval"):
            original_caps.add(json.loads((folder/split/"protocol.json").read_text(encoding="utf-8"))["max_new_tokens"])
    if len(original_caps) != 1:
        raise ValueError("Input reports use different output budgets")
    result["reference_output_budget"] = original_caps.pop()
    a.out.parent.mkdir(exist_ok=True, parents=True)
    write_json(a.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
