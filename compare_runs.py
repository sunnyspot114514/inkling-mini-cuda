"""Compare complete, paired GSM8K-200 and HumanEval-164 runs."""

import argparse
import json
from math import comb
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare(baseline, candidate, split, count):
    first, second = baseline / split, candidate / split
    left, right = read(first / "protocol.json"), read(second / "protocol.json")
    pair_settings = ("split", "data_sha256", "max_new_tokens", "greedy", "reasoning_effort",
                     "template_sha256", "kernel_sha256", "runner_sha256", "packed_loader_sha256",
                     "evaluation_sha256", "total_cases", "batch_size", "item_limit", "kv_dtype", "head_runner_sha256",
                     "runtime_versions")
    if any(left[k] != right[k] for k in pair_settings):
        raise ValueError(f"Different protocols for {split}")
    if left["total_cases"] != count or left["item_limit"] not in (None, count):
        raise ValueError(f"Expected the full {split} test split")
    wins = losses = baseline_score = candidate_score = 0
    for ix in range(count):
        a, b = read(first / f"{ix:04}.json"), read(second / f"{ix:04}.json")
        if a["id"] != b["id"] or a["index"] != ix or b["index"] != ix:
            raise ValueError(f"Task ID mismatch at {split} item {ix}")
        baseline_score += a["passed"]
        candidate_score += b["passed"]
        wins += int(b["passed"] and not a["passed"])
        losses += int(a["passed"] and not b["passed"])
    n = wins + losses
    p = min(1.0, 2 * sum(comb(n, x) for x in range(min(wins, losses) + 1)) / (2 ** n))
    return {"n": count, "baseline": baseline_score, "candidate": candidate_score,
            "candidate_only_correct": wins, "baseline_only_correct": losses,
            "mcnemar_exact_two_sided_p": p}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    results = {split: compare(args.baseline, args.candidate, split, n)
               for split, n in (("gsm8k", 200), ("humaneval", 164))}
    print(json.dumps(results, indent=2))
