"""Export a complete paired scorecard and per-problem grades without machine-local paths."""

import argparse
import hashlib
import json
from math import ceil
from pathlib import Path
from statistics import median

from build_candidate import STATIC_FILES
from compare_runs import compare


UPSTREAM_CARD = "https://huggingface.co/shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-8.9GB/blob/11a9b0f151e066c0f75a30c4b20ab5c48862a528/README.md"
HISTORICAL_CORRECTION = "https://huggingface.co/shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-8.9GB/blob/625429250450b361edf7be4da2230e917b7c4085/README.md"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def runtime_bytes(model):
    names = [*STATIC_FILES, "manifest.json", "steering.safetensors"]
    names.extend(f"layer_{i:02}_{suffix}" for i in range(24) for suffix in ("meta.json", "packed.npz"))
    if (model / "head_residual.json").is_file():
        names.extend(("head_residual.json", "head_residual.safetensors"))
    return sum((model / f).stat().st_size for f in names)


def channel_ids(model):
    decoder = read(model / "tokenizer_config.json")["added_tokens_decoder"]
    ids = {token["content"]: int(index) for index, token in decoder.items() if token.get("special")}
    return ids["<|content_text|>"], ids["<|content_thinking|>"]


def timing(rows, batch, channels):
    total = sum(max(x["batch_elapsed_seconds"] for x in rows[i:i+batch]) for i in range(0,len(rows),batch))
    tokens = sum(x["output_tokens"] for x in rows)
    final_id, thinking_id = channels
    positions = sorted(x["output_ids"].index(final_id) + 1 for x in rows if final_id in x["output_ids"])
    return {"tokens":tokens, "batch_generation_seconds":total, "aggregate_tokens_per_second":tokens/total,
            "limit_reached":sum(x["reached_limit"] for x in rows),
            "limit_without_final_text_marker":sum(x["reached_limit"] and "<|content_text|>" not in x["completion"]
                                                  for x in rows),
            "limit_reached_and_correct":sum(x["reached_limit"] and x["passed"] for x in rows),
            "responses_with_thinking_marker":sum(thinking_id in x["output_ids"] for x in rows),
            "responses_with_final_text_marker":len(positions),
            "tokens_through_first_final_text_marker_median":median(positions) if positions else None,
            "tokens_through_first_final_text_marker_p90":positions[ceil(0.9*len(positions))-1] if positions else None,
            "peak_torch_allocated_bytes":max(x["peak_torch_allocated_bytes"] for x in rows)}


def paired_percentile_interval(n, wins, losses):
    """Percentiles of item-paired empirical resampling; values are percentage points."""
    dist = [1.]
    dec, inc = losses / n, wins / n
    same = (n - wins - losses) / n
    for _ in range(n):
        nxt = [0.] * (len(dist) + 2)
        for j, value in enumerate(dist):
            nxt[j] += value * dec
            nxt[j+1] += value * same
            nxt[j+2] += value * inc
        dist = nxt
    bounds = []
    cumulative = 0.
    for ix, probability in enumerate(dist):
        cumulative += probability
        while len(bounds) < 2 and cumulative >= (0.025, 0.975)[len(bounds)]:
            bounds.append(100 * (ix-n) / n)
    return bounds


def grade(row):
    return {"passed":row["passed"], "tokens":row["output_tokens"], "limit_reached":row["reached_limit"],
            "completion_sha256":hashlib.sha256(row["completion"].encode("utf-8")).hexdigest()}


def paired_budget_groups(baseline, candidate):
    groups = {}
    for parent_hit, child_hit, name in ((False, False, "neither"), (True, False, "baseline_only"),
                                         (False, True, "candidate_only"), (True, True, "both")):
        pairs = [(a["passed"], b["passed"]) for a, b in zip(baseline, candidate, strict=True)
                 if a["reached_limit"] == parent_hit and b["reached_limit"] == child_hit]
        groups[name] = {
            "items": len(pairs), "baseline_correct": sum(a for a, b in pairs),
            "candidate_correct": sum(b for a, b in pairs),
            "baseline_only_correct": sum(a and not b for a, b in pairs),
            "candidate_only_correct": sum(b and not a for a, b in pairs),
        }
    return groups


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("baseline-results", "candidate-results", "baseline-model", "candidate-model", "out"):
        p.add_argument("--"+arg, type=Path, required=True)
    a=p.parse_args()
    sizes={"baseline":runtime_bytes(a.baseline_model), "candidate":runtime_bytes(a.candidate_model)}
    channels = channel_ids(a.baseline_model)
    if channels != channel_ids(a.candidate_model):
        raise ValueError("The checkpoints use different output-channel token IDs")
    recipe=a.candidate_model/"recipe.json"
    if recipe.is_file() and read(recipe)["runtime_file_bytes"]!=sizes["candidate"]:
        raise ValueError("Candidate payload size does not match its recorded recipe")
    result={"runtime_file_bytes":sizes, "bytes_saved":sizes["baseline"]-sizes["candidate"], "tasks":{}}
    items=[]
    lines=["# Paired RTX 3060 evaluation", "", "The candidate and the 8.87 GB LookAtMyQuant baseline ran on the same RTX 3060 12 GB.",
           "", "| Benchmark | Candidate | Baseline | Difference | Candidate-only correct | Baseline-only correct | Exact paired p | Paired 95% interval (pp) |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    runtime_lines = ["| Task | Checkpoint | Output limit reached | Limit hits without final-text marker | Aggregate output tok/s | Peak PyTorch GiB |",
                     "|---|---|---:|---:|---:|---:|"]
    budget_lines = ["| Task | Checkpoint hitting the output limit | Items | Candidate correct | Baseline correct | Candidate-only correct | Baseline-only correct |",
                    "|---|---|---:|---:|---:|---:|---:|"]
    for split,count,label in (("gsm8k",200,"GSM8K"),("humaneval",164,"HumanEval")):
        verdict=compare(a.baseline_results,a.candidate_results,split,count)
        dirs={"baseline":a.baseline_results/split,"candidate":a.candidate_results/split}
        records={k:[read(v/f"{i:04}.json") for i in range(count)] for k,v in dirs.items()}
        protocol=read(dirs["candidate"]/"protocol.json")
        ci = paired_percentile_interval(count, verdict["candidate_only_correct"], verdict["baseline_only_correct"])
        budget_groups = paired_budget_groups(records["baseline"], records["candidate"])
        result["tasks"][split]={**verdict, "paired_percentile_95ci_difference_pp":ci, "paired_budget_groups":budget_groups,
            "runtime":{k:timing(v,protocol["batch_size"],channels) for k,v in records.items()},
            "protocol":{k:protocol[k] for k in ("data_sha256","template_sha256","max_new_tokens","greedy",
                        "reasoning_effort","batch_size","kv_dtype","evaluation_sha256","runner_sha256",
                        "packed_loader_sha256","head_runner_sha256","kernel_sha256","runtime_versions")}}
        change=verdict["candidate"]-verdict["baseline"]
        lines.append(f"| {label} | {verdict['candidate']}/{count} | {verdict['baseline']}/{count} | {change:+d} | "
                     f"{verdict['candidate_only_correct']} | {verdict['baseline_only_correct']} | "
                     f"{verdict['mcnemar_exact_two_sided_p']:.3f} | [{ci[0]:+.2f}, {ci[1]:+.2f}] |")
        for model in ("candidate", "baseline"):
            stats = result["tasks"][split]["runtime"][model]
            runtime_lines.append(f"| {label} | {model} | {stats['limit_reached']}/{count} | "
                                 f"{stats['limit_without_final_text_marker']} | {stats['aggregate_tokens_per_second']:.3f} | "
                                 f"{stats['peak_torch_allocated_bytes']/(1<<30):.3f} |")
        for name, group in budget_groups.items():
            budget_lines.append(f"| {label} | {name.replace('_', ' ')} | {group['items']} | "
                                f"{group['candidate_correct']} | {group['baseline_correct']} | "
                                f"{group['candidate_only_correct']} | {group['baseline_only_correct']} |")
        for i,(baseline,candidate) in enumerate(zip(records["baseline"],records["candidate"])):
            items.append({"benchmark":split,"item":candidate["id"],"index":i,
                          "candidate":grade(candidate),"baseline":grade(baseline)})
    lines.extend(["", *runtime_lines, "",
                  "Both use the packaged official chat template with the generation prompt appended, greedy decoding, "
                  f"`reasoning_effort={protocol['reasoning_effort']}`, `max_new_tokens={protocol['max_new_tokens']}`, "
                  f"batch size {protocol['batch_size']} and {protocol['kv_dtype']} attention K/V storage. "
                  "Throughput counts all returned output tokens over batch generation time, including prompt processing. "
                  "The memory column uses torch.cuda.max_memory_allocated(). The JSON also records the median "
                  "and nearest-rank 90th percentile of the generated token position of the first final-text marker "
                  "among responses that reached that channel. Positions are one-based and include the marker.",
                  "", "## Paired outcomes by output-limit status", "", *budget_lines, "",
                  "Group membership depends on both generated responses. The main benchmark scores use every item. "
                  "These groups locate differences in observed outcomes at the recorded budget. Scores at a longer "
                  "budget require generation beyond the limit.",
                  "",f"The candidate uses {sizes['candidate']:,} bytes of runtime files and the baseline uses "
                     f"{sizes['baseline']:,} bytes, a reduction of {result['bytes_saved']:,} bytes "
                     f"({100*result['bytes_saved']/sizes['baseline']:.2f}%).",
                  "", "The matched counts identify changes in pass/fail status. The reported p value is the two-sided "
                      "exact McNemar test on those changes. The intervals are paired empirical-bootstrap percentiles of the "
                      "candidate-minus-baseline score difference, computed from the full resampling distribution of discordant pairs. "
                      "An equivalence claim would require a prespecified margin. The JSON scorecard includes truncation, aggregate generation throughput, "
                      "software versions and peak PyTorch allocations. Each item in the JSONL report includes the pass/fail result "
                      "and the hash of the generated text for both checkpoints. "
                      "[JSON scorecard](scorecard.json) · [Per-problem grades](items.jsonl)",
                  "", "Under the author's updated 1,536-token protocol the 8.87 GB row reports 192/200 GSM8K, "
                      "149/164 strict HumanEval and 4/4 visual checks. The BF16 row reports 193/200, 152/164 and 4/4. "
                      "The reported 345/349 (98.9%) uses the teacher's total correct as its denominator. The earlier "
                      "384-token evaluation could end during the thinking trace. The local paired scores use the output "
                      f"budget stated above. [Updated upstream table]({UPSTREAM_CARD}) · "
                      f"[Historical correction]({HISTORICAL_CORRECTION})"])
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/"scorecard.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    (a.out/"items.jsonl").write_text("".join(json.dumps(x,separators=(",",":"))+"\n" for x in items),encoding="utf-8")
    (a.out/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"Wrote {len(items)} paired grades to {a.out}")


if __name__=="__main__":
    main()
