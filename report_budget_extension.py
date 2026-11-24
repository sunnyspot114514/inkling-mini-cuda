"""Summarize completed 1,536-token extensions and their paired discordances."""

import argparse
import json
from math import comb
from pathlib import Path

from evaluate import sha, write_json
from report_results import paired_percentile_interval


MODELS = {"baseline":"8.87 GB parent", "two-layer":"8.63 GB two-layer", "expert":"8.35 GB expert mix"}
TASKS = {"gsm8k":("GSM8K",200), "humaneval":("HumanEval",164)}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verified_placement(folder,items):
    path=folder/"execution-placement.json"
    if not path.is_file():
        return None
    evidence=read(path)
    expected={f"{t}/{i:04}.json":r for (t,i),r in items.items() if r["replayed"]}
    resident=evidence["preexisting_fully_gpu_resident_items"]
    mapped=evidence["items"]
    if resident.keys()&mapped.keys() or not (resident.keys()|mapped.keys())<=expected.keys():
        raise ValueError("Placement receipt contains duplicate or unexpected replay IDs")
    for key,row in (resident|mapped).items():
        if (row["saved_record_sha256"]!=sha(folder/"replayed"/key)
                or row["verified_original_tokens"]!=expected[key]["original_tokens"]
                or row["extended_tokens"]!=expected[key]["extended_tokens"]):
            raise ValueError("Placement receipt does not match the saved replay")
    return {"fully_resident_replays":len(resident),"mapped_host_replays":len(mapped),
            "unattributed_replays":len(expected)-len(resident)-len(mapped),
            "mapped_run_active_fraction":evidence["active_fraction"],**evidence["placement"]}


def load_extension(folder):
    summary = read(folder/"summary.json")
    if summary["output_limit"] != 1536 or not summary["complete"] or not summary["all_replayed_original_prefixes_verified"]:
        raise ValueError(f"Incomplete 1,536-token extension in {folder.name}")
    rows = read(folder/"items.json")
    by_key = {(x["task"],x["index"]):x for x in rows}
    if len(rows)!=len(by_key) or len(by_key)!=sum(n for _,n in TASKS.values()):
        raise ValueError("Each task and index must appear exactly once")
    for task,(_,count) in TASKS.items():
        suite=[by_key[task,i] for i in range(count)]
        totals=summary["tasks"][task]
        if totals["items"]!=count or totals["original_cap"]!=1024:
            raise ValueError(f"Unexpected source population or budget for {task}")
        for name in ("original_correct","extended_correct","original_capped","extended_capped"):
            if totals[name]!=sum(x[name] for x in suite):
                raise ValueError(f"Stored {task} total disagrees with the per-item records")
        needed={i//2 for i,row in enumerate(suite) if row["original_capped"]}
        if any(row["replayed"]!=(i//2 in needed) for i,row in enumerate(suite)):
            raise ValueError("The replays must preserve every affected original batch")
        if totals["replayed_batches"]!=len(needed) or totals["replayed_sequences"]!=sum(x["replayed"] for x in suite):
            raise ValueError("The reported replay coverage differs from the item records")
        for row in suite:
            if not row["replayed"] and (row["original_correct"]!=row["extended_correct"] or
                                       row["original_tokens"]!=row["extended_tokens"] or row["extended_capped"]):
                raise ValueError("A reused EOS-complete response changed")
    return summary,by_key


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("baseline","expert","formal-comparison","out"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--two-layer",type=Path,help="Include a completed 8.63 GB extension")
    args=p.parse_args()
    dirs={"baseline":args.baseline}
    if args.two_layer:
        dirs["two-layer"]=args.two_layer
    dirs["expert"]=args.expert
    original=read(args.formal_comparison)
    sizes={"baseline":8874207051,"two-layer":8631359648,"expert":8349693242}
    formal={r["runtime_bytes"]:r for r in original["models"]}
    stats={}
    items={}
    for model,folder in dirs.items():
        stats[model],items[model]=load_extension(folder)
        placement=verified_placement(folder,items[model])
        if placement:
            stats[model]["execution_placement"]=placement
        for task in TASKS:
            if stats[model]["tasks"][task]["original_correct"]!=formal[sizes[model]][task+"_correct"]:
                raise ValueError("The 1,024-token evidence differs from the published formal comparison")
    shared=("split","data_sha256","template_sha256","max_new_tokens","greedy","reasoning_effort","kv_dtype",
            "batch_size","kernel_sha256","runner_sha256","packed_loader_sha256","head_runner_sha256",
            "evaluation_sha256","runtime_versions")
    for task in TASKS:
        protocols=[read(folder/"replayed"/task/"protocol.json") for folder in dirs.values()]
        if any(any(p[k]!=protocols[0][k] for k in shared) for p in protocols[1:]):
            raise ValueError(f"The extended {task} protocols differ across checkpoints")
    lines=["# Paired output-budget extension on the RTX 3060","",
           "| Model | GSM8K at 1,024 | GSM8K at 1,536 | Still capped at 1,536 | HumanEval at 1,024 | HumanEval at 1,536 | Still capped at 1,536 |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for model in dirs:
        cells=[MODELS[model]]
        for task,(_,count) in TASKS.items():
            suite=stats[model]["tasks"][task]
            cells.extend(f"{suite[k]}/{count}" for k in ("original_correct","extended_correct","extended_capped"))
        lines.append("| "+" | ".join(cells)+" |")
    lines.extend(["","| Difference at 1,536 | Task | First only | Second only | Grade changes with either model capped | Difference (pp) | Paired 95% interval (pp) | Exact paired p |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"])
    contrasts=[]
    for first,second in (("expert","baseline"),("two-layer","baseline"),("expert","two-layer")):
        if first not in dirs or second not in dirs:
            continue
        for task,(task_name,count) in TASKS.items():
            wins=losses=cap_changes=0
            uncapped_items=first_uncapped_correct=second_uncapped_correct=0
            for i in range(count):
                a,b=items[first][task,i],items[second][task,i]
                if a["id"]!=b["id"]:
                    raise ValueError("Task identities differ")
                wins+=int(a["extended_correct"] and not b["extended_correct"])
                losses+=int(b["extended_correct"] and not a["extended_correct"])
                if a["extended_capped"] or b["extended_capped"]:
                    cap_changes+=int(a["extended_correct"]!=b["extended_correct"])
                else:
                    uncapped_items+=1
                    first_uncapped_correct+=a["extended_correct"]
                    second_uncapped_correct+=b["extended_correct"]
            n=wins+losses
            prob=min(1.,2*sum(comb(n,k) for k in range(min(wins,losses)+1))/(2**n))
            ci=paired_percentile_interval(count,wins,losses)
            delta=100*(wins-losses)/count
            contrasts.append({"first":first,"second":second,"task":task,"first_only":wins,"second_only":losses,
                              "discordant_with_either_capped":cap_changes,
                              "neither_capped":{"items":uncapped_items,"first_correct":first_uncapped_correct,
                                               "second_correct":second_uncapped_correct},
                              "difference_pp":delta,"paired_95ci_pp":ci,"exact_mcnemar_p":prob})
            cap_detail=f"{cap_changes}/{n}" if n else "No grade changes"
            lines.append(f"| {MODELS[first]} vs {MODELS[second]} | {task_name} | {wins} | {losses} | {cap_detail} | "
                         f"{delta:+.2f} | [{ci[0]:+.2f}, {ci[1]:+.2f}] | {prob:.3f} |")
    lines.extend(["","Both budgets use identical local prompts, the packaged chat template, greedy decoding, effort "
                  "`none`, FP16 K/V and batch size two. Each original batch containing a capped response was replayed. "
                  "The full original greedy prefix was checked on every replayed sequence, including its batch companion. "
                  "Responses already ending at EOS in other batches retained their original grade. The local graders "
                  "and the author's independently reported scores have separate provenance. "
                  "[Per-item outcomes](items.jsonl) · [Score and replay counts](summary.json)"])
    for label,suite in stats.items():
        placement=suite.get("execution_placement")
        if placement:
            lines.extend(["",f"For {MODELS[label]}, {placement['fully_resident_replays']} replayed sequences used fully "
                          f"GPU-resident weights and {placement['mapped_host_replays']} used CUDA-mapped system RAM for "
                          f"{placement['expert_pairs']} expert pairs ({placement['host_bytes']:,} bytes). "
                          f"The mapped runs used a {placement['mapped_run_active_fraction']:.0%} duty target. "
                          f"Placement was unrecorded for {placement['unattributed_replays']} replays."])
    args.out.mkdir(parents=True,exist_ok=True)
    write_json(args.out/"summary.json",{"output_limit":1536,"models":stats,"paired_comparisons":contrasts})
    with (args.out/"items.jsonl").open("w",encoding="utf-8") as fp:
        for task,(_,count) in TASKS.items():
            for i in range(count):
                record={"task":task,"index":i,"id":items["baseline"][task,i]["id"]}
                record.update({model:{key:val for key,val in items[model][task,i].items()
                                      if key not in ("task","index","id")} for model in dirs})
                fp.write(json.dumps(record,separators=(",",":"))+"\n")
    (args.out/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
