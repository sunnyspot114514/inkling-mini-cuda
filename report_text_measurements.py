"""Build a public comparison from the completed RTX 3060 text measurement files."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

import numpy as np


MODELS = {"baseline":"8.87 GB parent", "two-layer":"8.63 GB", "expert":"8.35 GB"}
GIB = 1 << 30
MIB = 1 << 20


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value, digits=2, divisor=1):
    return "n/a" if value is None else f"{value/divisor:.{digits}f}"


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "|"+"|".join("---" for _ in headers)+"|",
            *("| "+" | ".join(map(str,row))+" |" for row in rows), ""]


def sampled_peak(trials, field):
    values=[r["wddm"].get(field) for r in trials if r["wddm"].get(field) is not None]
    return max(values) if values else None


def paired_window_kl(reference, candidate, domains):
    """Pair both checkpoints on identical token windows and preserve parallel XNLI source groups."""
    a,b=reference["windows"],candidate["windows"]
    if a.keys()!=b.keys() or reference["protocol"]["reference"]!=candidate["protocol"]["reference"]:
        raise ValueError("KL inputs or teacher distributions differ")
    by_domain={}
    for domain in domains:
        selected=sorted(k for k in a if a[k]["domain"]==domain)
        if not selected:
            raise ValueError(f"No KL windows in {domain}")
        if any(a[k]["token_ids_sha256"]!=b[k]["token_ids_sha256"] or
               a[k]["positions"]!=96 or b[k]["positions"]!=96 or b[k]["domain"]!=domain for k in selected):
            raise ValueError("KL positions or document identities differ")
        by_domain[domain]={k:b[k]["mean_kl_nats"]-a[k]["mean_kl_nats"] for k in selected}
    differences=[delta for rows in by_domain.values() for delta in rows.values()]
    result={"windows":len(differences),"difference_nats":float(np.mean(differences))}
    if all(d.startswith("xnli_") for d in domains):
        source_starts=(100,2600)
        parallel={start:[] for start in source_starts}
        for domain,rows in by_domain.items():
            prefix="xnli/test/"+domain.removeprefix("xnli_")+"/row-"
            if set(rows)!={prefix+str(start) for start in source_starts}:
                raise ValueError(f"Unexpected parallel XNLI source blocks for {domain}")
            for start in source_starts:
                parallel[start].append(rows[prefix+str(start)])
        result["parallel_source_start_deltas"]={str(start):float(np.mean(values)) for start,values in parallel.items()}
        return result
    rng=np.random.default_rng(20261123)
    resamples=np.zeros(10000,dtype=np.float64)
    for rows in by_domain.values():
        values=np.asarray(list(rows.values()))
        draws=rng.integers(len(values),size=(len(resamples),len(values)))
        resamples+=values[draws].sum(axis=1)
    result["paired_stratified_window_95ci"]=np.percentile(resamples/len(differences),[2.5,97.5]).tolist()
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--measurements",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    src=args.measurements
    report={"models":{},"speculation":{}}
    speed_rows,capacity_rows,latency_rows,kl_rows,effort_rows=[],[],[],[],[]
    trial_records=[]
    shared_inputs={}
    kl_data={name:read(src/f"kl-{name}.json") for name in MODELS}
    for name,label in MODELS.items():
        speed=read(src/f"speed-{name}.json")
        capacity=read(src/f"capacity-{name}.json")
        effort=read(src/f"effort-{name}/summary.json")
        kl=kl_data[name]
        metrics={"contexts":{}, "interactive":speed["interactive"], "kl":kl["groups"],"effort":effort["results"],
                 "timing_policy":speed["protocol"].get("timing_policy",{})}
        report["models"][name]=metrics
        protocols=[r["protocol"] for r in (speed,capacity)]
        if any(p["decode_steps"]!=128 or p["batch_size"]!=1 or p["kv_dtype"]!="float16" for p in protocols):
            raise ValueError("Expected the frozen single-sequence capacity protocol")
        if speed["protocol"]["lengths"]!=[512,4096] or capacity["protocol"]["capacity_lengths"]!=[16384,32768]:
            raise ValueError("Unexpected throughput or capacity lengths")
        resident=speed["loads"][-1]["loaded"]
        metrics["resident"]=resident
        for length in (512,4096,16384,32768):
            run=speed if length<=4096 else capacity
            trials=[r for r in run["trials"] if r["input_tokens"]==length]
            if len(trials)!=(3 if length<=4096 else 1):
                raise ValueError(f"Incomplete {name} trials at {length} tokens")
            for row in trials:
                signature=(row["input_ids_sha256"],run["protocol"]["kernel_sha256"],run["protocol"]["corpus_sha256"],
                           run["protocol"]["gpu"],run["protocol"]["torch"],run["protocol"]["prefill_chunk"])
                if shared_inputs.setdefault(length,signature)!=signature:
                    raise ValueError(f"Different input or kernel at {length}")
                fields=("input_tokens","repeat","status","input_ids_sha256","output_ids_sha256","cached_positions","final_cache",
                        "prefill_s","decode_s","time_to_first_model_token_s","prefill_tok_s","decode_tok_s",
                        "peak_torch_allocated_bytes","peak_torch_reserved_bytes","wddm","board_before","board_after")
                trial_records.append({"measurement":"single_sequence","target":name,
                                      "allocator_ceiling_bytes":row["before"].get("torch_allocator_ceiling_bytes"),
                                      "device_free_before_bytes":row["before"]["device_free_bytes"],
                                      **{k:row[k] for k in fields if k in row}})
            ok=[r for r in trials if r["status"]=="ok"]
            ceilings=sorted({r["before"]["torch_allocator_ceiling_bytes"] for r in trials
                             if r["before"].get("torch_allocator_ceiling_bytes") is not None})
            context={"successful_trials":len(ok),"trials":len(trials),
                     "outcomes":[r["status"] for r in trials],"input_ids_sha256":trials[0]["input_ids_sha256"],
                     "allocator_ceilings_bytes":ceilings,
                     "peak_torch_bytes":max(r["peak_torch_allocated_bytes"] for r in trials),
                     "peak_reserved_bytes":max(r["peak_torch_reserved_bytes"] for r in trials),
                     "peak_dedicated_bytes":sampled_peak(trials,"peak_dedicated_bytes"),
                     "peak_shared_bytes":sampled_peak(trials,"peak_shared_bytes")}
            if ok:
                if any(r["cached_positions"]!=length+128 for r in ok):
                    raise ValueError("The decode reserve was not consumed")
                context.update(prefill_tok_s=median(r["prefill_tok_s"] for r in ok),
                               decode_tok_s=median(r["decode_tok_s"] for r in ok),
                               ttft_s=median(r["time_to_first_model_token_s"] for r in ok),
                               stable_repeat_ids=len({r["output_ids_sha256"] for r in ok})==1)
                caches=[r["final_cache"] for r in ok if "final_cache" in r]
                if caches:
                    context["largest_final_cache"]=max(caches,key=lambda c:c["unique_storage_bytes"])
                for field in ("temperature.gpu","clocks.current.sm","power.draw","memory.used"):
                    values=[r.get(when,{}).get(field) for r in ok for when in ("board_before","board_after")]
                    valid=[v for v in values if v is not None]
                    if valid:
                        context.setdefault("board_snapshot_ranges",{})[field]=[min(valid),max(valid)]
            metrics["contexts"][str(length)]=context
            if length>=4096:
                completed=f"{len(ok)}/{len(trials)}"
                oom=context["outcomes"].count("out_of_memory")
                if oom:
                    completed+=f" ({oom} OOM)"
                limits=", ".join(fmt(v,3,GIB) for v in ceilings) or "n/a"
                cache_bytes=context.get("largest_final_cache",{}).get("unique_storage_bytes")
                capacity_rows.append([label,f"{length:,} + 128",completed,limits,fmt(cache_bytes,1,MIB),
                    fmt(context.get("peak_torch_bytes"),3,GIB),fmt(context.get("peak_reserved_bytes"),3,GIB),
                    fmt(context.get("peak_dedicated_bytes"),3,GIB),fmt(context.get("peak_shared_bytes"),1,MIB),
                    fmt(context.get("decode_tok_s"))])
        short,long=metrics["contexts"]["512"],metrics["contexts"]["4096"]
        speed_rows.append([label,fmt(resident["torch_allocated_bytes"],3,GIB),
                           fmt(resident.get("wddm_median_dedicated_bytes"),3,GIB),
                           fmt(short.get("prefill_tok_s")),fmt(short.get("decode_tok_s")),
                           fmt(long.get("prefill_tok_s")),fmt(long.get("decode_tok_s"))])
        probes=list(metrics["interactive"].values())
        if len(probes)!=4:
            raise ValueError("Incomplete interactive workloads")
        final=[r["first_nonblank_final_text_s"] for r in probes if r["first_nonblank_final_text_s"] is not None]
        latency_rows.append([label,f"{len(final)}/4",fmt(median(r["first_model_token_s"] for r in probes)),
                             fmt(median(final) if final else None),fmt(median(r["completion_s"] for r in probes))])
        for group,title in (("english_python_tutoring","English, Python and tutoring chat"),
                            ("xnli_13","XNLI, 13 languages"),("xnli_zh","XNLI, Chinese")):
            vals=kl["groups"][group]
            kl_rows.append([label,title,vals["positions"],fmt(vals["mean_kl_nats"],5),fmt(vals["top1_agreement"]*100,2)])
        for task,suite in effort["results"].items():
            n0,n1=suite["none"],suite["low"]
            count=suite["count"]
            if count!=12:
                raise ValueError("Incomplete effort sample")
            pair=lambda k:f"{n0[k]} / {n1[k]}"
            task_title={"gsm8k":"GSM8K","humaneval":"HumanEval"}[task]
            effort_rows.append([label,task_title,count,pair("passed"),pair("hit_limit"),
                    f"{count-n0['has_final_text']} / {count-n1['has_final_text']}",
                    pair("thinking_repeat"),pair("text_repeat")])
    delta_rows=[]
    report["paired_kl"]={}
    mult=sorted(d for d in kl_data["baseline"]["groups"] if d.startswith("xnli_") and d!="xnli_13")
    if len(mult)!=13:
        raise ValueError("Expected thirteen XNLI languages")
    for name in ("two-layer","expert"):
        report["paired_kl"][name]={}
        for label,domains in (("English / Python / tutoring",["english","python","tutoring_chat"]),("XNLI-13",mult)):
            delta=paired_window_kl(kl_data["baseline"],kl_data[name],domains)
            report["paired_kl"][name][label]=delta
            interval=delta.get("paired_stratified_window_95ci")
            text=f"[{interval[0]:+.5f}, {interval[1]:+.5f}]" if interval else "n/a"
            starts=delta.get("parallel_source_start_deltas",{})
            delta_rows.append([MODELS[name],label,delta["windows"],f"{delta['difference_nats']:+.5f}",text,
                               fmt(starts.get("100"),5),fmt(starts.get("2600"),5)])
    draft_rows,draft_memory=[],[]
    for key in ("dspark-baseline","dspark-expert","dspark-two-layer","dspark-native-baseline","dspark-native-expert"):
        data=read(src/(key+".json"))
        status=data["status"]
        model="two-layer" if key.endswith("two-layer") else key.rsplit("-",1)[-1]
        label=MODELS[model]
        side="native BF16" if key.startswith("dspark-native-") else "8.9-target 4-bit"
        summary={"status":status,"target":label,"sidecar":side,"draft_bytes":data["protocol"]["draft_bytes"]}
        if "timing_policy" in data["protocol"]:
            summary["timing_policy"]=data["protocol"]["timing_policy"]
        report["speculation"][key]=summary
        if status!="complete":
            summary["reason"]=data.get("reason") or data.get("error","").split(":",1)[0]
            display={"target_incompatible":"Rank-24 correction unsupported",
                     "drafter_out_of_memory":"Drafter load OOM"}.get(status,"Run failed")
            if status=="drafter_out_of_memory":
                load=data["loads"][-1]
                target=load["target_only"]
                serial=data["context_probe"]["serial"]
                if load["status"]!="out_of_memory":
                    raise ValueError("The drafter memory result differs from its recorded load attempt")
                summary["load"]={k:load[k] for k in ("status","drafter_s","target_only","wddm_load")}
                summary["context_probe"]={"serial":{"status":serial["status"],"memory":serial["wddm"]}}
                draft_memory.append([label,side,fmt(target.get("wddm_median_dedicated_bytes"),3,GIB),"Load OOM",
                                     fmt(serial["wddm"]["peak_dedicated_bytes"],3,GIB) if serial["status"]=="ok" else serial["status"],
                                     "n/a","n/a"])
            draft_rows.append([label,side,display,"n/a","n/a","n/a","n/a"])
            continue
        summary["domains"]=data["domains"]
        summary["load"]=data["loads"][-1]
        if len(data["trials"])!=sum(v["trials"] for v in data["domains"].values()):
            raise ValueError("DSpark summary counts disagree with the trial records")
        for trial in data["trials"]:
            record={"measurement":"dspark","target":model,"drafter":side,
                    **{k:trial[k] for k in ("id","domain","repeat","input_tokens","input_ids_sha256","exact_greedy_match")}}
            for engine in ("serial","draft"):
                timed=trial[engine]
                record[engine]={k:timed[k] for k in ("seconds","ttft_s","decode_s","board_before","board_after") if k in timed}
                record[engine].update(output_tokens=len(timed["tokens"]),output_ids_sha256=hashlib.sha256(
                    np.asarray(timed["tokens"],dtype="<u4").tobytes()).hexdigest())
                if engine=="draft":
                    record[engine]["verification_counts"]={k:timed["stats"][k] for k in
                        ("accepted_tokens","proposed_tokens","draft_checks")}
            if record["exact_greedy_match"]!=(record["serial"]["output_ids_sha256"]==record["draft"]["output_ids_sha256"]):
                raise ValueError("DSpark parity label differs from the observed output token IDs")
            trial_records.append(record)
        for domain,values in data["domains"].items():
            draft_rows.append([label,side,domain,f"{values['accepted']}/{values['proposed']}",
                                fmt(100*values["acceptance"] if values["acceptance"] is not None else None,1),
                                f"{values['matched_streams']}/{values['trials']}",fmt(values["paired_e2e_speedup"])])
        cap=data["context_probe"]
        summary["context_probe"]={k:cap[k] for k in ("input_tokens","output_limit","exact_greedy_match") if k in cap}
        loaded=summary["load"]
        drow=[label,side,fmt(loaded["target_only"].get("wddm_median_dedicated_bytes"),3,GIB),
              fmt(loaded["target_with_drafter"].get("wddm_median_dedicated_bytes"),3,GIB)]
        for engine in ("serial","draft"):
            sub=cap[engine]
            summary["context_probe"][engine]={"status":sub["status"],"memory":sub["wddm"]}
            drow.append(fmt(sub["wddm"]["peak_dedicated_bytes"],3,GIB) if sub["status"]=="ok" else sub["status"])
        equality=cap.get("exact_greedy_match")
        drow.append("n/a" if equality is None else "yes" if equality else "different outputs")
        draft_memory.append(drow)
    policy_rows=[]
    for model,label in MODELS.items():
        metrics=report["models"][model]
        policy=metrics["timing_policy"]
        policy_rows.append([label,fmt(policy.get("board_power_ceiling_w"),0),
                            fmt(policy.get("windows_job_cpu_ceiling_percent")),fmt(policy.get("requested_forward_active_fraction")),
                            fmt(metrics["resident"].get("torch_allocator_ceiling_bytes"),3,GIB)])
    paced=any(v["timing_policy"].get("requested_forward_active_fraction",1)<1 for v in report["models"].values())
    timing_note=("The reported throughput, latency and DSpark request times include the test machine's deliberate "
                 "cooldown waits between CUDA work groups." if paced else "These trials use the timing settings listed below.")
    lines=["# RTX 3060 12 GB text measurements","",
           "The three packed checkpoints use the same Windows 10 machine and the same input token IDs at each tested length. "
           "The speed columns use one active sequence. The 512- and 4,096-token values are medians of three warm trials. "+timing_note,"",
           *table(["Target","GPU power ceiling, W","Windows process-tree CPU cap, %","Requested forward duty fraction",
                   "Loaded process allocator ceiling, GiB"],policy_rows),
           *table(["Target","Loaded PyTorch GiB","Idle WDDM dedicated GiB","PP512 tok/s","Decode after PP512 tok/s",
                   "PP4096 tok/s","Decode after PP4096 tok/s"],speed_rows),
           "## Context capacity","",
           *table(["Target","Input + decode positions","Completed","Allocator ceiling GiB","Final cache MiB","Peak torch GiB","Peak reserved GiB",
                   "Peak WDDM dedicated GiB","Peak WDDM shared MiB","Decode tok/s"],capacity_rows),
           "Each completed trial consumed the stated input and 128 further decode steps. Memory peaks include all attempts. "
           "An OOM records the outcome at the listed process allocator ceiling and available device memory. "
           "The final cache counts distinct storage backing the target's K/V and short-convolution tensors after a successful trial. "
           "This memory is included in PyTorch's allocations. WDDM peaks are sampled every 0.5 seconds. "
           "Decode rates use completed trials.","","## Chat latency","",
           *table(["Target","Reached final text","Median first model token, s","Median first nonblank final text, s",
                   "Median completion, s"],latency_rows),
           "The final-text median uses responses that reached that channel. The four prompts cover English, Chinese, arithmetic and Python.",
           "","## Full-vocabulary KL against the native teacher","",
           *table(["Target","Public proxy","Positions","Mean KL, nats","Top-1 agreement, %"],kl_rows),
           *table(["Derivative vs 8.87 GB","Proxy","Paired windows","Mean KL difference, nats",
                   "Window bootstrap 95% interval, nats","XNLI row 100 delta","XNLI row 2600 delta"],delta_rows),
           "Each 128-token window contributes 96 scored positions. The English, Python and tutoring interval uses 10,000 "
           "paired window resamples stratified by domain. XNLI has two source starting rows shared by all 13 languages. "
           "The last two columns give each starting row's mean difference across languages, in nats.","","## Reasoning effort","",
           "Each pair of counts gives effort none followed by effort low at the same 1,024-token budget.","",
           *table(["Target","Task","Items","Correct","Hit limit","Missing final marker","Thinking repetition flags",
                   "Final-text repetition flags"],effort_rows),
           "## Direct DSpark comparison","",
           *table(["Target","Drafter","Workload or status","Accepted / proposed","Acceptance, %","Exact greedy streams",
                   "Matched request speedup"],draft_rows),
           *table(["Target","Drafter","Idle target GiB","Idle with drafter GiB","4K target-only peak GiB",
                   "4K with drafter peak GiB","4K greedy match"],draft_memory),
           "The DSpark time ratio uses the same native PTX engine and identical verified output IDs. The 4K capacity probe "
           "fills a 4,096-position target session and records WDDM memory with and without the drafter. The 8.63 GB target "
           "requires its rank-24 correction in the verifier. The model packages expose text generation. Vision VQA has no local score.",
           "","[Machine-readable summary](summary.json) · [Per-trial timing and telemetry](trials.jsonl)"]
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"summary.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    with (args.out/"trials.jsonl").open("w",encoding="utf-8") as stream:
        for trial in trial_records:
            stream.write(json.dumps(trial,separators=(",",":"))+"\n")
    (args.out/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
