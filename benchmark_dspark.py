"""Test an unmodified target and a target-specific DSpark sidecar in the same upstream native CUDA engine."""

import argparse
import gc
import json
from pathlib import Path
import time

import torch

from benchmark_text import board_conditions, residency, sampled_wddm, timing_policy, token_hash
from evaluate import load_cases, sha, write_json


PROSE = (
    ("Explain why a thermos keeps a drink warm.", "A thermos slows heat transfer in three ways. "),
    ("Explain to a gardener how a rain shadow forms.", "When moist air reaches a mountain range, "),
    ("Describe how a library can find a misfiled book.", "The librarian starts by checking "),
    ("Explain why a bicycle feels different in a low gear.", "A bicycle's gears change "),
)


def workload(tokenizer, data):
    _, tasks = load_cases(data, "humaneval")
    prompts = []
    for i in (2, 15, 34, 70):
        item = tasks[i]
        prompts.append(("code", item["task_id"], "Complete this Python function.", "```python\n"+item["prompt"]))
    prompts.extend(("prose", f"prose/{i}", prompt, prefix) for i,(prompt,prefix) in enumerate(PROSE))
    prepared = []
    for domain, label, prompt, prefix in prompts:
        chat = tokenizer.apply_chat_template([{"role":"user","content":prompt}], tokenize=False,
                                             add_generation_prompt=True, reasoning_effort="none")
        ids = tokenizer.encode(chat+"<|message_model|><|content_text|>"+prefix, add_special_tokens=False)
        prepared.append((domain, label, ids))
    return prepared


def measure(model, drafter, ids, limit, eos, block):
    from runtime_cuda import stream_tokens
    stats = {}
    torch.cuda.synchronize()
    start = time.perf_counter()
    first = None
    output = []
    for token in stream_tokens(model, ids, max_tokens=limit, eos_ids=eos, drafter=drafter,
                               draft_block_size=block, temperature=0., stats_out=stats):
        output.append(int(token))
        if first is None:
            first = time.perf_counter()
    torch.cuda.synchronize()
    stop = time.perf_counter()
    if first is None:
        raise ValueError("Native engine returned no tokens")
    return {"tokens":output, "seconds":stop-start, "ttft_s":first-start, "decode_s":stop-first,
            "stats":stats}


def context_probe(model, draft, ids, limit, block):
    torch.cuda.reset_peak_memory_stats()
    with sampled_wddm() as counters:
        try:
            trial=measure(model,draft,ids,limit,set(),block)
            if len(trial["tokens"])!=limit:
                raise RuntimeError("The fixed-length native context probe ended early")
            result={"status":"ok","target_context":len(ids)+limit-1,"output_ids_sha256":token_hash(trial["tokens"]),
                    "request_seconds":trial["seconds"],"native_stats":trial["stats"]}
        except (RuntimeError, MemoryError) as exc:
            result={"status":"failed","error":f"{type(exc).__name__}: {exc}"}
    result.update(wddm=counters,peak_torch_allocated_bytes=torch.cuda.max_memory_allocated())
    torch.cuda.empty_cache()
    return result


def measured_mount(loader,model,sidecar,context):
    before=residency()
    with sampled_wddm() as wddm:
        start=time.perf_counter()
        try:
            draft=loader(model,sidecar,context=context)
            torch.cuda.synchronize()
        except RuntimeError as exc:
            if not isinstance(exc,torch.cuda.OutOfMemoryError) and not str(exc).endswith("failed with CUDA_ERROR_OUT_OF_MEMORY"):
                raise
            return None,{"status":"out_of_memory","drafter_s":time.perf_counter()-start,"target_only":before,
                         "wddm_load":wddm,"error":str(exc)}
        elapsed=time.perf_counter()-start
        loaded=residency()
    return draft,{"status":"mounted","drafter_s":elapsed,"target_only":before,
                  "target_with_drafter":loaded,"wddm_load":wddm}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "drafter", "data", "out"):
        p.add_argument("--"+name, required=True, type=Path)
    p.add_argument("--block", type=int, choices=(2,4,8), default=8)
    p.add_argument("--limit", type=int, default=128)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--context", type=int, default=4096)
    args = p.parse_args()
    if min(args.limit, args.repeats, args.context) < 1:
        p.error("Limits and repeat count must be positive")
    args.out.parent.mkdir(exist_ok=True, parents=True)
    spec = {"model":args.model.name, "model_manifest_sha256":sha(args.model/"manifest.json"),
            "draft_filename":args.drafter.name, "draft_bytes":args.drafter.stat().st_size,
            "draft_sha256":sha(args.drafter), "upstream_revision":"9dc5cbf7adb08cc41b5ba45c6313556e16f30bdc",
            "script_sha256":sha(Path(__file__)), "block":args.block, "limit":args.limit,
            "repeats":args.repeats, "drafter_context":args.context,
            "workload":"Greedy continuations with explicit assistant final-text prefixes, one request at a time"}
    if (args.model/"head_residual.json").exists():
        write_json(args.out, {"protocol":spec, "status":"target_incompatible",
                              "reason":"The stock native verifier omits this checkpoint's required rank-24 output correction"})
        return
    import runtime_cuda
    from runtime_cuda import eos_ids, load_drafter, open_model
    spec["timing_policy"]=timing_policy()
    engine_dir=Path(runtime_cuda.__file__).parent
    native_files=("device_model.py","device_ptx.py","drafter.py","driver.py","matvec.ptx","model.py","packed.py")
    spec["engine_source_sha256"]={name:sha(engine_dir/name) for name in
                                 ("__init__.py",*("inkling_cuda/"+f for f in native_files))}
    if args.out.exists():
        report = json.loads(args.out.read_text(encoding="utf-8"))
        if report["protocol"] != spec:
            raise ValueError("Existing DSpark result has different settings")
    else:
        report = {"protocol":spec, "trials":[], "loads":[]}
    report["status"]="running"
    report.pop("reason",None)
    report.pop("error",None)
    try:
        with open_model(args.model, engine="native", device="cuda:0") as (model, tokenizer):
            prompts = workload(tokenizer, args.data)
            if max(len(ids) for _,_,ids in prompts) + args.limit - 1 > args.context:
                raise ValueError("A workload exceeds the requested drafter context")
            eos = eos_ids(model, tokenizer)
            gsm_data=args.data/"gsm8k-test.jsonl"
            corpus="\n".join(json.loads(line)["question"] for line in gsm_data.read_text(encoding="utf-8").splitlines())
            width=args.context-args.limit+1
            full_ids=tokenizer.encode(corpus,add_special_tokens=False)[:width]
            if len(full_ids)!=width:
                raise ValueError("Native context probe corpus is shorter than its requested input")
            cap=report.setdefault("context_probe",{"input_tokens":width,"input_ids_sha256":token_hash(full_ids),
                       "corpus_sha256":sha(gsm_data),"output_limit":args.limit,"eos_policy":"continue to fixed length"})
            if "serial" not in cap:
                cap["serial"]=context_probe(model,None,full_ids,args.limit,args.block)
                write_json(args.out,report)
            draft,load=measured_mount(load_drafter,model,args.drafter,args.context)
            report["loads"].append(load)
            if load["status"]=="out_of_memory":
                report.update(status="drafter_out_of_memory",reason="GPU allocation failed while loading the drafter at the tested context")
                write_json(args.out,report)
                return
            write_json(args.out, report)
            for sidecar in (None, draft):
                measure(model, sidecar, prompts[0][2], 8, eos, args.block)
            for rep in range(args.repeats):
                for index, (domain,label,ids) in enumerate(prompts):
                    if any(r["id"]==label and r["repeat"]==rep for r in report["trials"]):
                        continue
                    result = {}
                    order = (False, True) if (rep+index)%2==0 else (True, False)
                    for enabled in order:
                        before=board_conditions()
                        trial=measure(model,draft if enabled else None,ids,args.limit,eos,args.block)
                        trial.update(board_before=before,board_after=board_conditions())
                        result["draft" if enabled else "serial"]=trial
                    a,b=result["serial"]["tokens"], result["draft"]["tokens"]
                    mismatch=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),None)
                    if mismatch is None and len(a)!=len(b):
                        mismatch=min(len(a),len(b))
                    report["trials"].append({"id":label, "domain":domain, "repeat":rep, "input_tokens":len(ids),
                                            "input_ids_sha256":token_hash(ids), "exact_greedy_match":a==b,
                                            "first_mismatch_zero_based":mismatch, **result})
                    write_json(args.out, report)
                    print(label, rep+1, "parity", a==b, flush=True)
            if "draft" not in cap:
                cap["draft"]=context_probe(model,draft,full_ids,args.limit,args.block)
                if cap["serial"]["status"]==cap["draft"]["status"]=="ok":
                    cap["exact_greedy_match"]=cap["serial"]["output_ids_sha256"]==cap["draft"]["output_ids_sha256"]
                write_json(args.out,report)
            del draft
            gc.collect()
        report["status"] = "complete"
        report["domains"] = {}
        for domain in ("code","prose"):
            rows=[r for r in report["trials"] if r["domain"]==domain]
            accepted=sum(r["draft"]["stats"]["accepted_tokens"] for r in rows)
            proposed=sum(r["draft"]["stats"]["proposed_tokens"] for r in rows)
            checks=sum(r["draft"]["stats"]["draft_checks"] for r in rows)
            matched=all(r["exact_greedy_match"] for r in rows)
            if not 0 <= accepted <= proposed:
                raise ValueError("Invalid upstream draft acceptance counters")
            times={side:sum(r[side]["seconds"] for r in rows) for side in ("serial","draft")}
            throughput={}
            for side in ("serial","draft"):
                total_tokens=sum(len(r[side]["tokens"]) for r in rows)
                decode_s=sum(r[side]["decode_s"] for r in rows)
                throughput[side]={"tokens_including_eos":total_tokens,"request_seconds":times[side],
                                  "request_tok_s":total_tokens/times[side],
                                  "decode_tok_s":(total_tokens-len(rows))/decode_s if decode_s else None}
            ratio=times["serial"]/times["draft"]
            report["domains"][domain]={"trials":len(rows),"all_streams_match":matched,
                                      "matched_streams":sum(r["exact_greedy_match"] for r in rows),"draft_checks":checks,
                                      "accepted":accepted,"proposed":proposed,"acceptance":accepted/proposed if proposed else None,
                                      "anchors_plus_accepted_per_check":1+accepted/checks if checks else None,
                                      "throughput":throughput, "observed_request_time_ratio":ratio,
                                      "paired_e2e_speedup":ratio if matched else None}
        write_json(args.out, report)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(args.out, report)
        raise


if __name__ == "__main__":
    main()
