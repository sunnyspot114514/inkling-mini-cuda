"""Extend capped greedy responses in their original batches and verify the saved token prefixes."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from evaluate import run_split, sha, write_json


MORE_BATCHES=3


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def windows(starts, batch_size):
    groups = []
    for start in starts:
        if groups and start == groups[-1][1]:
            groups[-1][1] += batch_size
        else:
            groups.append([start,start+batch_size])
    return groups


def verify_prefix(prior,after,task,index):
    prefix=after["output_ids"][:len(prior["output_ids"])] if prior["reached_limit"] else after["output_ids"]
    if prefix!=prior["output_ids"] or prior["id"]!=after["id"]:
        raise ValueError(f"Greedy prefix mismatch at {task}/{index}; a full independent run is needed")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("formal","model","data","library","out"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--max-new-tokens",type=int,default=1536)
    p.add_argument("--bwrap",default="/usr/bin/bwrap")
    p.add_argument("--wsl-distro",default="Ubuntu-22.04")
    p.add_argument("--plan-only",action="store_true")
    p.add_argument("--batches-per-process",type=int,help="Exit with status 3 after this many new original batches if work remains")
    args=p.parse_args()
    remaining=args.batches_per_process
    if remaining is not None and remaining<1:
        p.error("The number of batches per process must be positive")
    if args.out.resolve().is_relative_to(args.formal.resolve()):
        p.error("Choose a separate destination for the extension")
    model,tokenizer=None,None
    results={}
    items=[]
    for task,n in (("gsm8k",200),("humaneval",164)):
        source=args.formal/task
        spec=read(source/"protocol.json")
        old=[read(source/f"{i:04}.json") for i in range(n)]
        if not (spec["greedy"] and spec["total_cases"]==n and spec["item_limit"] in (None,n)
                and args.max_new_tokens>spec["max_new_tokens"]):
            raise ValueError("The source must be a complete greedy evaluation with a smaller output limit")
        width=spec["batch_size"]
        starts=sorted({(i//width)*width for i,row in enumerate(old) if row["reached_limit"]})
        print(task, "capped",sum(r["reached_limit"] for r in old),"batches to replay",len(starts),flush=True)
        if args.plan_only:
            continue
        dest=args.out/"replayed"/task
        run_args=SimpleNamespace(model=args.model,library=args.library,data=args.data,out=dest,split=task,
                                 batch_size=width,kv_dtype=spec["kv_dtype"],max_new_tokens=args.max_new_tokens,
                                 bwrap=args.bwrap,wsl_distro=args.wsl_distro,start_index=0,stop_index=0,limit=None)
        # Materialize the exact new protocol without generating a case.
        model,tokenizer=run_split(run_args,model,tokenizer)
        extended_spec=read(dest/"protocol.json")
        ignored={"max_new_tokens","model"}
        if {k:v for k,v in spec.items() if k not in ignored}!={k:v for k,v in extended_spec.items() if k not in ignored}:
            raise ValueError("The replay would change an input, model fingerprint, executable or decoder setting")
        missing=[start for start in starts if any(not (dest/f"{i:04}.json").is_file()
                                                  for i in range(start,min(start+width,n)))]
        selected=missing if remaining is None else missing[:remaining]
        for start,end in windows(selected,width):
            run_args.start_index,run_args.stop_index=start,end
            model,tokenizer=run_split(run_args,model,tokenizer)
        if remaining is not None:
            remaining-=len(selected)
            if len(selected)<len(missing):
                for start in selected:
                    for i in range(start,min(start+width,n)):
                        verify_prefix(old[i],read(dest/f"{i:04}.json"),task,i)
                print("Saved and verified completed replay batches. More original batches remain.",flush=True)
                return MORE_BATCHES
        replayed={i for start in starts for i in range(start,min(start+width,n))}
        for i,prior in enumerate(old):
            after=read(dest/f"{i:04}.json") if i in replayed else prior
            if i in replayed:
                verify_prefix(prior,after,task,i)
            items.append({"task":task,"index":i,"id":prior["id"],"replayed":i in replayed,
                          "original_correct":prior["passed"], "extended_correct":after["passed"],
                          "original_tokens":prior["output_tokens"], "extended_tokens":after["output_tokens"],
                          "original_capped":prior["reached_limit"],"extended_capped":after["reached_limit"]})
        suite=[r for r in items if r["task"]==task]
        results[task]={"items":n,"source_protocol_sha256":sha(source/"protocol.json"),
                       "extended_protocol_sha256":sha(dest/"protocol.json"),"original_cap":spec["max_new_tokens"],
                       "replayed_batches":len(starts),"replayed_sequences":len(replayed),
                       **{k:sum(r[k] for r in suite) for k in ("original_correct","extended_correct","original_capped","extended_capped")}}
        write_json(args.out/"items.json",items)
        write_json(args.out/"summary.json",{"output_limit":args.max_new_tokens,"tasks":results,
                    "complete":len(results)==2,"all_replayed_original_prefixes_verified":True})


if __name__=="__main__":
    raise SystemExit(main())
