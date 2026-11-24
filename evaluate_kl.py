"""Cache native BF16 reference distributions, then measure paired teacher-to-packed KL on one GPU."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import struct
from unittest.mock import patch

import numpy as np
import torch

from evaluate import sha, write_json


NATIVE_REVISION = "ed40e7da625f594fc9c4990f5f66be24873c22d7"
NATIVE_SHA = "65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1"


def manifest(directory, protocol):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/"protocol.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Existing distribution cache uses different weights, inputs or settings")
    write_json(path, protocol)


def expert_device_map(blueprint, execution_device):
    """Use disjoint module roots so an ancestor hook cannot claim an offloaded expert parameter."""
    names={f"model.language_model.layers.{i}.mlp.experts" for i in range(len(blueprint.model.language_model.layers))}
    placements={}
    def visit(prefix, module):
        if prefix in names:
            placements[prefix]="disk"
            return
        if prefix and not any(n.startswith(prefix+".") for n in names):
            placements[prefix]=execution_device
            return
        if any(True for _ in module.parameters(recurse=False)):
            raise ValueError(f"Offload ancestor {prefix} has direct parameters that need placement")
        for child,submodule in module.named_children():
            visit(f"{prefix}.{child}" if prefix else child,submodule)
    visit("",blueprint)
    return placements


@contextmanager
def original_expert_scales(checkpoint):
    """Preserve the native FP32 NVFP4 factors when the HF loader casts its source tensors."""
    from transformers.core_model_loading import Inkling2LoopDequantizeExperts
    standard=Inkling2LoopDequantizeExperts.convert
    restored=set()
    with checkpoint.open("rb") as raw:
        header_length,=struct.unpack("<Q",raw.read(8))
        if not 0<header_length<32_000_000:
            raise ValueError("Invalid native safetensors header")
        header=json.loads(raw.read(header_length))
        def convert(op,input_dict,source_patterns,target_patterns,model=None,full_layer_name=None,**kwargs):
            if len(source_patterns)==4 and input_dict.get(source_patterns[2]) is not None:
                match=re.search(r"(?:^|\.)layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$",full_layer_name)
                if match is None:
                    raise ValueError(f"Unrecognized native expert name {full_layer_name}")
                layer,side=int(match[1]),match[2]
                name="w13_weight" if side=="gate_up_proj" else "w2_weight"
                base=f"model.llm.layers.{layer}.mlp.experts.{name}"
                first=input_dict[source_patterns[0]]
                device=(first[0] if isinstance(first,list) else first).device
                input_dict=dict(input_dict)
                for pattern,suffix in zip(source_patterns[2:],("_amax","_outer_scale_denominator"),strict=True):
                    meta=header[base+suffix]
                    begin,end=meta["data_offsets"]
                    if meta["dtype"]!="F32" or end-begin!=int(np.prod(meta["shape"]))*4:
                        raise ValueError(f"Unexpected native factor type for {base+suffix}")
                    raw.seek(8+header_length+begin)
                    value=torch.from_numpy(np.frombuffer(raw.read(end-begin),dtype="<f4").copy().reshape(meta["shape"]))
                    input_dict[pattern]=value.to(device)
                restored.add((layer,side))
            return standard(op,input_dict,source_patterns,target_patterns,model=model,full_layer_name=full_layer_name,**kwargs)
        with patch.object(Inkling2LoopDequantizeExperts,"convert",convert):
            yield restored


def native_model(path, offload):
    from transformers import AutoConfig
    from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopForConditionalGeneration
    # Keep the original BF16 expert tensors on disk between physical layer visits.
    # Accelerate stages the two projections of the active layer on CUDA during its forward pass.
    config=AutoConfig.from_pretrained(path,local_files_only=True)
    with torch.device("meta"):
        blueprint=Inkling2LoopForConditionalGeneration(config)
    device_map=expert_device_map(blueprint,0)
    del blueprint
    offload.mkdir(parents=True, exist_ok=True)
    with original_expert_scales(path/"model-00001-of-00001.safetensors") as restored:
        model, info = Inkling2LoopForConditionalGeneration.from_pretrained(
            path, local_files_only=True, dtype=torch.bfloat16, device_map=device_map,
            offload_folder=str(offload), attn_implementation="eager", experts_implementation="eager",
            output_loading_info=True)
    expected={(i,side) for i in range(20) for side in ("gate_up_proj","down_proj")}
    if restored!=expected:
        raise ValueError(f"Expected all 40 native NVFP4 factor pairs, observed {len(restored)}")
    for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if info.get(name):
            raise RuntimeError(f"Native model loading mismatch in {name}: {str(info[name])[:600]}")
    return model.eval()


def lp_chunks(model, hidden, vocab):
    for start in range(0, len(hidden), 8):
        logits = model.logit_softcap(model.lm_head(hidden[start:start+8]).float())
        if logits.shape[-1] != vocab:
            raise ValueError(f"Wrong unpadded vocabulary size {logits.shape[-1]}")
        yield start, torch.log_softmax(logits, dim=-1)


def cache_teacher(model, hidden, path, vocab):
    temp = path.with_suffix(".part.npy")
    array = np.lib.format.open_memmap(temp, mode="w+", dtype="<f4", shape=(len(hidden), vocab))
    try:
        for offset, lp in lp_chunks(model, hidden, vocab):
            values = lp.cpu().numpy()
            if not np.isfinite(values).all():
                raise FloatingPointError("A teacher log probability is nonfinite")
            array[offset:offset+len(values)] = values
        array.flush()
    finally:
        del array
    os.replace(temp, path)


def compare_window(model, hidden, path, vocab):
    teacher = np.load(path, mmap_mode="r", allow_pickle=False)
    if teacher.shape != (len(hidden), vocab) or teacher.dtype != np.dtype("<f4"):
        raise ValueError(f"Invalid teacher distribution {path.name}")
    divergences, matches = [], []
    for offset, lp in lp_chunks(model, hidden, vocab):
        ref = torch.from_numpy(np.array(teacher[offset:offset+len(lp)],copy=True)).to(lp.device)
        if not torch.isfinite(lp).all() or not torch.allclose(ref.logsumexp(-1), torch.zeros(len(lp),device=lp.device), atol=3e-5):
            raise FloatingPointError("Invalid next-token distribution")
        kls = (ref.exp() * (ref-lp)).sum(-1,dtype=torch.float64)
        divergences.extend(kls.tolist())
        matches.extend((ref.argmax(-1)==lp.argmax(-1)).tolist())
    if min(divergences) < -1e-5 or not np.isfinite(divergences).all():
        raise FloatingPointError("Invalid KL divergence")
    return {"positions":len(matches), "mean_kl_nats":float(np.mean(divergences)),
            "mean_top1_agreement":float(np.mean(matches)), "per_position_kl_nats":divergences,
            "per_position_top1_agreement":matches}


@torch.inference_mode()
def main():
    import transformers
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("teacher","score"))
    for name in ("model","windows","teacher-cache"):
        p.add_argument("--"+name, required=True, type=Path)
    p.add_argument("--library", type=Path)
    p.add_argument("--offload", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()
    if args.batch_size < 1 or (args.mode=="teacher" and not args.offload) or (args.mode=="score" and not(args.library and args.out)):
        p.error("Choose a positive batch size, provide --offload for teacher or --library and --out for score")
    data=json.loads(args.windows.read_text(encoding="utf-8"))
    if sha(args.model/"tokenizer.json")!=data["tokenizer_sha256"] or sha(args.model/"chat_template.jinja")!=data["chat_template_sha256"]:
        raise ValueError("The reference and candidate need the token IDs and chat template used to freeze the windows")
    lo,hi,vocab=(data[k] for k in ("score_position_start","score_position_stop","vocab_size"))
    refspec = {"windows_sha256":sha(args.windows), "native_revision":NATIVE_REVISION,"native_weights_sha256":NATIVE_SHA,
               "evaluation_sha256":sha(Path(__file__)),
               "teacher":"Native checkpoint, Transformers Inkling2Loop BF16 conversion with source FP32 NVFP4 amax and denominator",
               "torch":str(torch.__version__),"transformers":transformers.__version__,
               "real_vocabulary":vocab,"score_position_start":lo,"score_position_stop":hi,
               "stored_dtype":"little-endian float32 log probabilities","attention":"eager, full forced window, no KV cache"}
    if args.mode=="teacher":
        if sha(args.model/"model-00001-of-00001.safetensors") != NATIVE_SHA:
            raise ValueError("Native weight file differs from the pinned original")
        manifest(args.teacher_cache, refspec)
        model=native_model(args.model,args.offload)
        todo=[w for w in data["windows"] if not (args.teacher_cache/(w["token_ids_sha256"]+".npy")).is_file()]
    else:
        if json.loads((args.teacher_cache/"protocol.json").read_text())!=refspec:
            raise ValueError("The teacher and candidate evaluation protocols differ")
        teacher_run=json.loads((args.teacher_cache/"complete.json").read_text(encoding="utf-8"))
        if teacher_run["windows"]!=len(data["windows"]) or teacher_run["positions"]!=(hi-lo)*len(data["windows"]):
            raise ValueError("The teacher cache does not cover all frozen positions")
        from run_inkling_cuda import load_model
        model,_=load_model(args.model,args.library,progress=True)
        settings={"reference":refspec,"model":args.model.name,"manifest_sha256":sha(args.model/"manifest.json"),
                  "teacher_configured_batch_size":teacher_run["configured_batch_size"],
                  "target_configured_batch_size":args.batch_size,
                  "kernel_sha256":sha(args.library),"head_correction_sha256":sha(args.model/"head_residual.safetensors")
                  if (args.model/"head_residual.safetensors").is_file() else None}
        args.out.parent.mkdir(parents=True,exist_ok=True)
        report=json.loads(args.out.read_text()) if args.out.exists() else {"protocol":settings,"windows":{}}
        if report["protocol"]!=settings:
            raise ValueError("Existing KL score has different inputs")
        todo=[w for w in data["windows"] if w["id"] not in report["windows"]]
    for start in range(0,len(todo),args.batch_size):
        batch=todo[start:start+args.batch_size]
        tokens=torch.tensor([w["ids"] for w in batch],dtype=torch.long,device="cuda")
        hidden=model.model(input_ids=tokens,use_cache=False).last_hidden_state
        for index,w in enumerate(batch):
            ref=args.teacher_cache/(w["token_ids_sha256"]+".npy")
            states=hidden[index,lo:hi]
            if args.mode=="teacher":
                cache_teacher(model,states,ref,vocab)
            else:
                row=compare_window(model,states,ref,vocab)
                report["windows"][w["id"]]={"domain":w["domain"],"token_ids_sha256":w["token_ids_sha256"],**row}
                write_json(args.out,report)
        print(f"{args.mode} {start+len(batch)}/{len(todo)} windows",flush=True)
    if args.mode=="teacher":
        for w in data["windows"]:
            arr=np.load(args.teacher_cache/(w["token_ids_sha256"]+".npy"),mmap_mode="r",allow_pickle=False)
            if arr.shape!=(hi-lo,vocab) or arr.dtype!=np.dtype("<f4"):
                raise ValueError(f"Invalid completed reference {w['id']}")
        write_json(args.teacher_cache/"complete.json",{"windows":len(data["windows"]), "positions":(hi-lo)*len(data["windows"]),
                   "configured_batch_size":args.batch_size,
                   "peak_torch_allocated_bytes":torch.cuda.max_memory_allocated()})
        return
    domains=sorted({w["domain"] for w in data["windows"]})
    groups={d:[d] for d in domains}
    groups["english_python_tutoring"]=["english","python","tutoring_chat"]
    groups["xnli_13"]=[d for d in domains if d.startswith("xnli_")]
    report["groups"]={}
    for label,members in groups.items():
        rows=[r for r in report["windows"].values() if r["domain"] in members]
        numbers=[v for r in rows for v in r["per_position_kl_nats"]]
        agreement=[v for r in rows for v in r["per_position_top1_agreement"]]
        report["groups"][label]={"windows":len(rows),"positions":len(numbers),"mean_kl_nats":float(np.mean(numbers)),
                                 "median_kl_nats":float(np.median(numbers)),"top1_agreement":float(np.mean(agreement))}
    write_json(args.out,report)


if __name__=="__main__":
    main()
