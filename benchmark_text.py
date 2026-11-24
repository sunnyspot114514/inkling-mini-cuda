"""Measure single-sequence prefill, decode and context capacity in the packed CUDA runner."""

import argparse
from contextlib import contextmanager
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import struct
import subprocess
import threading
import time

import torch

from evaluate import sha, write_json
from run_inkling_cuda import HalfKVCache, load_model
from wddm_memory import ProcessGpuMemory


INTERACTIONS = {
    "english": "In two sentences, explain why snow muffles sound.",
    "chinese": "请用两句话说明为什么盛夏的树荫下通常比阳光直射处凉快。",
    "math": "A baker sells 14 loaves every weekday and 9 on each weekend day. How many loaves are sold in one week?",
    "python": "Write a Python function named runs that counts contiguous runs in a list. Include one example.",
}


def token_hash(ids):
    return hashlib.sha256(struct.pack(f"<{len(ids)}I", *ids)).hexdigest()


def memory():
    free, total = torch.cuda.mem_get_info()
    return {"torch_allocated_bytes": torch.cuda.memory_allocated(),
            "torch_reserved_bytes": torch.cuda.memory_reserved(),
            "torch_allocator_ceiling_bytes": round(torch.cuda.get_per_process_memory_fraction()*total),
            "device_free_bytes": free, "device_total_bytes": total}


def board_conditions():
    fields=("driver_version","pstate","temperature.gpu","clocks.current.sm","clocks.current.memory",
            "power.limit","power.draw","utilization.gpu","memory.used")
    try:
        proc=subprocess.run(["nvidia-smi","-i",str(torch.cuda.current_device()),"--query-gpu="+",".join(fields),
                             "--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=8)
    except (OSError,subprocess.TimeoutExpired) as exc:
        return {"unavailable":type(exc).__name__}
    values=proc.stdout.strip().split(",")
    if proc.returncode or len(values)!=len(fields):
        return {"unavailable":"nvidia-smi did not return the requested counters"}
    result={}
    for key,value in zip(fields,values,strict=True):
        value=value.strip()
        if key in ("driver_version","pstate"):
            result[key]=value
        else:
            try:
                result[key]=float(value)
            except ValueError:
                result[key]=None
    return result


def timing_policy():
    """Identify power and CPU limits separately from variable clock, temperature and memory samples."""
    cpu_percent=None
    if os.name=="nt":
        rate=(ctypes.c_uint32*2)()
        ok=ctypes.windll.kernel32.QueryInformationJobObject(None,15,ctypes.byref(rate),ctypes.sizeof(rate),None)
        if ok and rate[0]&0x5==0x5:
            cpu_percent=rate[1]/100
    return {"board_power_ceiling_w":board_conditions().get("power.limit"),
            "windows_job_cpu_ceiling_percent":cpu_percent,
            "requested_forward_active_fraction":float(os.environ.get("INKLING_GPU_ACTIVE_FRACTION","1")),
            "pacing_scope":os.environ.get("INKLING_TIMING_SCOPE","none")}


def residency():
    result=memory()
    try:
        meter=ProcessGpuMemory(os.getpid())
    except (OSError, AttributeError) as exc:
        result["wddm_unavailable"]=str(exc)
        return result
    readings=[]
    try:
        for i in range(3):
            if i:
                time.sleep(.5)
            sample=meter.read()
            if sample is not None:
                readings.append(sample)
    except OSError as exc:
        result["wddm_error"]=str(exc)
    finally:
        meter.close()
    result["wddm_idle_samples"]=len(readings)
    if readings:
        for field in ("dedicated_bytes","shared_bytes"):
            result["wddm_median_"+field]=median(r[field] for r in readings)
    return result


@contextmanager
def sampled_wddm():
    result = {"interval_s": 0.5, "samples": 0, "peak_dedicated_bytes": None, "peak_shared_bytes": None}
    try:
        meter = ProcessGpuMemory(os.getpid())
    except (OSError, AttributeError) as exc:
        result["unavailable"] = str(exc)
        yield result
        return
    stop = threading.Event()

    def sample():
        try:
            while True:
                reading = meter.read()
                if reading is not None:
                    result["samples"] += 1
                    for key, value in reading.items():
                        result["peak_" + key] = max(result["peak_" + key] or 0, value)
                if stop.wait(result["interval_s"]):
                    break
        except OSError as exc:
            result["counter_error"] = str(exc)

    worker = threading.Thread(target=sample, daemon=True)
    worker.start()
    try:
        yield result
    finally:
        stop.set()
        worker.join()
        meter.close()


def cache_footprint(cache):
    groups={"global_kv":[],"sliding_kv":[],"short_convolution":[]}
    conv_slots=0
    for layer,sliding in zip(cache.layers,cache.is_sliding,strict=True):
        groups["sliding_kv" if sliding else "global_kv"].extend((layer.keys,layer.values))
        states=[v for v in layer.conv_states.values() if v is not None]
        conv_slots+=bool(states)
        groups["short_convolution"].extend(states)
    total_storage={}
    result={}
    for name,tensors in groups.items():
        storage={}
        logical_bytes=0
        for tensor in tensors:
            backing=tensor.untyped_storage()
            storage[(tensor.device,backing.data_ptr())]=backing.nbytes()
            logical_bytes+=tensor.numel()*tensor.element_size()
        result[name]={"tensors":len(tensors),"logical_bytes":logical_bytes,"storage_bytes":sum(storage.values())}
        total_storage.update(storage)
    result.update(global_slots=len(groups["global_kv"])//2,sliding_slots=len(groups["sliding_kv"])//2,
                  convolution_slots=conv_slots,unique_storage_bytes=sum(total_storage.values()))
    return result


@torch.inference_mode()
def trace(model, ids, steps, *, stop_at_eos, prefill_chunk=64, cache_metrics=None):
    """Return per-token arrival times. Fixed-step runs include one first prediction plus subsequent steps."""
    inputs = torch.tensor([ids], dtype=torch.long, device="cuda")
    cache = HalfKVCache(config=model.config)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for offset in range(0, len(ids), prefill_chunk):
        out = model(input_ids=inputs[:, offset:offset + prefill_chunk], cache=cache, use_cache=True, logits_to_keep=1)
        cache = out.cache
    torch.cuda.synchronize()
    prefill = time.perf_counter() - start
    current = out.logits[:, -1].argmax(-1, keepdim=True)
    tokens = [current.item()]
    arrivals = [time.perf_counter() - start]
    for _ in range(steps):
        if stop_at_eos and tokens[-1] == model.config.eos_token_id:
            break
        out = model(input_ids=current, cache=cache, use_cache=True, logits_to_keep=1)
        cache = out.cache
        current = out.logits[:, -1].argmax(-1, keepdim=True)
        tokens.append(current.item())
        arrivals.append(time.perf_counter() - start)
    expected = len(ids) + len(tokens) - 1
    if len(cache.layers) != model.config.total_num_hidden_layers:
        raise RuntimeError("The looped decoder did not create separate cache slots for every logical visit")
    for layer, sliding in enumerate(cache.is_sliding):
        if not sliding and cache.get_seq_length(layer) != expected:
            raise RuntimeError(f"Global cache slot {layer} does not contain {expected} positions")
    if cache_metrics is not None:
        cache_metrics.update(cache_footprint(cache))
    return tokens, arrivals, prefill


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "library", "corpus", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--lengths", type=int, nargs="*", default=[512, 4096])
    p.add_argument("--capacity-lengths", type=int, nargs="*", default=[16384, 32768])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--decode-steps", type=int, default=128)
    p.add_argument("--interactive", action="store_true", help="Measure first final-text arrival on four chat prompts")
    args = p.parse_args()
    lengths = args.lengths + args.capacity_lengths
    if not lengths or min(*lengths, args.repeats, args.decode_steps) < 1 or len(lengths) != len(set(lengths)):
        p.error("Supply distinct positive lengths, a positive repeat count and a positive decode length")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    protocol = {"model": args.model.name, "manifest_sha256": sha(args.model/"manifest.json"),
                "kernel_sha256": sha(args.library), "corpus_sha256": sha(args.corpus),
                "script_sha256": sha(Path(__file__)), "lengths": args.lengths, "capacity_lengths": args.capacity_lengths,
                "repeats": args.repeats, "decode_steps": args.decode_steps, "prefill_chunk": 64,
                "kv_dtype": "float16", "batch_size": 1, "eos_policy": "continue greedy for fixed decode steps",
                "interactive":args.interactive, "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
                "timing_policy":timing_policy()}
    if args.out.exists():
        report = json.loads(args.out.read_text(encoding="utf-8"))
        if report["protocol"] != protocol:
            raise ValueError("Existing report has a different benchmark protocol")
    else:
        report = {"protocol": protocol, "loads": [], "trials": []}
    before = memory()
    with sampled_wddm() as load_os:
        start = time.perf_counter()
        model, tokenizer = load_model(args.model, args.library, progress=True)
        torch.cuda.synchronize()
        load_s = time.perf_counter() - start
        resident = residency()
    report["loads"].append({"seconds": load_s, "before": before, "loaded": resident, "wddm": load_os})
    write_json(args.out, report)
    ids = tokenizer.encode(args.corpus.read_text(encoding="utf-8"), add_special_tokens=False)
    if len(ids) < max(lengths):
        raise ValueError(f"Corpus provides {len(ids)} tokens for a {max(lengths)}-token test")
    trace(model, ids[:64], 4, stop_at_eos=False)
    for length in lengths:
        repeats = args.repeats if length in args.lengths else 1
        for rep in range(repeats):
            if any((row["input_tokens"], row["repeat"]) == (length, rep) for row in report["trials"]):
                continue
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            row = {"input_tokens": length, "repeat": rep, "input_ids_sha256": token_hash(ids[:length]),
                   "before": memory(),"board_before":board_conditions()}
            with sampled_wddm() as os_memory:
                try:
                    cache_metrics={}
                    tokens, arrivals, prefill = trace(model, ids[:length], args.decode_steps, stop_at_eos=False,
                                                      cache_metrics=cache_metrics)
                    row.update(status="ok", cached_positions=length + args.decode_steps,
                               final_cache=cache_metrics,
                               output_ids_sha256=token_hash(tokens), prefill_s=prefill, prefill_tok_s=length/prefill,
                               time_to_first_model_token_s=arrivals[0], decode_s=arrivals[-1]-arrivals[0],
                               decode_tok_s=args.decode_steps/(arrivals[-1]-arrivals[0]),
                               first_eos_position=next((i+1 for i,t in enumerate(tokens) if t==model.config.eos_token_id), None))
                except torch.cuda.OutOfMemoryError as exc:
                    row.update(status="out_of_memory", error=str(exc))
            row.update(wddm=os_memory, peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(), after=memory(),
                       board_after=board_conditions())
            report["trials"].append(row)
            write_json(args.out, report)
            print(f"{args.model.name} pp{length} repeat {rep+1}: {row['status']}", flush=True)
    summary = {}
    for length in lengths:
        rows = [r for r in report["trials"] if r["input_tokens"] == length and r["status"] == "ok"]
        if rows:
            summary[str(length)] = {
                "completed_repeats": len(rows), "median_prefill_tok_s": median(r["prefill_tok_s"] for r in rows),
                "median_decode_tok_s": median(r["decode_tok_s"] for r in rows),
                "median_ttft_s": median(r["time_to_first_model_token_s"] for r in rows),
                "peak_torch_allocated_bytes": max(r["peak_torch_allocated_bytes"] for r in rows),
            }
    report["summary"] = summary
    if args.interactive:
        recorded = report.setdefault("interactive", {})
        final_id = tokenizer.convert_tokens_to_ids("<|content_text|>")
        for label,prompt in INTERACTIONS.items():
            if label in recorded:
                continue
            input_ids = tokenizer.apply_chat_template([{"role":"user","content":prompt}],tokenize=True,
                        add_generation_prompt=True,reasoning_effort="none",return_dict=True)["input_ids"]
            tokens,arrivals,prefill = trace(model,input_ids,1023,stop_at_eos=True)
            first_text = None
            if final_id in tokens:
                channel = tokens.index(final_id)+1
                for pos in range(channel,len(tokens)):
                    if tokenizer.decode(tokens[channel:pos+1],skip_special_tokens=True).strip():
                        first_text=pos
                        break
            recorded[label] = {"input_tokens":len(input_ids),"input_ids_sha256":token_hash(input_ids),
                               "output_tokens_including_eos":len(tokens),"output_ids_sha256":token_hash(tokens),
                               "stopped_at_eos":tokens[-1]==model.config.eos_token_id,"prefill_s":prefill,
                               "first_model_token_s":arrivals[0],"first_nonblank_final_text_s":arrivals[first_text]
                               if first_text is not None else None,"completion_s":arrivals[-1]}
            write_json(args.out,report)
    write_json(args.out, report)


if __name__ == "__main__":
    main()
