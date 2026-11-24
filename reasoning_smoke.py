"""Compare template effort low with saved none responses on 24 fixed, batch-matched problems."""

import argparse
import json
from pathlib import Path
import random

from evaluate import (CODE_PROMPT, GSM_PROMPT, check_code, completion_code, final_text, load_cases,
                      number, sandbox_command, sandbox_selftest, sha, write_json)


def selected_pairs(seed):
    rng = random.Random(seed)
    return {task: sorted(2 * p for p in rng.sample(range(n // 2), 6))
            for task, n in (("gsm8k", 200), ("humaneval", 164))}


def repeat_flags(ids, think, final):
    # A third consecutive identical block is the flag. A channel change ends the current segment.
    segments = {"thinking": [], "text": []}
    current = []
    for token in ids:
        if token in (think, final):
            current = []
            segments["thinking" if token == think else "text"].append(current)
        else:
            current.append(token)
    def repeats(segment):
        for width in (16, 32, 64):
            for start in range(len(segment) - width * 3 + 1):
                block = segment[start:start + width]
                if block == segment[start + width:start + 2*width] == segment[start + 2*width:start + 3*width]:
                    return True
        return False
    return {k: any(repeats(segment) for segment in values) for k, values in segments.items()}


def metrics(case, think, final):
    ids = case["output_ids"]
    flags = repeat_flags(ids, think, final)
    return {"passed": bool(case["passed"]), "output_tokens": len(ids), "hit_limit": bool(case["reached_limit"]),
            "has_thinking": think in ids, "has_final_text": final in ids,
            "thinking_repeat": flags["thinking"], "text_repeat": flags["text"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "library", "data", "formal", "out"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261123)
    p.add_argument("--bwrap", default="/usr/bin/bwrap")
    p.add_argument("--wsl-distro", default="Ubuntu-22.04")
    p.add_argument("--plan-only", action="store_true")
    args = p.parse_args()
    pairs = selected_pairs(args.seed)
    if args.plan_only:
        print(json.dumps(pairs))
        return
    from run_inkling_cuda import generate_prompts, load_model
    args.out.mkdir(parents=True, exist_ok=True)
    spec = {"model": args.model.name, "seed": args.seed, "batch_starts": pairs, "max_new_tokens": 1024,
            "efforts": ["none", "low"], "sample_size": 24, "batch_size": 2, "kv_dtype": "float16",
            "script_sha256": sha(Path(__file__)), "library_sha256": sha(args.library),
            "reference_protocol_sha256": {task:sha(args.formal/task/"protocol.json") for task in pairs},
            "repeat_rule": "Any adjacent identical token block of length 16, 32 or 64 repeated at least three times within the same output channel"}
    protocol_file = args.out/"protocol.json"
    if protocol_file.exists() and json.loads(protocol_file.read_text()) != spec:
        raise ValueError("Existing smoke results use different inputs or settings")
    for task in pairs:
        source = json.loads((args.formal/task/"protocol.json").read_text())
        data_file, _ = load_cases(args.data, task)
        required = {"batch_size": 2, "kv_dtype": "float16", "greedy": True, "reasoning_effort":"none",
                    "max_new_tokens": 1024, "kernel_sha256": sha(args.library),
                    "template_sha256": sha(args.model/"chat_template.jinja"), "data_sha256": sha(data_file),
                    "model_manifest_sha256": sha(args.model/"manifest.json"), "steering_sha256": sha(args.model/"steering.safetensors"),
                    "runner_sha256":sha(Path(__file__).with_name("run_inkling_cuda.py")),
                    "evaluation_sha256":sha(Path(__file__).with_name("evaluate.py"))}
        if any(source[k] != value for k,value in required.items()):
            raise ValueError(f"Different reference protocol for {task}")
    write_json(protocol_file, spec)
    command = sandbox_command(args.bwrap, args.wsl_distro)
    sandbox_selftest(command)
    model, tokenizer = load_model(args.model, args.library, progress=True)
    think = tokenizer.convert_tokens_to_ids("<|content_thinking|>")
    final = tokenizer.convert_tokens_to_ids("<|content_text|>")
    rows = []
    for task, starts in pairs.items():
        _, cases = load_cases(args.data, task)
        for start in starts:
            path = args.out/f"{task}-{start:04}.json"
            if not path.exists():
                batch = cases[start:start+2]
                prompts = []
                for item in batch:
                    prompt = CODE_PROMPT+item["prompt"] if task=="humaneval" else GSM_PROMPT+item["question"]
                    messages = [{"role":"user", "content":prompt}]
                    kwargs = dict(add_generation_prompt=True, reasoning_effort="low")
                    rendered = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
                    if tokenizer.encode(rendered, add_special_tokens=False) != tokenizer.apply_chat_template(
                            messages, tokenize=True, return_dict=True, **kwargs)["input_ids"]:
                        raise ValueError("Rendered prompt token IDs differ from the official chat template")
                    prompts.append(rendered)
                generated = generate_prompts(model, tokenizer, prompts, 1024, raw=True, kv_dtype="float16")
                output = []
                for i, (item, gen) in enumerate(zip(batch, generated, strict=True), start):
                    parent = json.loads((args.formal/task/f"{i:04}.json").read_text(encoding="utf-8"))
                    answer = final_text(gen["completion"])
                    if task == "humaneval":
                        gen["passed"], _ = check_code(command, completion_code(answer,item),item)
                    else:
                        pred = number(answer)
                        gen["passed"] = pred is not None and pred == number(item["answer"])
                    output.append({"task":task, "index":i, "none":metrics(parent,think,final),
                                   "low":metrics(gen,think,final), "low_output_ids":gen["output_ids"]})
                write_json(path, output)
            rows.extend(json.loads(path.read_text(encoding="utf-8")))
    summary = {}
    for task in pairs:
        suite = [r for r in rows if r["task"]==task]
        summary[task] = {"count":len(suite)}
        for effort in spec["efforts"]:
            summary[task][effort] = {key:sum(r[effort][key] for r in suite) for key in suite[0][effort]}
    write_json(args.out/"summary.json", {**spec,"results":summary})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
