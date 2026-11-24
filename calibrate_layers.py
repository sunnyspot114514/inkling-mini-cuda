"""Screen alternate layer assignments on labelled GSM8K/MBPP training documents."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import re

import numpy as np
from safetensors import safe_open
import torch
import torch.nn.functional as F

from evaluate import CODE_PROMPT, GSM_PROMPT, write_json
from packed_cuda import CudaOps, PackedExperts, PackedLinear
from run_inkling_cuda import load_model


def examples(data, tok, samples, seed, excluded):
    rng = random.Random(seed)
    result = []
    for domain, file in (("math", "gsm8k-train.jsonl"), ("code", "mbpp-train.jsonl")):
        rows = list(enumerate(json.loads(line) for line in (data / file).read_text(encoding="utf-8").splitlines()))
        rng.shuffle(rows)
        count = 0
        for ix, item in rows:
            case_id = str(item.get("task_id", ix))
            if (domain, case_id) in excluded:
                continue
            if domain == "math":
                if 400 <= ix < 432:
                    continue
                prompt = GSM_PROMPT + item["question"]
                response = re.sub(r"<<[^>]*>>", "", item["answer"])
            else:
                if item["task_id"] <= 632:
                    continue
                prompt = CODE_PROMPT + item["text"]
                response = "```python\n" + item["code"].strip() + "\n```"
            context = tok.apply_chat_template([{"role": "user", "content": prompt}], reasoning_effort="none",
                                              add_generation_prompt=True, return_dict=True)["input_ids"]
            context += tok.encode("<|message_model|><|content_text|>", add_special_tokens=False)
            answer = tok.encode(response, add_special_tokens=False)
            if len(answer) < 8 or len(context) + len(answer) > 224:
                continue
            positions = np.linspace(0, len(answer) - 1, min(24, len(answer)), dtype=np.int64)
            result.append((domain, case_id, torch.tensor([context + answer], device="cuda"),
                           torch.as_tensor(positions + len(context) - 1, device="cuda"),
                           torch.tensor([answer[int(j)] for j in positions], device="cuda")))
            count += 1
            if count == samples:
                break
        if count != samples:
            raise ValueError(f"Could not select {samples} training examples for {domain}")
    return result


@torch.inference_mode()
def measure(model, docs, feature_file=None, router_file=None):
    domains = {d: [] for d in ("math", "code")}
    features, targets, boundaries = [], [], [0]
    hooks = []
    if router_file is not None:
        shape = (len(model.model.layers), model.config.n_routed_experts)
        mass = {key: torch.zeros(shape, dtype=torch.float32, device="cuda") for key in domains}
        lengths = dict.fromkeys(domains, 0)

        def recorder(layer_index):
            def record(_module, _inputs, output):
                _, weights, experts = output
                values = (weights.float() / model.config.route_scale).square().reshape(-1)
                mass[domain][layer_index].scatter_add_(0, experts.reshape(-1), values)
            return record

        hooks = [layer.mlp.gate.register_forward_hook(recorder(i)) for i, layer in enumerate(model.model.layers)]
    for domain, index, ids, positions, next_tokens in docs:
        if router_file is not None:
            lengths[domain] += ids.shape[-1] - 1
        hidden = model.model(input_ids=ids[:, :-1], use_cache=False).last_hidden_state
        selected = hidden.index_select(1, positions)
        scores = model.logit_softcap(model.lm_head(selected)).squeeze(0)
        nll = F.cross_entropy(scores.float(), next_tokens).item()
        if not np.isfinite(nll):
            raise ValueError(f"Non-finite training loss at {domain}/{index}")
        domains[domain].append({"id": index, "nll": nll})
        if feature_file is not None:
            features.append(selected.squeeze(0).float().cpu().numpy())
            targets.append(next_tokens.cpu().numpy())
            boundaries.append(boundaries[-1] + len(next_tokens))
    if feature_file is not None:
        np.savez(feature_file, hidden=np.concatenate(features), targets=np.concatenate(targets),
                 boundaries=np.array(boundaries), domains=np.array([r[0] for r in docs]),
                 case_ids=np.array([r[1] for r in docs]))
    if router_file is not None:
        for hook in hooks:
            hook.remove()
        np.savez(router_file, **{d: (mass[d] / lengths[d]).cpu().numpy() for d in domains},
                 token_counts=np.array([lengths[d] for d in domains]))
    return {domain: {"mean_nll": float(np.mean([x["nll"] for x in values])), "cases": values}
            for domain, values in domains.items()}


@torch.inference_mode()
def set_layer(model, ops, directory, luts, index):
    layer = model.model.layers[index]
    cfg = model.config
    metadata = json.loads((directory / f"layer_{index:02}_meta.json").read_text(encoding="utf-8"))
    # Drop all old device slabs before reading their replacements. No other
    # layer references these objects; the logical loop reuses this module.
    for key in metadata["linears"]:
        parent, name = key.rsplit(".", 1)
        delattr(layer.get_submodule(parent), name)
    del layer.mlp.experts
    torch.cuda.empty_cache()
    with np.load(io.BytesIO((directory / f"layer_{index:02}_packed.npz").read_bytes()), allow_pickle=False) as a:
        for key, info in metadata["linears"].items():
            parent, name = key.rsplit(".", 1)
            setattr(layer.get_submodule(parent), name, PackedLinear(ops, a, info, luts))
        layer.mlp.experts = PackedExperts(ops, a, luts, cfg.hidden_size, cfg.moe_intermediate_size,
                                         cfg.mlp_gate_softcap, cfg.mlp_up_softcap)
    prefix = f"model.layers.{index}."
    with safe_open(str(directory / "steering.safetensors"), framework="pt", device="cpu") as sf:
        steering = {k[len(prefix):]: sf.get_tensor(k).to(device=ops.device, dtype=torch.float32)
                    for k in sf.keys() if k.startswith(prefix)}
    layer.load_state_dict(steering, assign=True, strict=True)
    layer.eval()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("large", "small", "library", "data", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--variants", nargs="+", required=True, help="Comma-separated physical layer IDs per variant")
    p.add_argument("--samples", type=int, default=4, help="Labelled examples per training domain")
    p.add_argument("--seed", type=int, default=17291)
    p.add_argument("--feature-dir", type=Path, help="Store selected hidden vectors for cheap output-head comparisons")
    p.add_argument("--exclude-scan", type=Path, help="Exclude every document from another calibration scan")
    p.add_argument("--router-profile", type=Path, help="Save baseline training-domain expert gate mass")
    args = p.parse_args()
    if args.samples < 1 or args.out.exists():
        p.error("Use a positive sample count and a fresh results path")
    candidates = [{int(x) for x in v.split(",")} for v in args.variants]
    if any(not group or not group < set(range(24)) for group in candidates):
        p.error("Select nonempty proper subsets of physical layers 0 through 23")
    model, tok = load_model(args.large, args.library, progress=True)
    ops = CudaOps(args.library)
    with np.load(args.large / "luts.npz", allow_pickle=False) as a:
        luts = {k: ops.tensor(a[k], torch.float32) for k in a.files}
    if hashlib.sha256((args.large / "luts.npz").read_bytes()).digest() != hashlib.sha256((args.small / "luts.npz").read_bytes()).digest():
        raise ValueError("Layer mixing requires matching codebooks")
    excluded = set()
    if args.exclude_scan:
        previous = json.loads(args.exclude_scan.read_text(encoding="utf-8"))["results"][0]["metrics"]
        excluded = {(domain, row["id"]) for domain, side in previous.items() for row in side["cases"]}
    docs = examples(args.data, tok, args.samples, args.seed, excluded)
    report = {"seed": args.seed, "samples_per_domain": args.samples, "metric": "mean teacher-forced answer-token NLL",
              "excluded_previous_cases": len(excluded), "results": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.feature_dir is not None:
        args.feature_dir.mkdir(parents=True, exist_ok=False)
    current = set()
    for target in [set()] + candidates:
        for i in sorted(current ^ target):
            set_layer(model, ops, args.small if i in target else args.large, luts, i)
        current = target
        name = "layers-" + "-".join(str(i) for i in sorted(target)) if target else "large"
        feature_file = None if args.feature_dir is None else args.feature_dir / (name + ".npz")
        metrics = measure(model, docs, feature_file, args.router_profile if not target else None)
        entry = {"compact_layers": sorted(target), "metrics": metrics}
        report["results"].append(entry)
        write_json(args.out, report)
        print(sorted(target), {k: round(v["mean_nll"], 5) for k, v in metrics.items()}, flush=True)


if __name__ == "__main__":
    main()
