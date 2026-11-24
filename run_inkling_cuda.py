"""Run a local text-only packed Inkling checkpoint on a CUDA GPU."""

import argparse
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
from safetensors.torch import load_file
import torch
from transformers import PreTrainedTokenizerFast
from transformers.cache_utils import DynamicCache
from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig
from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopForCausalLM

from packed_cuda import CudaOps, PackedExperts, PackedLinear, PackedTable
from head_residual import attach_if_present


class HalfKVCache(DynamicCache):
    """Store key/value histories in FP16 and keep the decoder's attention input dtype."""

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        keys, values = super().update(key_states.half(), value_states.half(), layer_idx, *args, **kwargs)
        return keys.to(key_states.dtype), values.to(value_states.dtype)


@torch.inference_mode()
def load_model(path, library, progress=False):
    path = Path(path)
    ops = CudaOps(library)
    raw = json.loads((path / "config.json").read_text(encoding="utf-8"))
    cfg = Inkling2LoopTextConfig.from_dict(raw["text_config"])
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "eager"
    with torch.device("meta"):
        model = Inkling2LoopForCausalLM(cfg)
    with np.load(path / "luts.npz", allow_pickle=False) as a:
        luts = {k: ops.tensor(a[k], torch.float32) for k in a.files}
    embed_norm = model.model.embed_tokens.embed_norm
    model.model.embed_tokens = PackedTable(ops, path, "embed", luts, cfg.hidden_size, embed_norm)
    model.lm_head = PackedTable(ops, path, "head", luts, cfg.hidden_size)
    for i, layer in enumerate(model.model.layers):
        info = json.loads((path / f"layer_{i:02}_meta.json").read_text(encoding="utf-8"))
        # Read each layer sequentially so an external hard drive does not seek
        # between the individual matrix and scale members of the zip archive.
        layer_file = path / f"layer_{i:02}_packed.npz"
        with np.load(io.BytesIO(layer_file.read_bytes()), allow_pickle=False) as a:
            for key, desc in info["linears"].items():
                parent, name = key.rsplit(".", 1)
                old_layer = layer.get_submodule(key)
                if (old_layer.out_features, old_layer.in_features) != (desc["m"], desc["n"]):
                    raise ValueError(f"Incompatible matrix layer {i}, {key}")
                setattr(layer.get_submodule(parent), name, PackedLinear(ops, a, desc, luts))
            layer.mlp.experts = PackedExperts(ops, a, luts, cfg.hidden_size, cfg.moe_intermediate_size,
                                             cfg.mlp_gate_softcap, cfg.mlp_up_softcap)
        if progress and (i + 1) % 6 == 0:
            print(f"Loaded {i+1}/{len(model.model.layers)} layers", file=sys.stderr, flush=True)
    steering = {name: value.to(device=ops.device, dtype=torch.float32) for name, value in
                load_file(str(path / "steering.safetensors"), device="cpu").items()}
    model.load_state_dict(steering, strict=True, assign=True)
    if any(t.is_meta for t in (*model.parameters(), *model.buffers())):
        raise RuntimeError("Unmaterialized parameters remained after loading")
    attach_if_present(model, path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(path, local_files_only=True)
    torch.cuda.empty_cache()
    return model.eval(), tokenizer


@torch.inference_mode()
def generate_prompts(model, tokenizer, prompts, limit, raw=False, kv_dtype="model"):
    if not prompts or limit < 1:
        raise ValueError("Provide at least one prompt and a positive output limit")
    token_lists = []
    for prompt in prompts:
        if raw:
            tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        else:
            tokens = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                   add_generation_prompt=True, reasoning_effort="none",
                                                   tokenize=True, return_dict=True)["input_ids"]
        token_lists.append(tokens)
    lengths = [len(tokens) for tokens in token_lists]
    width = max(lengths)
    padded = [[tokenizer.pad_token_id] * (width - len(tokens)) + tokens for tokens in token_lists]
    ids = torch.tensor(padded, dtype=torch.long, device="cuda")
    mask = None
    if min(lengths) != width:
        mask = torch.tensor([[0] * (width - n) + [1] * n for n in lengths], device="cuda")
    start = time.perf_counter()
    if kv_dtype not in ("model", "float16"):
        raise ValueError(f"Unsupported KV cache dtype {kv_dtype}")
    cache = HalfKVCache(config=model.config) if kv_dtype == "float16" else None
    results = [[] for _ in prompts]
    active = list(range(len(prompts)))
    finished_at = [None] * len(prompts)
    first_token = None
    chunk_size = max(1, 64 // len(prompts))
    for offset in range(0, width, chunk_size):
        end = min(width, offset + chunk_size)
        out = model(input_ids=ids[:, offset:end], attention_mask=None if mask is None else mask[:, :end],
                    cache=cache, use_cache=True, logits_to_keep=1)
        cache = out.cache
    for step in range(limit):
        current = out.logits[:, -1].argmax(-1, keepdim=True)
        next_tokens = current.flatten().tolist()
        if first_token is None:
            first_token = time.perf_counter()
        survivors = []
        for slot, token in enumerate(next_tokens):
            i = active[slot]
            if token == model.config.eos_token_id:
                finished_at[i] = time.perf_counter() - start
            else:
                results[i].append(token)
                survivors.append(slot)
        if not survivors or step + 1 == limit:
            break
        if len(survivors) < len(active):
            indices = torch.tensor(survivors, device=current.device, dtype=torch.long)
            cache.reorder_cache(indices)
            current = current.index_select(0, indices)
            if mask is not None:
                mask = mask.index_select(0, indices)
            active = [active[j] for j in survivors]
        if mask is not None:
            mask = torch.cat((mask, torch.ones_like(mask[:, :1])), dim=-1)
        out = model(input_ids=current, attention_mask=mask, cache=cache, use_cache=True, logits_to_keep=1)
        cache = out.cache
    elapsed = time.perf_counter() - start
    return [{
        "completion": tokenizer.decode(tokens, skip_special_tokens=False),
        "output_ids": tokens,
        "input_tokens": n, "output_tokens": len(tokens), "reached_limit": len(tokens) == limit,
        "batch_size": len(prompts), "batch_elapsed_seconds": elapsed,
        "completion_seconds": ended if ended is not None else elapsed,
        "first_token_seconds": first_token - start,
        "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
    } for tokens, n, ended in zip(results, lengths, finished_at)]


def generate(model, tokenizer, prompt, limit, raw=False, kv_dtype="model"):
    return generate_prompts(model, tokenizer, [prompt], limit, raw, kv_dtype)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--kv-dtype", choices=["model", "float16"], default="model")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    lm, tok = load_model(args.model_dir, args.library, progress=True)
    print(json.dumps(generate(lm, tok, args.prompt, args.max_new_tokens, args.raw, args.kv_dtype),
                     ensure_ascii=False, indent=2))
