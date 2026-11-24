"""Optional low-rank correction applied to the packed output head before softcapping."""

import json
import math
from pathlib import Path

from safetensors.torch import load_file
import torch
from torch import nn


def read_factors(path, device, vocab, width):
    tensors = load_file(str(path), device="cpu")
    down, up = tensors["down"], tensors["up"]
    if set(tensors) != {"up", "down"} or down.ndim != 2 or up.shape != (vocab, len(down)) or down.shape[1] != width:
        raise ValueError("Output correction dimensions do not match the checkpoint")
    return down.to(device=device, dtype=torch.float32), up.to(device=device, dtype=torch.float32)


def logit_delta(x, down, up):
    return (x.float() @ down.T) @ up.T


class CorrectedOutput(nn.Module):
    def __init__(self, packed, down, up, strength):
        super().__init__()
        self.packed = packed
        self.register_buffer("down", down, persistent=False)
        self.register_buffer("up", up, persistent=False)
        self.strength = strength

    def forward(self, hidden_states):
        return self.packed(hidden_states) + self.strength * logit_delta(hidden_states, self.down, self.up)


def attach_if_present(model, directory):
    directory = Path(directory)
    meta, weights = directory / "head_residual.json", directory / "head_residual.safetensors"
    if not meta.exists() and not weights.exists():
        return
    info = json.loads(meta.read_text(encoding="utf-8"))
    strength = float(info["strength"])
    if info["format"] != "inkling-output-residual-v1" or not math.isfinite(strength):
        raise ValueError("Invalid output correction metadata")
    down, up = read_factors(weights, "cuda", model.config.unpadded_vocab_size, model.config.hidden_size)
    if info["rank"] != down.shape[0]:
        raise ValueError("Output correction rank disagrees with the metadata")
    model.lm_head = CorrectedOutput(model.lm_head, down, up, strength)
