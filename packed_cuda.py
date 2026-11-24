"""Read LookAtMyQuant's packed Inkling matrices with a native CUDA backend."""

import ctypes
from functools import lru_cache
import io
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn


FORMATS = {"ht1": (1, 4), "ht2": (2, 4), "ht3": (3, 2),
           "rtr1": (1, 1), "rtr2": (2, 1), "rtr3": (3, 1), "rtr4": (4, 1),
           "lm6": (6, 1), "lm8": (8, 1)}
EXPERT_FORMATS = ("ht1", "ht2", "ht3", "rtr1", "rtr2", "rtr3", "rtr4")
PLAN = np.dtype([("codes", "<u8"), ("scales", "<u8"), ("lookup", "<u8"),
                 ("bits", "<i4"), ("group", "<i4"), ("shift", "<i4"), ("scalar", "<i4")])


class CudaOps:
    def __init__(self, library, device="cuda"):
        self.device = torch.device(device)
        self.library = ctypes.CDLL(str(Path(library).resolve()))
        ptr, integer = ctypes.c_void_p, ctypes.c_int
        self.library.packed_matvec.argtypes = [ptr] * 4 + [integer] * 4 + [ptr]
        self.library.packed_gather.argtypes = [ptr] * 4 + [integer] * 2 + [ptr]
        self.library.packed_matvec.restype = integer
        self.library.packed_gather.restype = integer

    def tensor(self, array, dtype=None):
        return torch.from_numpy(np.ascontiguousarray(array)).to(device=self.device, dtype=dtype)

    def plans(self, entries):
        values = np.array(entries, dtype=PLAN)
        assert values.dtype.itemsize == 40
        return self.tensor(values.view(np.uint8))

    def matvec(self, plans, x, rows, expert_ids=None, divisor=1):
        x = x.reshape(-1, x.shape[-1]).float().contiguous()
        if expert_ids is not None:
            expert_ids = expert_ids.reshape(-1).long().contiguous()
        slots = len(x) if expert_ids is None else len(expert_ids)
        if slots != len(x) * divisor:
            raise ValueError("Input vectors and expert slots disagree")
        out = torch.empty((slots, rows), dtype=torch.float32, device=x.device)
        status = self.library.packed_matvec(plans.data_ptr(), expert_ids.data_ptr() if expert_ids is not None else 0,
                                            x.data_ptr(), out.data_ptr(), slots, rows, x.shape[-1], divisor,
                                            torch.cuda.current_stream(self.device).cuda_stream)
        if status:
            raise RuntimeError(f"Packed CUDA matvec launch failed with error {status}")
        return out

    def gather(self, plans, token_map, ids, width):
        ids = ids.reshape(-1).long().contiguous()
        out = torch.empty((len(ids), width), dtype=torch.float32, device=self.device)
        status = self.library.packed_gather(plans.data_ptr(), token_map.data_ptr(), ids.data_ptr(),
                                           out.data_ptr(), len(ids), width,
                                           torch.cuda.current_stream(self.device).cuda_stream)
        if status:
            raise RuntimeError(f"Packed CUDA gather launch failed with error {status}")
        return out


def sylvester(size):
    h = np.ones((1, 1), dtype=np.float32)
    while len(h) < size:
        h = np.block([[h, h], [h, -h]])
    return h


@lru_cache(maxsize=16)
def rotation_factors(width, device):
    for power in range(width.bit_length() - 1, -1, -1):
        a, p = 1 << power, width // (1 << power)
        if a * p != width:
            continue
        if p == 1:
            size_a = 1 << (power // 2)
            left, right = sylvester(size_a), sylvester(width // size_a)
            break
        q = p - 1
        if q % 4 != 3 or any(q % j == 0 for j in range(2, math.isqrt(q) + 1)):
            continue
        squares = {j * j % q for j in range(1, q)}
        chi = np.array([0] + [1 if j in squares else -1 for j in range(1, q)], dtype=np.float32)
        right = np.eye(p, dtype=np.float32)
        right[0, 1:], right[1:, 0] = 1, -1
        right[1:, 1:] += chi[(np.arange(q)[None, :] - np.arange(q)[:, None]) % q]
        left = sylvester(a)
        break
    else:
        raise ValueError(f"No rotation for width {width}")
    return torch.as_tensor(left, device=device), torch.as_tensor(right, device=device)


def rotate(x, signs, inverse=False):
    width = x.shape[-1]
    a, b = rotation_factors(width, x.device)
    if inverse:
        blocks = x.float().reshape(-1, len(a), len(b))
        return ((a.T @ blocks @ b).reshape(x.shape) * signs) * (width ** -0.5)
    blocks = (x.float() * signs).reshape(-1, len(a), len(b))
    return (a @ blocks @ b.T).reshape(x.shape) * (width ** -0.5)


class Slab:
    def __init__(self, ops, codes, scales, lookup, fmt, rows, width):
        bits, group = FORMATS[fmt]
        self.rows = rows
        self.scales = ops.tensor(scales.reshape(-1), torch.float32)
        self.codes = ops.tensor(codes.reshape(-1).view(np.uint8))
        self.lookup = lookup
        self.expert_bytes = rows * width * bits // 8
        if self.codes.numel() % self.expert_bytes or self.scales.numel() != self.codes.numel() // (width * bits // 8):
            raise ValueError(f"Inconsistent {fmt} slab size")
        entries = lookup.numel() // group
        self.info = (bits, group, 32 - (entries.bit_length() - 1), int(fmt.startswith("lm")))

    def descriptor(self, index=0):
        if index < 0 or (index + 1) * self.expert_bytes > self.codes.numel():
            raise IndexError(f"Invalid packed expert {index}")
        return (self.codes.data_ptr() + index * self.expert_bytes,
                self.scales.data_ptr() + index * self.rows * self.scales.element_size(),
                self.lookup.data_ptr(), *self.info)


class PackedLinear(nn.Module):
    def __init__(self, ops, arrays, meta, luts):
        super().__init__()
        self.ops = ops
        base = "lin_" + meta["safe"]
        fmt = meta["fmt"]
        self.out_features = meta["m"]
        self.in_features = meta["n"]
        lut = ops.tensor(arrays[base + "_levels"], torch.float32) if fmt.startswith("lm") else luts[fmt]
        self.slab = Slab(ops, arrays[base + "_codes"], arrays[base + "_scales"], lut, fmt,
                         self.out_features, self.in_features)
        self.plans = ops.plans([self.slab.descriptor()])
        self.signs_in = ops.tensor(arrays[base + "_signs_in"], torch.float32)
        self.signs_out = ops.tensor(arrays[base + "_signs_out"], torch.float32) if meta.get("out_rot") else None

    def forward(self, x):
        y = self.ops.matvec(self.plans, rotate(x, self.signs_in), self.out_features)
        if self.signs_out is not None:
            y = rotate(y, self.signs_out, inverse=True)
        return y.reshape(*x.shape[:-1], self.out_features).to(x.dtype)


class ExpertSide:
    def __init__(self, ops, arrays, luts, side, rows, width):
        self.ops, self.rows = ops, rows
        self.signs = ops.tensor(arrays[f"{side}_signs_in"], torch.float32)
        expert_map = arrays[f"{side}_map"]
        self.slabs = {}
        for fid, fmt in enumerate(EXPERT_FORMATS):
            key = f"{side}_{fmt}"
            if key + "_codes" in arrays.files:
                self.slabs[fid] = Slab(ops, arrays[key + "_codes"], arrays[key + "_scales"], luts[fmt],
                                       fmt, rows, width)
        self.plans = ops.plans([self.slabs[int(fid)].descriptor(int(index)) for fid, index in expert_map])

    def __call__(self, x, ids, divisor):
        return self.ops.matvec(self.plans, rotate(x, self.signs), self.rows, ids, divisor)


class PackedExperts(nn.Module):
    def __init__(self, ops, arrays, luts, hidden, intermediate, gate_cap, up_cap):
        super().__init__()
        self.gu = ExpertSide(ops, arrays, luts, "gu", 2 * intermediate, hidden)
        self.dn = ExpertSide(ops, arrays, luts, "dn", hidden, intermediate)
        self.gate_cap, self.up_cap = gate_cap, up_cap

    def forward(self, hidden_states, top_k_index, top_k_weights):
        gate, up = self.gu(hidden_states, top_k_index, top_k_index.shape[-1]).chunk(2, dim=-1)
        act = ((gate / self.gate_cap).tanh() * self.gate_cap) * gate.sigmoid()
        act = act * ((up / self.up_cap).tanh() * self.up_cap)
        result = self.dn(act, top_k_index, 1).reshape(*top_k_index.shape, -1)
        return (result * top_k_weights.unsqueeze(-1)).sum(dim=1).to(hidden_states.dtype)


class PackedTable(nn.Module):
    def __init__(self, ops, directory, name, luts, width, embed_norm=None):
        super().__init__()
        self.ops, self.width, self.embed_norm = ops, width, embed_norm
        self.slabs, self.signs, self.rows, self.bucket_plans = [], [], [], []
        metadata = json.loads((directory / f"{name}_meta.json").read_text(encoding="utf-8"))
        with np.load(io.BytesIO((directory / f"{name}_packed.npz").read_bytes()), allow_pickle=False) as archive:
            order = archive["order"].astype(np.int64)
            if not np.array_equal(np.sort(order), np.arange(len(order))):
                raise ValueError(f"Invalid token order in {name}")
            token_map = np.empty((len(order), 2), dtype=np.int32)
            start = 0
            for index, meta in enumerate(metadata):
                base, fmt, rows = f"b{meta['kb']}", meta["opt"], meta["rows"]
                lookup = ops.tensor(archive[base + "_levels"], torch.float32) if fmt.startswith("lm") else luts[fmt]
                slab = Slab(ops, archive[base + "_codes"], archive[base + "_scales"], lookup, fmt, rows, width)
                self.slabs.append(slab)
                self.bucket_plans.append(ops.plans([slab.descriptor()]))
                self.signs.append(ops.tensor(archive[base + "_signs_in"], torch.float32).reshape(width))
                self.rows.append(rows)
                token_map[order[start:start + rows], 0] = index
                token_map[order[start:start + rows], 1] = np.arange(rows)
                start += rows
        if start != len(order):
            raise ValueError(f"Incomplete token table in {name}")
        self.plans = ops.plans([slab.descriptor() for slab in self.slabs])
        self.token_map = ops.tensor(token_map)
        self.signs = torch.stack(self.signs)
        self.inverse = ops.tensor(np.argsort(order), torch.long)

    def forward(self, x):
        if self.embed_norm is not None:
            ids = x.reshape(-1)
            raw = self.ops.gather(self.plans, self.token_map, ids, self.width)
            signs = self.signs[self.token_map[ids, 0].long()]
            return self.embed_norm(rotate(raw, signs, inverse=True).reshape(*x.shape, self.width))
        logits = [self.ops.matvec(plan, rotate(x, self.signs[i]), self.rows[i])
                  for i, plan in enumerate(self.bucket_plans)]
        return torch.cat(logits, dim=-1).index_select(-1, self.inverse).reshape(*x.shape[:-1], -1)
