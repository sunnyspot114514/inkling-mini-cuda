"""Allocate compact expert pairs using training-domain routing, sampled error, and actual byte savings."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
from threadpoolctl import threadpool_limits

from build_candidate import SOURCES, STATIC_FILES, sha256
from verify_candidate import inventory


FORMATS = ("ht1", "ht2", "ht3", "rtr1", "rtr2", "rtr3", "rtr4")
BITS = np.array((1, 2, 3, 1, 2, 3, 4), dtype=np.int64)
GROUP = (4, 4, 2, 1, 1, 1, 1)
N_EXPERTS = 128
ROWS = 2304
WIDTH = {"gu": 2304, "dn": 1152}
SLAB = re.compile(r"^(?:gu|dn)_(?:map|(?:ht|rtr)\d+_(?:codes|scales))$")


def archive(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Pinned input digest mismatch for {path}")
    return np.load(io.BytesIO(data), allow_pickle=False)


def vector_rows(codes, scales, lut, width, bits, group):
    padded = np.pad(codes, ((0, 0), (0, 0), (0, 1)))
    bit = np.arange(width-group, -1, -group, dtype=np.int64).astype(np.uint32) * np.uint32(bits)
    word, shift = bit >> 5, bit & 31
    index = ((padded[..., word] >> shift) | np.where(shift == 0, 0, padded[..., word+1] << (32-shift))) & 4095
    if group > 1:
        index = ((index.astype(np.uint64) * 2654435761) & 0xffffffff) >> (32-(len(lut).bit_length()-1))
    return lut[index].reshape(*codes.shape[:2], width).astype(np.float32) * scales[..., None]


def sampled_side(z, side, output_rows, luts):
    mapping = z[side+"_map"].astype(np.int64)
    if mapping.shape != (N_EXPERTS, 2):
        raise ValueError("Unrecognized expert map")
    out = np.empty((N_EXPERTS, len(output_rows), WIDTH[side]), dtype=np.float32)
    for fid in np.unique(mapping[:, 0]):
        name, g, k = FORMATS[fid], GROUP[fid], BITS[fid]
        ids = np.flatnonzero(mapping[:, 0] == fid)
        local = mapping[ids, 1]
        if not np.array_equal(np.sort(local), np.arange(len(ids))):
            raise ValueError("Expert map is not a permutation of its slab")
        stem = f"{side}_{name}_"
        packed = z[stem + "codes"][:, output_rows, :]
        scales = z[stem + "scales"][:, output_rows].astype(np.float32)
        values = vector_rows(packed, scales, luts[name].reshape(-1, g), WIDTH[side], k, g)
        out[ids] = values[local]
    cost = ROWS * WIDTH[side] * BITS[mapping[:, 0]] // 8
    return out, cost


def sample_layer(path, digest, row_ids, luts):
    with archive(path, digest) as z:
        signs = {side: z[side+"_signs_in"] for side in WIDTH}
        samples = {side: sampled_side(z, side, row_ids[side], luts) for side in WIDTH}
    return samples, signs


def saved_bytes_plan(large, small, sources, mass, original_size, budget, seed, nrows):
    with np.load(large / "luts.npz") as z:
        luts = {k: z[k] for k in z.files}
    ranked = []
    for i in range(24):
        rng = np.random.default_rng(seed+i)
        half = ROWS//2
        row_ids = {"gu": np.r_[rng.choice(half, nrows//2, replace=False),
                               half+rng.choice(half, nrows-nrows//2, replace=False)],
                   "dn": rng.choice(ROWS, nrows, replace=False)}
        name = f"layer_{i:02}_packed.npz"
        q, q_signs = sample_layer(large/name, sources["large"][name]["lfs"]["sha256"], row_ids, luts)
        s, s_signs = sample_layer(small/name, sources["small"][name]["lfs"]["sha256"], row_ids, luts)
        risk = np.zeros(N_EXPERTS, np.float64)
        savings = np.zeros(N_EXPERTS, np.int64)
        for side in WIDTH:
            if not np.array_equal(q_signs[side], s_signs[side]):
                raise ValueError(f"Incompatible rotations in layer {i} {side}")
            a, ca = q[side]
            b, cb = s[side]
            squared = np.sum((a-b)**2, axis=(1,2), dtype=np.float64)
            power = np.sum(a*a, axis=(1,2), dtype=np.float64)
            risk += squared / np.maximum(power, 1e-20)
            savings += ca-cb
        per_domain = mass[:, i, :]
        smooth = per_domain.max(axis=0)
        smoothed = .98*smooth + .02*smooth.mean()
        risk = .5*risk*smoothed
        protected = set(np.argsort(-per_domain, axis=1, kind="stable")[:, :16].flatten().tolist())
        for e in range(N_EXPERTS):
            if savings[e] > 0 and e not in protected:
                ranked.append((float(risk[e] / savings[e]), i, e, int(savings[e])))
        print(f"Profiled expert pairs in layer {i+1}/24", flush=True)
    chosen = [set() for _ in range(24)]
    size = original_size
    for _, i, e, savings in sorted(ranked):
        if size <= budget-131072:
            break
        chosen[i].add(e)
        size -= savings
    if size > budget:
        raise ValueError("The requested size cannot be reached with the protected experts")
    return chosen, size


def array_digest(a):
    return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast("B")).hexdigest()


def mix_layer(high, low, compact, target):
    arrays = {k: high[k] for k in high.files if not SLAB.match(k)}
    formats = {}
    use_low = np.array([i in compact for i in range(N_EXPERTS)])
    for side in WIDTH:
        hi_map, lo_map = high[side+"_map"], low[side+"_map"]
        fid_per_expert = np.where(use_low, lo_map[:,0], hi_map[:,0]).astype(np.int64)
        mapping = np.empty_like(hi_map)
        formats[side] = {}
        for fid in np.unique(fid_per_expert):
            fmt = FORMATS[fid]
            ids = np.flatnonzero(fid_per_expert == fid)
            mapping[ids,0], mapping[ids,1] = fid, np.arange(len(ids))
            formats[side][fmt] = len(ids)
            for kind in ("codes","scales"):
                key = f"{side}_{fmt}_{kind}"
                output = None
                for take_small, source, source_map in ((False, high, hi_map), (True, low, lo_map)):
                    dests = np.flatnonzero(use_low[ids] == take_small)
                    if not len(dests):
                        continue
                    slab = source[key]
                    if output is None:
                        output = np.empty((len(ids), *slab.shape[1:]), dtype=slab.dtype)
                    if output.dtype != slab.dtype or output.shape[1:] != slab.shape[1:]:
                        raise ValueError(f"Slab layout mismatch for {key}")
                    for d in dests:
                        local = int(source_map[ids[d], 1])
                        output[d] = slab[local]
                        if not np.array_equal(output[d], slab[local]):
                            raise ValueError("Expert matrix copy failed")
                    del slab
                arrays[key] = output
        arrays[side+"_map"] = mapping
    expected = {key: array_digest(value) for key,value in arrays.items()}
    np.savez(target, **arrays)
    del arrays
    with np.load(target, allow_pickle=False) as verified:
        if set(verified.files) != set(expected):
            raise ValueError("Written layer has a different array inventory")
        for key, digest in expected.items():
            if array_digest(verified[key]) != digest:
                raise ValueError(f"Written array failed its checksum: {key}")
    return formats


def build(args):
    if args.out.exists() or args.report.exists():
        raise FileExistsError("Use a fresh candidate and report path")
    refs = {name: inventory(repo,rev) for name,(repo,rev) in SOURCES.items()}
    for name in ("config.json","tokenizer.json","luts.npz"):
        if sha256(args.large/name) != sha256(args.small/name):
            raise ValueError(f"Unmatched parent file {name}")
    with np.load(args.router_profile, allow_pickle=False) as p:
        mass = np.stack([p["math"],p["code"]]).astype(np.float64)
    if mass.shape != (2,24,N_EXPERTS) or not np.isfinite(mass).all() or (mass < 0).any():
        raise ValueError("Invalid routing profile")
    metadata = [f"layer_{i:02}_{suffix}" for i in range(24) for suffix in ("meta.json","packed.npz")]
    fixed = [*STATIC_FILES, "steering.safetensors"]
    source_size = sum((args.large/k).stat().st_size for k in [*fixed,*metadata,"manifest.json"])
    compact, estimate = saved_bytes_plan(args.large,args.small,refs,mass,source_size,args.budget,args.seed,args.rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent, prefix=".expert-build-") as tmp:
        dest = Path(tmp)/"model"
        dest.mkdir()
        for name in fixed:
            shutil.copyfile(args.large/name,dest/name)
        for i, selected in enumerate(compact):
            name, meta = f"layer_{i:02}_packed.npz", f"layer_{i:02}_meta.json"
            if selected:
                with archive(args.large/name,refs["large"][name]["lfs"]["sha256"]) as high, \
                     archive(args.small/name,refs["small"][name]["lfs"]["sha256"]) as low:
                    desc = json.loads((args.large/meta).read_text(encoding="utf-8"))
                    desc["experts"] = mix_layer(high,low,selected,dest/name)
                (dest/meta).write_text(json.dumps(desc,indent=2)+"\n",encoding="utf-8")
            else:
                shutil.copyfile(args.large/name,dest/name)
                shutil.copyfile(args.large/meta,dest/meta)
                if sha256(dest/name) != refs["large"][name]["lfs"]["sha256"]:
                    raise ValueError(f"Unchanged layer checksum mismatch: {name}")
            print(f"Wrote and checked layer {i+1}/24, {len(selected)} compact experts",flush=True)
        manifest = json.loads((args.large/"manifest.json").read_text(encoding="utf-8"))
        manifest.pop("plan_total_gb",None)
        manifest["packed_text_gb"] = round(sum(p.stat().st_size for p in dest.iterdir())/1e9,5)
        (dest/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
        hashes = {p.name:sha256(p) for p in dest.iterdir()}
        size = sum(p.stat().st_size for p in dest.iterdir())
        if size > args.budget:
            raise ValueError(f"Candidate is over its byte budget: {size}")
        recipe = {"method":"Routing-weighted, sampled-error expert selection from two tuned quantized parents",
                  "source_revisions":{k:{"model":v[0],"revision":v[1]} for k,v in SOURCES.items()},
                  "runtime_file_bytes":size,"estimated_bytes":estimate,"size_budget":args.budget,
                  "router_sha256":sha256(args.router_profile),"sampling_seed":args.seed,"sampled_rows":args.rows,
                  "compact_experts_per_layer":[sorted(s) for s in compact],"expert_pairs_retained":24*N_EXPERTS,
                  "compact_expert_pairs":sum(map(len,compact)),"quality_evaluation":"pending"}
        (dest/"recipe.json").write_text(json.dumps(recipe,indent=2)+"\n",encoding="utf-8")
        os.rename(dest,args.out)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps({"model":args.out.name,"runtime_file_bytes":size,"sha256":hashes,
                                      "compact_expert_pairs":recipe["compact_expert_pairs"]},indent=2)+"\n",encoding="utf-8")
    print(f"Candidate bytes {size}, compact expert pairs {recipe['compact_expert_pairs']}/3072",flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("large","small","router-profile","out","report"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--budget",type=int,default=8_350_000_000)
    p.add_argument("--seed",type=int,default=65937)
    p.add_argument("--rows",type=int,default=8)
    a=p.parse_args()
    if a.rows < 2 or a.rows > ROWS:
        p.error("Sample between two and 2304 projection rows")
    with threadpool_limits(2):
        build(a)
