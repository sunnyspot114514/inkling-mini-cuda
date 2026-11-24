"""Verify packed layers against pinned Hub objects and mixed steering against both parents."""

import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen

from build_candidate import SOURCES, STATIC_FILES, read_header


def inventory(repo, revision):
    request = Request(f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true",
                      headers={"User-Agent": "inkling-artifact-verification"})
    with urlopen(request, timeout=40) as response:
        data = json.load(response)
    if data["sha"] != revision:
        raise ValueError(f"Hub returned a different revision of {repo}")
    return {x["rfilename"]: x for x in data["siblings"]}


def hash_match(path, record):
    size = path.stat().st_size
    if size != record["size"]:
        raise ValueError(f"Unexpected size for {path}")
    sha = hashlib.sha256()
    git_blob = hashlib.sha1(f"blob {size}\0".encode())
    expected_lfs = (record.get("lfs") or {}).get("sha256")
    with path.open("rb") as stream:
        while block := stream.read(4 << 20):
            sha.update(block)
            if not expected_lfs:
                git_blob.update(block)
    if expected_lfs:
        if sha.hexdigest() != expected_lfs:
            raise ValueError(f"SHA-256 mismatch for {path}")
    elif git_blob.hexdigest() != record["blobId"]:
        raise ValueError(f"Git blob mismatch for {path}")
    return sha.hexdigest()


def check_steering(candidate, large, small, compact):
    with candidate.open("rb") as c, large.open("rb") as l, small.open("rb") as s:
        current, start = read_header(c)
        parents = {False: (l, *read_header(l)), True: (s, *read_header(s))}
        checked = {"large": 0, "small": 0}
        if any(keys.keys() != current.keys() for _, keys, _ in parents.values()):
            raise ValueError("Steering inventories differ")
        for key, info in current.items():
            layer = re.match(r"^model\.layers\.(\d+)\.", key)
            use_small = bool(layer and int(layer[1]) in compact)
            source, keys, source_start = parents[use_small]
            ref = keys[key]
            lo, hi = info["data_offsets"]
            rlo, rhi = ref["data_offsets"]
            if info["dtype"] != ref["dtype"] or info["shape"] != ref["shape"] or hi-lo != rhi-rlo:
                raise ValueError(f"Incompatible steering field {key}")
            source.seek(source_start + rlo)
            c.seek(start + lo)
            if c.read(hi - lo) != source.read(rhi - rlo):
                raise ValueError(f"Steering mismatch for {key}")
            checked["small" if use_small else "large"] += 1
    return checked


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for label in ("candidate", "large", "small", "report"):
        p.add_argument("--" + label, type=Path, required=True)
    args = p.parse_args()
    recipe = json.loads((args.candidate / "recipe.json").read_text(encoding="utf-8"))
    compact = set(recipe["compact_layers"])
    for key, (repo, rev) in SOURCES.items():
        if recipe["source_revisions"][key] != {"model": repo, "revision": rev}:
            raise ValueError("The recipe uses an unsupported upstream revision")
    refs = {name: inventory(*target) for name, target in SOURCES.items()}
    from_parent = dict.fromkeys(STATIC_FILES, "large")
    for i in range(24):
        for suffix in ("meta.json", "packed.npz"):
            from_parent[f"layer_{i:02}_{suffix}"] = "small" if i in compact else "large"
    hashes = {}
    for name, parent in from_parent.items():
        hashes[name] = hash_match(args.candidate / name, refs[parent][name])
    steering = "steering.safetensors"
    for parent in ("large", "small"):
        hash_match(getattr(args, parent) / steering, refs[parent][steering])
    steering_counts = check_steering(args.candidate / steering, args.large / steering,
                                     args.small / steering, compact)
    extras = []
    if "output_correction" in recipe:
        meta = json.loads((args.candidate / "head_residual.json").read_text(encoding="utf-8"))
        if meta != recipe["output_correction"] or meta["format"] != "inkling-output-residual-v1":
            raise ValueError("Correction settings differ from the recipe")
        extras = ["head_residual.json", "head_residual.safetensors"]
    for name in ("manifest.json", steering, *extras):
        with (args.candidate / name).open("rb") as f:
            hashes[name] = hashlib.file_digest(f, "sha256").hexdigest()
    if extras and hashes["head_residual.safetensors"] != meta["factor_sha256"]:
        raise ValueError("Correction factor checksum mismatch")
    total = sum((args.candidate / name).stat().st_size for name in hashes)
    if total != recipe["runtime_file_bytes"]:
        raise ValueError("The declared size differs from verified runtime inputs")
    report = {"runtime_file_bytes": total, "steering_tensors": steering_counts, "sha256": hashes}
    args.report.parent.mkdir(exist_ok=True, parents=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(hashes)} files, {sum(steering_counts.values())} steering tensors, {total} bytes.")


if __name__ == "__main__":
    main()
