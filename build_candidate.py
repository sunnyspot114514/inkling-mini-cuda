"""Build a self-contained text checkpoint from two independently tuned Inkling packs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import tempfile


MODEL_ROOT = "shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-"
SOURCES = {
    "small": (MODEL_ROOT + "5.4GB", "fb17b5be547b11e76a9f8f26239b4b4f59286fec"),
    "large": (MODEL_ROOT + "8.9GB", "ab695af99964ea87d0e609ad8060374f05306ae7"),
}
STATIC_FILES = (
    "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "generation_config.json", "chat_template.jinja", "embed_meta.json", "embed_packed.npz",
    "head_meta.json", "head_packed.npz", "luts.npz",
)


def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_header(f):
    length = struct.unpack("<Q", f.read(8))[0]
    if length > 64 * 1024 * 1024:
        raise ValueError("Invalid safetensors header size")
    items = json.loads(f.read(length))
    items.pop("__metadata__", None)
    return items, 8 + length


def splice_steering(large, small, output, compact_layers):
    shutil.copyfile(large, output)
    copied = 0
    with output.open("r+b") as dst, small.open("rb") as src:
        original, offset_dst = read_header(dst)
        donor, offset_src = read_header(src)
        if original.keys() != donor.keys():
            raise ValueError("Different steering tensor inventories")
        for key, info in original.items():
            match = re.match(r"^model\.layers\.(\d+)\.", key)
            if not match or int(match[1]) not in compact_layers:
                continue
            source = donor[key]
            start, end = info["data_offsets"]
            lo, hi = source["data_offsets"]
            if (info["dtype"], info["shape"], end - start) != (source["dtype"], source["shape"], hi - lo):
                raise ValueError(f"Incompatible steering tensor {key}")
            src.seek(offset_src + lo)
            data = src.read(hi - lo)
            if len(data) != hi - lo:
                raise EOFError(key)
            dst.seek(offset_dst + start)
            dst.write(data)
            copied += 1
    if not copied:
        raise ValueError("No layer-local steering tensors selected")
    return copied


def build(large, small, output, compact_layers):
    if not compact_layers or not compact_layers < set(range(24)):
        raise ValueError("Select a nonempty proper subset of layers 0 through 23")
    if output.exists():
        raise FileExistsError(output)
    for name in ("config.json", "luts.npz", "tokenizer.json", "chat_template.jinja"):
        if sha256(small / name) != sha256(large / name):
            raise ValueError(f"Base packs disagree on {name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".inkling-build-", dir=output.parent) as staging:
        candidate = Path(staging) / "model"
        candidate.mkdir()
        for name in STATIC_FILES:
            shutil.copyfile(large / name, candidate / name)
        for i in range(24):
            source = small if i in compact_layers else large
            for suffix in ("meta.json", "packed.npz"):
                name = f"layer_{i:02}_{suffix}"
                shutil.copyfile(source / name, candidate / name)
        steering = "steering.safetensors"
        mixed = splice_steering(large / steering, small / steering, candidate / steering, compact_layers)
        manifest = json.loads((large / "manifest.json").read_text(encoding="utf-8"))
        manifest.pop("plan_total_gb", None)
        manifest["packed_text_gb"] = round(sum(p.stat().st_size for p in candidate.iterdir()) / 1e9, 5)
        (candidate / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        recipe = {
            "method": "Layer-wise combination of calibrated, fine-tuned quantized checkpoints",
            "source_revisions": {k: dict(zip(("model", "revision"), v)) for k, v in SOURCES.items()},
            "compact_layers": sorted(compact_layers),
            "embedding_head_and_global_steering": "large",
            "compact_layer_steering_tensors": mixed,
            "runtime_file_bytes": sum(p.stat().st_size for p in candidate.iterdir()),
            "quality_evaluation": "pending",
        }
        if recipe["runtime_file_bytes"] >= 8_870_000_000:
            raise ValueError("Candidate exceeds the experiment's 8.87 GB ceiling")
        (candidate / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
        os.rename(candidate, output)
    return recipe


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--large", type=Path, required=True)
    parser.add_argument("--small", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compact-layers", type=int, nargs="+", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.large, args.small, args.output, set(args.compact_layers)), indent=2))
