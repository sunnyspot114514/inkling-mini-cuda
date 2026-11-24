"""Fetch the fixed public benchmark revisions and verify their published input bytes."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen


GSM = "https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/"
HE = "https://raw.githubusercontent.com/openai/human-eval/b9453e8f1865df0a0057a1383767caa400acfbcf/data/"
MBPP = "https://huggingface.co/datasets/google-research-datasets/mbpp/resolve/4bb6404fdc6cacfda99d4ac4205087b89d32030c/"
INPUTS = [
    ("gsm8k-test.jsonl", GSM + "test.jsonl", "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"),
    ("HumanEval.jsonl.gz", HE + "HumanEval.jsonl.gz", "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef"),
]
CALIBRATION = [
    ("gsm8k-train.jsonl", GSM + "train.jsonl", "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465"),
    ("mbpp-train.parquet", MBPP + "full/train-00000-of-00001.parquet", "09d125ca31edacb7800be8c67c45abff618faf0214ff551291817d06bdb914ae"),
]


def fetch(path, url, digest):
    if path.exists():
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() == digest:
                return
        raise ValueError(f"Checksum mismatch for existing {path}")
    with urlopen(url, timeout=60) as source, tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as dst:
        tmp = Path(dst.name)
        try:
            checksum = hashlib.sha256()
            while chunk := source.read(1 << 20):
                dst.write(chunk)
                checksum.update(chunk)
            if checksum.hexdigest() != digest:
                raise ValueError(f"Checksum mismatch for {url}")
        except BaseException:
            dst.close()
            tmp.unlink(missing_ok=True)
            raise
    os.replace(tmp, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--calibration", action="store_true")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    for name, url, digest in INPUTS + (CALIBRATION if args.calibration else []):
        fetch(args.directory / name, url, digest)
    if args.calibration:
        import pyarrow.parquet as parquet
        data = parquet.read_table(args.directory / "mbpp-train.parquet")
        tmp = args.directory / "mbpp-train.jsonl.tmp"
        with tmp.open("w", encoding="utf-8", newline="\n") as out:
            for item in data.to_pylist():
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
        os.replace(tmp, args.directory / "mbpp-train.jsonl")
