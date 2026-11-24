"""Run both frozen checkpoints in fixed windows on one GPU and resume completed items."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from compare_runs import compare
from evaluate import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "library", "data", "out"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--first-window", type=int, default=8)
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--kv-dtype", choices=["model", "float16"], default="float16")
    p.add_argument("--bwrap", default="/usr/bin/bwrap")
    p.add_argument("--wsl-distro", default="Ubuntu-22.04")
    args = p.parse_args()
    if min(args.first_window, args.window, args.batch_size, args.max_new_tokens) < 1:
        p.error("Window sizes, batch size, and output limit must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    start = 0
    while start < 200:
        end = min(200, start + (args.first_window if start == 0 else args.window))
        for name, directory in (("candidate", args.candidate), ("baseline", args.baseline)):
            dest = args.out / name
            available = all((dest / split / f"{ix:04}.json").is_file()
                            for split,n in (("gsm8k",200),("humaneval",164))
                            for ix in range(start, min(end,n)))
            if available and start > 0:
                continue
            cmd = [sys.executable, "-u", str(Path(__file__).with_name("evaluate.py")), "--model", str(directory),
                   "--library", str(args.library), "--data", str(args.data), "--out", str(dest),
                   "--split", "gsm8k", "humaneval", "--start-index", str(start), "--stop-index", str(end),
                   "--batch-size", str(args.batch_size), "--max-new-tokens", str(args.max_new_tokens),
                   "--kv-dtype", args.kv_dtype, "--bwrap", args.bwrap, "--wsl-distro", args.wsl_distro]
            print(f"{name}: test window [{start}, {end})", flush=True)
            subprocess.run(cmd, check=True)
        start = end
    results = {split: compare(args.out / "baseline", args.out / "candidate", split, n)
               for split, n in (("gsm8k", 200), ("humaneval", 164))}
    write_json(args.out / "paired-results.json", results)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
