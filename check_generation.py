"""Verify short math, executable Python, and batched cache handling before a long run."""

import argparse
from pathlib import Path
import time

import torch

from evaluate import check_code, completion_code, final_text, number, sandbox_command, sandbox_selftest, write_json
from run_inkling_cuda import generate, generate_prompts, load_model


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "library", "report"):
        p.add_argument("--"+key, type=Path, required=True)
    p.add_argument("--kv-dtype", default="float16", choices=["model", "float16"])
    p.add_argument("--bwrap", default="/usr/bin/bwrap")
    p.add_argument("--wsl-distro", default="Ubuntu-22.04")
    a = p.parse_args()
    command = sandbox_command(a.bwrap, a.wsl_distro)
    sandbox_selftest(command)
    before = time.perf_counter()
    model, tok = load_model(a.model, a.library, progress=True)
    loaded = time.perf_counter()-before
    prompts = ["What is 17 times 23? Give just the number.",
               "Return a short Python function named add_one(n) that returns n + 1. Use a Python code block."]
    pair = generate_prompts(model, tok, prompts, 256, kv_dtype=a.kv_dtype)
    single = generate(model, tok, prompts[0], 256, kv_dtype=a.kv_dtype)
    case = {"entry_point": "add_one", "prompt": "def add_one(n):\n",
            "test": "def check(f):\n    assert f(3) == 4 and f(-1) == 0 and f(0) == 1"}
    code_pass, detail = check_code(command, completion_code(final_text(pair[1]["completion"]), case), case)
    checks = {"single_math": number(final_text(single["completion"])) == 391,
              "batched_math": number(final_text(pair[0]["completion"])) == 391, "batched_code": code_pass}
    report = {"device": torch.cuda.get_device_name(), "model": a.model.name, "kv_dtype": a.kv_dtype,
              "load_seconds": loaded, "checks": checks, "code_test": detail,
              "math_tokens_identical": single["output_ids"] == pair[0]["output_ids"],
              "batch": pair, "single": single}
    a.report.parent.mkdir(parents=True, exist_ok=True)
    write_json(a.report, report)
    print(checks, "identical math tokens:", report["math_tokens_identical"], flush=True)
    if not all(checks.values()):
        raise SystemExit("Short generation control requires inspection")
