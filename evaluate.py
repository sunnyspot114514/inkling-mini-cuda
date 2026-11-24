"""Paired, resumable greedy GSM8K and executable HumanEval assessment."""

import argparse
import ast
import copy
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import textwrap


GSM_PROMPT = "Solve the following problem. Show your work and finish with #### followed by the numeric answer.\n\n"
CODE_PROMPT = "Complete the Python function below. Return the complete function and any required imports in a Python code block.\n\n"
WRAPPER = """
import contextlib, json, os, sys
case = json.load(sys.stdin)
ns = {"__name__": "__candidate__"}
passed = False
detail = "unknown"
with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
    try:
        if case["preamble"]:
            exec(compile(case["preamble"], "<benchmark prompt>", "exec"), ns)
        exec(compile(case["code"], "<submission>", "exec"), ns)
        exec(compile(case["test"], "<tests>", "exec"), ns)
        ns["check"](ns[case["entry_point"]])
        passed, detail = True, "passed"
    except BaseException as exc:
        detail = type(exc).__name__
print(json.dumps({"passed": passed, "detail": detail}))
"""


def sandbox_command(bwrap, distro):
    prefix = ["wsl", "-d", distro, "--exec"] if os.name == "nt" else []
    return prefix + [
        "/usr/bin/timeout", "-k", "1", "9", bwrap, "--unshare-all", "--new-session", "--die-with-parent",
        "--cap-drop", "ALL", "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib",
        "--ro-bind", "/lib64", "/lib64", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp",
        "--chdir", "/tmp", "--clearenv", "--setenv", "PATH", "/usr/bin", "--", "/usr/bin/prlimit",
        "--cpu=5", "--as=536870912", "--nproc=16", "--fsize=1048576", "--", "/usr/bin/python3",
        "-I", "-S", "-c", WRAPPER,
    ]


def check_code(command, code, item):
    # A full-function chat answer still gets the imports and helper definitions
    # supplied in the official prompt, just as a function-body completion does.
    supplied = item.get("prompt", "") if str(item.get("task_id", "")).startswith("HumanEval/") else ""
    preamble = supplied if supplied and not code.startswith(supplied) else ""
    payload = json.dumps({"code": code, "preamble": preamble, "test": item["test"], "entry_point": item["entry_point"]})
    proc = subprocess.run(command, input=payload.encode(), capture_output=True, timeout=40)
    if proc.returncode in (124, 137, 152):
        return False, "timeout or resource limit"
    if proc.returncode:
        raise RuntimeError(f"Isolated runner failed ({proc.returncode}): {proc.stderr[:300]!r}")
    result = json.loads(proc.stdout)
    return result["passed"], result["detail"]


def sandbox_selftest(command):
    tests = "def check(fn):\n    import os, socket\n    assert not os.path.exists('/mnt/c') and not os.path.exists('/home')\n    assert all(name == 'lo' for ix, name in socket.if_nameindex())\n    assert fn(2) == 3"
    item = {"test": tests, "entry_point": "f"}
    if not check_code(command, "def f(x): return x + 1", item)[0]:
        raise RuntimeError("Isolated evaluator failed its positive control")
    if check_code(command, "def f(x): return x - 1", item)[0]:
        raise RuntimeError("Isolated evaluator failed its negative control")


def final_text(generation):
    if "<|content_text|>" in generation:
        generation = generation.rsplit("<|content_text|>", 1)[-1]
    elif "<|content_thinking|>" in generation:
        return ""
    return re.split(r"<\|(?:end_message|end_turn|endoftext)\|>", generation)[0].strip("\r\n")


def number(text):
    marked = "####" in text
    candidates = re.findall(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?", text.rsplit("####", 1)[-1])
    if not candidates:
        return None
    try:
        return Decimal(candidates[0 if marked else -1].replace(",", ""))
    except InvalidOperation:
        return None


def completion_code(text, item):
    blocks = re.findall(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    pieces = blocks or [text]
    for part in reversed(pieces):
        try:
            tree = ast.parse(part)
        except (SyntaxError, ValueError):
            continue
        if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == item["entry_point"]
               for node in tree.body):
            return part
    return item["prompt"] + pieces[-1]


def sha(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_json(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", delete=False) as f:
        tmp = Path(f.name)
        try:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write("\n")
        except BaseException:
            f.close()
            tmp.unlink(missing_ok=True)
            raise
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_cases(directory, split):
    if split == "mbpp-calibration":
        path = directory / "mbpp-train.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if len(rows) != 374 or min(row["task_id"] for row in rows) != 601:
            raise ValueError("Unexpected MBPP training split")
        cases = []
        for row in rows:
            try:
                defined = {n.name for n in ast.parse(row["code"]).body if isinstance(n, ast.FunctionDef)}
                called = {n.func.id for n in ast.walk(ast.parse("\n".join(row["test_list"])))
                          if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
            except SyntaxError:
                continue
            names = defined & called
            if len(names) != 1:
                continue
            name, = names
            test = (row["test_setup_code"] or "") + "\ndef check(candidate):\n"
            test += textwrap.indent("\n".join(row["test_list"]), "    ")
            cases.append({"task_id": f"MBPP/{row['task_id']}", "entry_point": name, "test": test,
                          "prompt": f"{row['text']}\nUse the function name {name}."})
            if len(cases) == 32:
                break
        return path, cases
    if split == "humaneval":
        path = directory / "HumanEval.jsonl.gz"
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            cases = [json.loads(line) for line in stream]
        if len(cases) != 164:
            raise ValueError("Unexpected HumanEval task count")
        return path, cases
    path = directory / ("gsm8k-test.jsonl" if split == "gsm8k" else "gsm8k-train.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if split == "gsm8k" and len(rows) != 1319:
        raise ValueError("Unexpected GSM8K test task count")
    return path, rows[:200] if split == "gsm8k" else rows[400:432]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", nargs="+", choices=["gsm8k", "humaneval", "gsm8k-calibration", "mbpp-calibration"], required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--start-index", type=int, default=0, help="Resume or schedule a fixed test window")
    p.add_argument("--stop-index", type=int, help="Exclusive end of the scheduled window")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--kv-dtype", choices=["model", "float16"], default="model")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--bwrap", default="/usr/bin/bwrap")
    p.add_argument("--wsl-distro", default="Ubuntu-22.04")
    a = p.parse_args()
    if a.batch_size < 1 or a.max_new_tokens < 1 or (a.limit is not None and a.limit < 1):
        p.error("The generation and item limits must be positive")
    if len(set(a.split)) != len(a.split):
        p.error("Specify each benchmark split once")
    if a.start_index < 0 or (a.stop_index is not None and a.stop_index <= a.start_index):
        p.error("Use an increasing nonnegative index window")
    return a


def run_split(a, lm=None, tok=None):
    file, cases = load_cases(a.data, a.split)
    protocol = {
        "split": a.split, "data_sha256": sha(file), "max_new_tokens": a.max_new_tokens,
        "greedy": True, "reasoning_effort": "none", "template_sha256": sha(a.model / "chat_template.jinja"),
        "model": str(a.model.resolve()), "model_manifest_sha256": sha(a.model / "manifest.json"),
        "steering_sha256": sha(a.model / "steering.safetensors"), "kernel_sha256": sha(a.library),
        "runner_sha256": sha(Path(__file__).with_name("run_inkling_cuda.py")),
        "packed_loader_sha256": sha(Path(__file__).with_name("packed_cuda.py")),
        "evaluation_sha256": sha(Path(__file__)), "total_cases": len(cases),
        "batch_size": a.batch_size, "item_limit": a.limit,
        "runtime_versions": {key: version(key) for key in ("torch", "transformers", "tokenizers", "safetensors")},
        "kv_dtype": a.kv_dtype, "head_runner_sha256": sha(Path(__file__).with_name("head_residual.py")),
        "head_correction_sha256": sha(a.model / "head_residual.safetensors") if (a.model / "head_residual.safetensors").is_file() else None,
        "head_settings_sha256": sha(a.model / "head_residual.json") if (a.model / "head_residual.json").is_file() else None,
    }
    a.out.mkdir(parents=True, exist_ok=True)
    spec = a.out / "protocol.json"
    if spec.exists() and json.loads(spec.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Resume requires the same model and evaluation protocol. Use a new output directory.")
    write_json(spec, protocol)
    paths = [a.out / f"{ix:04}.json" for ix in range(len(cases))]
    command = sandbox_command(a.bwrap, a.wsl_distro)
    code_task = a.split in ("humaneval", "mbpp-calibration")
    if code_task:
        sandbox_selftest(command)
    from run_inkling_cuda import generate_prompts, load_model
    total = len(cases) if a.limit is None else min(len(cases), a.limit)
    stop = total if a.stop_index is None else min(total, a.stop_index)
    first_batch = (a.start_index // a.batch_size) * a.batch_size
    for start in range(first_batch, stop, a.batch_size):
        indices = range(start, min(start + a.batch_size, total))
        wanted = [ix for ix in indices if a.start_index <= ix < stop]
        if all(paths[ix].exists() for ix in wanted):
            continue
        if lm is None:
            lm, tok = load_model(a.model, a.library, progress=True)
        prompts = [(CODE_PROMPT + cases[ix]["prompt"]) if code_task else (GSM_PROMPT + cases[ix]["question"])
                   for ix in indices]
        for ix, result in zip(indices, generate_prompts(lm, tok, prompts, a.max_new_tokens, kv_dtype=a.kv_dtype)):
            if ix not in wanted or paths[ix].exists():
                continue
            item = cases[ix]
            answer = final_text(result["completion"])
            if code_task:
                passed, detail = check_code(command, completion_code(answer, item), item)
            else:
                predicted, expected = number(answer), number(item["answer"])
                passed, detail = predicted is not None and predicted == expected, str(predicted)
            result.update({"index": ix, "passed": passed, "detail": detail, "id": item.get("task_id", str(ix))})
            write_json(paths[ix], result)
            print(f"{a.split} {ix+1}/{len(cases)} passed={passed} tokens={result['output_tokens']}", flush=True)
    finished = [json.loads(path.read_text(encoding="utf-8")) for path in paths if path.exists()]
    summary = {"scored": len(finished), "correct": sum(v["passed"] for v in finished), "total": len(cases)}
    write_json(a.out / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return lm, tok


def main():
    args = parse_args()
    model, tokenizer = None, None
    for split in args.split:
        a = copy.copy(args)
        a.split = split
        a.out = args.out / split if len(args.split) > 1 else args.out
        model, tokenizer = run_split(a, model, tokenizer)


if __name__ == "__main__":
    main()
