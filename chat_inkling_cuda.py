"""Keep a packed Inkling checkpoint resident for an interactive text conversation."""

import argparse
from pathlib import Path

from evaluate import final_text
from run_inkling_cuda import generate, load_model


def format_turn(tokenizer, previous, prompt, budget):
    messages = [*previous, {"role": "user", "content": prompt}]
    trimmed = 0
    while True:
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  reasoning_effort="none")
        length = len(tokenizer.encode(formatted, add_special_tokens=False))
        if length <= budget:
            return messages, formatted, trimmed
        if len(messages) < 3:
            raise ValueError(f"This prompt needs {length} input tokens; the available input budget is {budget}.")
        messages = messages[2:]
        trimmed += 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model_dir", type=Path)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--kv-dtype", choices=["model", "float16"], default="float16")
    a = p.parse_args()
    if not 0 < a.max_new_tokens < a.context:
        p.error("The output limit must be positive and below the context size")
    model, tokenizer = load_model(a.model_dir, a.library, progress=True)
    history = []
    print("Ready. Use /reset to clear history or /exit to quit.")
    while True:
        try:
            prompt = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if prompt == "/exit":
            break
        if prompt == "/reset":
            history.clear()
            print("History cleared.")
            continue
        if not prompt:
            continue
        try:
            messages, rendered, removed = format_turn(tokenizer, history, prompt, a.context - a.max_new_tokens)
        except ValueError as err:
            print(err)
            continue
        if removed:
            print(f"Removed {removed} earlier turns to fit the context.")
        result = generate(model, tokenizer, rendered, a.max_new_tokens, raw=True, kv_dtype=a.kv_dtype)
        answer = final_text(result["completion"]).strip()
        if answer:
            print(answer, flush=True)
            history = [*messages, {"role": "assistant", "content": answer}]
        else:
            print(f"Generation ended before a final answer ({result['output_tokens']} output tokens).", flush=True)


if __name__ == "__main__":
    main()
