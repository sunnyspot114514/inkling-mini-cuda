"""Freeze public English, Python, tutoring-chat and multilingual token windows for KL."""

import argparse
import hashlib
import json
from pathlib import Path
import struct

import pyarrow.parquet as pq
from transformers import PreTrainedTokenizerFast

from evaluate import sha, write_json


LANGUAGES = ("ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "th", "tr", "vi", "zh")
CODE_MODELS = ("bert", "gpt2", "llama", "vit")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "tokenizer", "transformers-source", "out"):
        p.add_argument("--"+name, type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tok = PreTrainedTokenizerFast.from_pretrained(args.tokenizer, local_files_only=True)
    wiki = args.data/"supplemental/wikitext-2-raw-v1/test-00000-of-00001.parquet"
    xnli = args.data/"supplemental/all_languages/test-00000-of-00001.parquet"
    gsm = args.data/"gsm8k-test.jsonl"
    text = "\n".join(pq.read_table(wiki, columns=["text"]).column("text").to_pylist())
    (args.out/"wikitext-test.txt").write_bytes(text.encode("utf-8"))
    encode = lambda content: tok.encode(content, add_special_tokens=False)
    all_ids = encode(text)
    windows = []
    def add(domain, label, ids):
        ids = ids[:128]
        if len(ids)!=128 or any(t<0 or t>=200058 for t in ids):
            raise ValueError(f"Invalid sample {label}")
        digest = hashlib.sha256(struct.pack("<128I",*ids)).hexdigest()
        windows.append({"domain":domain,"id":label,"token_ids_sha256":digest,"ids":ids})
    for i in range(4):
        start = i * (len(all_ids)//4)
        add("english",f"wikitext/raw-test/token-{start}",all_ids[start:start+128])
    code_hashes = {}
    for name in CODE_MODELS:
        path = args.transformers_source/f"models/{name}/modeling_{name}.py"
        source = path.read_text(encoding="utf-8")
        code_hashes[path.name] = hashlib.sha256(source.encode()).hexdigest()
        add("python",f"transformers-5.24.0/{name}/token-2000",encode(source)[2000:2128])
    questions = [json.loads(line) for line in gsm.read_text(encoding="utf-8").splitlines()]
    for start in (277, 567, 823, 1135):
        messages=[]
        for item in questions[start:start+3]:
            messages.extend(({"role":"user","content":item["question"]},{"role":"assistant","content":item["answer"]}))
        ids = tok.apply_chat_template(messages, tokenize=True, return_dict=True, reasoning_effort="none")["input_ids"]
        add("tutoring_chat",f"gsm8k/test/{start}/three-turns",ids)
    rows = pq.read_table(xnli, columns=["premise","hypothesis"]).to_pylist()
    for language in LANGUAGES:
        for start in (100,2600):
            lines=[]
            for row in rows[start:start+40]:
                hyp = row["hypothesis"]
                translation = dict(zip(hyp["language"],hyp["translation"],strict=True))
                lines.extend((row["premise"][language],translation[language]))
            add("xnli_"+language,f"xnli/test/{language}/row-{start}",
                encode("\n".join(lines)))
    result = {"format":1, "window_tokens":128, "score_position_start":32, "score_position_stop":128,
              "vocab_size":200058, "tokenizer_sha256":sha(args.tokenizer/"tokenizer.json"),
              "chat_template_sha256":sha(args.tokenizer/"chat_template.jinja"),
              "sources":{"wikitext":{"repo":"Salesforce/wikitext","revision":"b08601e04326c79dfdd32d625aee71d232d685c3","sha256":sha(wiki)},
                         "xnli":{"repo":"facebook/xnli","revision":"b8dd5d7af51114dbda02c0e3f6133f332186418e","sha256":sha(xnli)},
                         "gsm8k":{"revision":"3101c7d5072418e28b9008a6636bde82a006892c","sha256":sha(gsm)},
                         "transformers":{"version":"5.24.0","normalized_source_sha256":code_hashes}},
              "windows":windows}
    if len({w["token_ids_sha256"] for w in windows})!=len(windows):
        raise ValueError("Duplicate token windows")
    write_json(args.out/"windows.json",result)
    print(f"Frozen {len(windows)} windows, {len(windows)*96} scored positions, {len(all_ids)} prefill corpus tokens")


if __name__ == "__main__":
    main()
