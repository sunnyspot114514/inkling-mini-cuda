# Inkling-2-Mini Compact CUDA · 8.35 GB

[🇺🇸 English](README.md) | [🇨🇳 中文说明](README.zh-CN.md)

**This checkpoint uses mixed-bit quantization, predominantly 2-bit and 3-bit.** Those two widths account for 90.18% of the routed experts' matrix weights. The complete text weights average **2.55 BPW (bits per weight)** on disk, including scales, codebooks and retained BF16/FP32 tensors. [Bit-width breakdown](https://huggingface.co/sunny114514/Inkling-2-Mini-Compact-CUDA/blob/d6443b1cf4f3dae2afcb48bd40c53d9f3fe4bc85/evaluation/quantization.json)

[Download the weights](https://huggingface.co/sunny114514/Inkling-2-Mini-Compact-CUDA) · [CUDA runner and source](https://github.com/sunnyspot114514/inkling-mini-cuda) · [Windows DLL](https://github.com/sunnyspot114514/inkling-mini-cuda/releases/tag/v0.1.0)

Run Inkling-2-Mini locally on a **12 GB GeForce RTX 3060**. The 8.35 GB expert mix keeps all 24 physical layers and all 128 experts per layer. Its CUDA runner reads the packed weights on the GPU and uses the model's chat template for arithmetic, code generation and multi-turn conversation.

## What you get

- **A smaller checkpoint.** The 61 runtime files total **8,349,693,242 bytes**, saving **524,513,809 bytes (5.91%)** against the 8.87 GB parent. That is 524.5 MB less to download and store.
- **147 of 164 HumanEval functions passed.** At a 1,536-token output budget, this model passed two more Python tasks than its parent on the same RTX 3060. Its GSM8K score was 192/200. [Paired scores and per-task totals](https://huggingface.co/sunny114514/Inkling-2-Mini-Compact-CUDA/blob/v0.1.0/evaluation/budget-1536/paired/summary.json)
- **A practical Windows setup.** The release includes the exact SM 8.6 DLL used for evaluation, the CUDA source, a checksum verifier and a chat command that keeps the model loaded between turns.

## Results

| Checkpoint | Runtime GB | GSM8K / 200 at 1,024 | HumanEval / 164 at 1,024 | GSM8K / 200 at 1,536 | HumanEval / 164 at 1,536 |
|---|---:|---:|---:|---:|---:|
| **Compact CUDA expert mix** | **8.350** | 191 | **132** | 192 | **147** |
| LookAtMyQuant parent | 8.874 | 191 | 130 | 195 | 145 |

These are paired local runs using the first 200 GSM8K test questions and all 164 executable HumanEval tasks (pass@1). Both targets used the packaged chat template, greedy decoding, `reasoning_effort="none"`, batch size two and FP16 K/V. The column headings give the maximum generated tokens per response. For the longer budget, all 70 replayed sequences for the expert mix matched their original greedy prefixes. [1,024-token scorecard](https://huggingface.co/sunny114514/Inkling-2-Mini-Compact-CUDA/blob/v0.1.0/evaluation/scorecard.json) · [1,536-token per-item outcomes](https://huggingface.co/sunny114514/Inkling-2-Mini-Compact-CUDA/blob/v0.1.0/evaluation/budget-1536/items.json)

Peak PyTorch allocations for this checkpoint during the 1,024-token runs were **8.329 GiB on GSM8K** and **8.380 GiB on HumanEval**. Aggregate output throughput for the two simultaneous requests was 5.401 and 5.636 tokens/s respectively. File sizes above use decimal GB. GPU memory here uses binary GiB. [Runtime measurements](https://huggingface.co/sunny114514/Inkling-2-Mini-Compact-CUDA/blob/v0.1.0/evaluation/scorecard.json)

## Quick start on Windows

The tested setup is Windows 10 x64, Python 3.12.10, RTX 3060 12 GB (SM 8.6), NVIDIA driver 560.94 and PyTorch 2.6.0+cu124. The commands below use PowerShell and Git. Run the blocks in the same PowerShell session.

### 1. Install the runner

Choose the drive where you want the project and weights. This example uses `E:\`. The virtual environment is local to the clone.

```powershell
Set-Location E:\
git clone --branch v0.1.0 --depth 1 https://github.com/sunnyspot114514/inkling-mini-cuda.git
Set-Location .\inkling-mini-cuda
py -3.12 -m venv .venv
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
& $py -m pip install -r requirements.txt "huggingface-hub==2.4.0"
& $py -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

The last command prints the installed PyTorch version and the NVIDIA GPU name.

### 2. Download and verify the release

This downloads the tagged weights into `expert-mix-8350`, fetches the prebuilt Windows kernel and verifies both against the release checksums.

```powershell
& .\.venv\Scripts\hf.exe download sunny114514/Inkling-2-Mini-Compact-CUDA --revision v0.1.0 --local-dir .\expert-mix-8350
Invoke-WebRequest -Uri "https://github.com/sunnyspot114514/inkling-mini-cuda/releases/download/v0.1.0/inkling_packed.dll" -OutFile .\inkling_packed.dll
& $py .\verify_download.py .\expert-mix-8350 --release-dll .\inkling_packed.dll
```

The verifier prints the file count and the SHA-256 of the DLL. The weight folder contains `manifest.json`, the tokenizer and the packed layer files. You can substitute an absolute path to that folder in the commands below.

### 3. Generate a first answer

```powershell
& $py .\run_inkling_cuda.py .\expert-mix-8350 --library .\inkling_packed.dll --prompt "What is 17 times 23? Give just the number." --max-new-tokens 256 --kv-dtype float16
```

The runner prints loading progress and then a JSON result. For this smoke prompt the tested final answer is **391**. The `completion` field contains the generated text, `output_tokens` is its length, and `reached_limit` tells you whether it used the full output budget.

### 4. Start a conversation

```powershell
& $py .\chat_inkling_cuda.py .\expert-mix-8350 --library .\inkling_packed.dll --context 4096 --max-new-tokens 1024 --kv-dtype float16
```

Type your message at the `>` prompt. Try `Write a Python function named add_one(n) that returns n + 1.` The CLI prints the final answer and keeps the conversation history in memory. Enter `/reset` to start a fresh conversation or `/exit` to release the model.

| Option | How to use it |
|---|---|
| `--max-new-tokens` | Sets the output budget for one reply. The examples use 256 for the arithmetic check and 1,024 for chat. |
| `--context` | Sets the chat history and reply budget together. With 4,096 total and 1,024 reserved for output, up to 3,072 tokens are available for the formatted conversation. The CLI trims older turns as history grows. |
| `--kv-dtype float16` | Uses the same FP16 attention cache as the published paired evaluations. |
| `--library` | Points to the downloaded DLL or your own build of `inkling_packed.cu`. |

### Build the CUDA kernel from source

Open an x64 MSVC Developer PowerShell in the source folder. The tested compiler is NVCC 12.6.20. To keep the supplied binary alongside your build, use a separate output name.

```powershell
nvcc -O3 -std=c++17 -arch=sm_86 --shared -o inkling_packed_local.dll inkling_packed.cu
& $py .\run_inkling_cuda.py .\expert-mix-8350 --library .\inkling_packed_local.dll --prompt "What is 17 times 23? Give just the number." --max-new-tokens 256 --kv-dtype float16
```

## How the expert mix was built

The allocation uses routing observations from GSM8K and MBPP training examples. The builder keeps the larger representation for the 16 most heavily weighted experts in each layer for either domain, then chooses compact expert pairs by estimated projection error, routing weight and bytes saved. **450 of 3,072 expert pairs** use both projections from the 5.38 GB parent. The 8.87 GB parent supplies the other expert projections, attention, routers, embeddings, output head and steering tensors. [Exact recipe](https://huggingface.co/sunny114514/Inkling-2-Mini-Compact-CUDA/blob/v0.1.0/recipe.json) · [Layer allocation](https://huggingface.co/sunny114514/Inkling-2-Mini-Compact-CUDA/blob/v0.1.0/evaluation/allocation-by-region.json)

To reproduce the assignment, run the following from the source checkout. It uses immutable parent revisions, the published routing profile and the original 8,350,000,000-byte target.

```powershell
& $py -m pip install threadpoolctl==3.6.0
& .\.venv\Scripts\hf.exe download shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-5.4GB --revision fb17b5be547b11e76a9f8f26239b4b4f59286fec --local-dir .\parents\small
& .\.venv\Scripts\hf.exe download shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-8.9GB --revision ab695af99964ea87d0e609ad8060374f05306ae7 --local-dir .\parents\large
& $py .\build_expert_candidate.py --large .\parents\large --small .\parents\small --router-profile .\evaluation\router-profile.npz --budget 8350000000 --rows 8 --seed 65937 --out .\rebuilt-835 --report .\rebuilt-hashes.json
```

## Credits and licenses

Thinking Machines Lab created Inkling-2-Mini. Shin Megami Boson produced the two tuned LookAtMyQuant parents, codebooks and packed format. This release contributes the routing-weighted mixed allocation, CUDA runner and paired evaluation. The derivative weights and this project's original code use Apache 2.0. The upstream reference implementation's MIT notice is included with the source. [Original model](https://huggingface.co/thinkingmachines/Inkling-2-Mini) · [5.38 GB parent](https://huggingface.co/shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-5.4GB) · [8.87 GB parent](https://huggingface.co/shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-8.9GB) · [Apache 2.0](https://github.com/sunnyspot114514/inkling-mini-cuda/blob/v0.1.0/LICENSE) · [MIT notice](https://github.com/sunnyspot114514/inkling-mini-cuda/blob/v0.1.0/licenses/LookAtMyQuant-MIT.txt) · [Original model use policy](https://thinkingmachines.ai/model-acceptable-use-policy/)

## Limitations

- The published runner handles text. Vision inference and a VQA score are outside this release. Native DSpark and 4-bit drafter acceptance rates and speedups remain unmeasured for this checkpoint.
- Single-user prefill and decode speeds at 512 and 4,096 input tokens, maximum usable context, original-teacher KL and the low-reasoning-effort comparison remain unmeasured. The 4,096-token chat setting is a default. The quoted throughput is a batch-two measurement.
- The binary and example environment were tested on Windows x64 with an RTX 3060. Other GPU architectures and operating systems require separate validation. These paired benchmark scores describe the tested prompts and output budgets.
