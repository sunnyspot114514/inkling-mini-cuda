# Inkling-2-Mini Compact CUDA · 8.35 GB

[🇺🇸 English](README.md) | [🇨🇳 中文说明](README.zh-CN.md)

[下载模型权重](https://huggingface.co/sunny114514/Inkling-2-Mini-Compact-CUDA) · [CUDA 运行器与源码](https://github.com/sunnyspot114514/inkling-mini-cuda) · [Windows DLL](https://github.com/sunnyspot114514/inkling-mini-cuda/releases/tag/v0.1.0)

在 **RTX 3060 12 GB** 上本地运行 Inkling-2-Mini。8.35 GB 专家混合版保留了 24 个物理层和每层全部 128 个专家。CUDA 运行器在显卡上直接读取压缩权重，使用模型自带的对话模板，支持数学问答、代码生成和多轮对话。

## 这个版本的优点

- **下载和存储都更省。** 61 个运行文件共 **8,349,693,242 字节**，比 8.87 GB 父模型省下 **524,513,809 字节（5.91%）**，约合 524.5 MB。
- **HumanEval 通过 147/164 题。** 在同一张 RTX 3060 上，每次最多生成 1,536 个 token，这个版本比父模型多通过两道可执行 Python 题，GSM8K 得分为 192/200。[配对成绩与各项计数](https://huggingface.co/sunny114514/Inkling-2-Mini-Compact-CUDA/blob/v0.1.0/evaluation/budget-1536/paired/summary.json)
- **Windows 上可直接使用预编译内核。** 发布包提供实测所用的 SM 8.6 DLL、CUDA 源码、文件校验工具，以及可常驻模型的多轮对话命令。

## 实测成绩

| 模型 | 运行文件 GB | GSM8K / 200，上限 1,024 | HumanEval / 164，上限 1,024 | GSM8K / 200，上限 1,536 | HumanEval / 164，上限 1,536 |
|---|---:|---:|---:|---:|---:|
| **Compact CUDA 专家混合版** | **8.350** | 191 | **132** | 192 | **147** |
| LookAtMyQuant 父模型 | 8.874 | 191 | 130 | 195 | 145 |

GSM8K 取测试集前 200 题，HumanEval 对全部 164 个函数进行可执行 pass@1 评分。两个模型均使用随权重提供的对话模板、贪心解码、`reasoning_effort="none"`、双样本批次和 FP16 K/V 缓存。表头的上限指每次回答最多生成的 token 数。延长输出预算时，本版本重跑了涉及原上限的 70 条输出，全部匹配原先的贪心前缀。[1,024-token 评分卡](https://huggingface.co/sunny114514/Inkling-2-Mini-Compact-CUDA/blob/v0.1.0/evaluation/scorecard.json) · [1,536-token 逐题结果](https://huggingface.co/sunny114514/Inkling-2-Mini-Compact-CUDA/blob/v0.1.0/evaluation/budget-1536/items.json)

在 1,024-token 测试中，本版本的 PyTorch 张量峰值分配为 **GSM8K 8.329 GiB、HumanEval 8.380 GiB**。两个请求合计的输出吞吐分别为 5.401 和 5.636 token/s。文件体积使用十进制 GB，显存使用二进制 GiB。[运行实测数据](https://huggingface.co/sunny114514/Inkling-2-Mini-Compact-CUDA/blob/v0.1.0/evaluation/scorecard.json)

## Windows 使用指南

实测环境为 Windows 10 x64、Python 3.12.10、RTX 3060 12 GB（SM 8.6）、NVIDIA 驱动 560.94 和 PyTorch 2.6.0+cu124。准备好 PowerShell 和 Git，以下命令在同一个 PowerShell 窗口中依次执行。

### 1. 安装运行器

先选择存放项目和模型的盘符，示例使用 `E:\`。Python 环境会建在项目目录里。

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

最后一条命令会显示 PyTorch 版本和 NVIDIA 显卡名称。

### 2. 下载并校验文件

下列命令把指定发布版本的权重下载到 `expert-mix-8350`，同时取回预编译 Windows 内核，再核对两者的发布校验值。

```powershell
& .\.venv\Scripts\hf.exe download sunny114514/Inkling-2-Mini-Compact-CUDA --revision v0.1.0 --local-dir .\expert-mix-8350
Invoke-WebRequest -Uri "https://github.com/sunnyspot114514/inkling-mini-cuda/releases/download/v0.1.0/inkling_packed.dll" -OutFile .\inkling_packed.dll
& $py .\verify_download.py .\expert-mix-8350 --release-dll .\inkling_packed.dll
```

校验工具会列出通过检查的文件数量和 DLL 的 SHA-256。权重目录里有 `manifest.json`、分词器和逐层压缩文件。模型放在其他位置时，把后面命令中的目录名换成实际绝对路径即可。

### 3. 运行第一个问题

```powershell
& $py .\run_inkling_cuda.py .\expert-mix-8350 --library .\inkling_packed.dll --prompt "What is 17 times 23? Give just the number." --max-new-tokens 256 --kv-dtype float16
```

运行器会先打印加载进度，再输出 JSON。这道冒烟题的实测最终答案是 **391**。`completion` 是生成文本，`output_tokens` 是输出长度，`reached_limit` 表示是否用满本次输出预算。

### 4. 进入多轮对话

```powershell
& $py .\chat_inkling_cuda.py .\expert-mix-8350 --library .\inkling_packed.dll --context 4096 --max-new-tokens 1024 --kv-dtype float16
```

在 `>` 提示符后输入问题，例如「用 Python 写一个 `add_one(n)` 函数，返回 `n + 1`」。程序显示最终回答，并在内存中保留对话历史。输入 `/reset` 开始新对话，输入 `/exit` 退出并释放模型。

| 参数 | 用法 |
|---|---|
| `--max-new-tokens` | 一次回答的输出预算。示例为算术题分配 256 个 token，为多轮对话的每次回答分配 1,024 个。 |
| `--context` | 对话历史和本次回答共同使用的预算。总量 4,096、预留输出 1,024 时，格式化后的对话可使用 3,072 个 token。历史变长后程序会移除较早的轮次。 |
| `--kv-dtype float16` | 使用与上表配对评测相同的 FP16 注意力缓存。 |
| `--library` | 指向下载的 DLL，或从 `inkling_packed.cu` 自行编译得到的内核。 |

### 自行编译 CUDA 内核

在源码目录打开 x64 MSVC Developer PowerShell。实测编译器为 NVCC 12.6.20。以下命令使用独立文件名，方便同时保留发布版和自行编译的 DLL。

```powershell
nvcc -O3 -std=c++17 -arch=sm_86 --shared -o inkling_packed_local.dll inkling_packed.cu
& $py .\run_inkling_cuda.py .\expert-mix-8350 --library .\inkling_packed_local.dll --prompt "What is 17 times 23? Give just the number." --max-new-tokens 256 --kv-dtype float16
```

## 权重如何构建

构建器从 GSM8K 和 MBPP 的训练样本中统计路由权重。对每一层，数学或代码样本中权重最高的 16 个专家保留较大的编码，再根据投影误差、路由权重和实际可节省字节数选择压缩对象。**3,072 个专家对中有 450 个**同时采用 5.38 GB 父模型的两组投影。其余专家投影、注意力、路由器、词嵌入、输出头和 steering 张量来自 8.87 GB 父模型。[完整构建配方](https://huggingface.co/sunny114514/Inkling-2-Mini-Compact-CUDA/blob/v0.1.0/recipe.json) · [分层分配数据](https://huggingface.co/sunny114514/Inkling-2-Mini-Compact-CUDA/blob/v0.1.0/evaluation/allocation-by-region.json)

要复现这次分配，请在源码目录执行以下命令。命令固定了两个父模型的版本，使用随源码发布的路由统计和原始 8,350,000,000 字节预算。

```powershell
& $py -m pip install threadpoolctl==3.6.0
& .\.venv\Scripts\hf.exe download shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-5.4GB --revision fb17b5be547b11e76a9f8f26239b4b4f59286fec --local-dir .\parents\small
& .\.venv\Scripts\hf.exe download shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-8.9GB --revision ab695af99964ea87d0e609ad8060374f05306ae7 --local-dir .\parents\large
& $py .\build_expert_candidate.py --large .\parents\large --small .\parents\small --router-profile .\evaluation\router-profile.npz --budget 8350000000 --rows 8 --seed 65937 --out .\rebuilt-835 --report .\rebuilt-hashes.json
```

## 致谢与许可证

Thinking Machines Lab 创建了 Inkling-2-Mini。Shin Megami Boson 制作了两个经过调优的 LookAtMyQuant 量化父模型、码本和压缩格式。本次发布的贡献是基于路由权重的混合分配、CUDA 运行器和配对评测。派生权重和本项目原创代码采用 Apache 2.0，源码随附上游参考实现的 MIT 声明。[原始模型](https://huggingface.co/thinkingmachines/Inkling-2-Mini) · [5.38 GB 父模型](https://huggingface.co/shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-5.4GB) · [8.87 GB 父模型](https://huggingface.co/shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-8.9GB) · [Apache 2.0](https://github.com/sunnyspot114514/inkling-mini-cuda/blob/v0.1.0/LICENSE) · [MIT 声明](https://github.com/sunnyspot114514/inkling-mini-cuda/blob/v0.1.0/licenses/LookAtMyQuant-MIT.txt) · [原模型使用政策](https://thinkingmachines.ai/model-acceptable-use-policy/)

## 限制

- 本次发布的运行器支持文本生成。视觉推理和 VQA 成绩未包含在本版中。本模型使用原生或 4-bit DSpark 草稿器时的接受率与加速比尚未实测。
- 单用户在 512 和 4,096 输入 token 时的预填充与解码速度、最大可用上下文、相对原模型的 KL 和低思考档对比尚未实测。4,096 是对话程序的默认预算，正文中的吞吐是双样本批次数据。
- 二进制内核及示例环境已在 Windows x64 和 RTX 3060 上验证。其他操作系统和显卡架构仍需单独验证。配对成绩适用于列出的题目和输出预算。
