# Spectral Convolution 算子与 FNO-NS

本项目面向 2026 飞翔杯“模型与算子开发”赛道：

- 必选题：使用 SUPA / `torch.extension` 实现二维 Spectral Convolution；
- 进阶题 C：复用自研谱卷积，构建 FNO 求解二维 Navier–Stokes 方程；
- 目标平台：壁仞 BIREN GPU，CPU/CUDA 仅用于参考实现开发和数值对照。

> 当前已完成 BIREN 自定义算子前向/反向正确性、64/128/256 性能测试，
> 以及公开 Navier–Stokes 数据的 100 epoch 严格 SUPA FNO 训练与独立
> 测试。当前可引用结果与限制统一记录在
> [results/REPORT.md](results/REPORT.md)。

## 计算流程

输入 `x [B, C_in, H, W]` 先经 FFT 进入频域，仅保留上下两组低频
模态；低频复数特征与可学习权重完成通道缩并，再经逆 FFT 返回
`y [B, C_out, H, W]`。核心频域复数乘及其输入/权重梯度由 SUPA
extension 实现，PyTorch 版本只作为正确性基线。

## 仓库结构

```text
src/reference/       PyTorch 参考实现
src/supa/            SUPA kernel 与 PyTorch 扩展
tests/               正确性、反向与性能测试
fno_ns/              FNO-NS 数据、训练与评估流程
scripts/             环境与硬件探针
results/             可复现报告、表格和精选图表
agent_logs/          真实 Agent 协作记录
references/          外部实现调研与架构参考
skill.md             作品能力与调用入口
```

`agent_logs/chat_exports/` 另含 7 段从本次 Codex 主任务及并行 Agent
真实回合提取、已脱敏的对话记录，覆盖赛题列出的全部 6 类 Agent 使用场景。

## 快速开始

### 本地参考环境

项目根目录已有 `.venv` 时直接复用。Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python tests\test_correctness.py
python tests\test_backward.py
```

不要在 BIREN 官方容器中用 PyPI 的标准 `torch` 覆盖平台预装版本。平台与本地依赖差异见 [DEPENDENCIES.md](DEPENDENCIES.md)。

### BIREN 硬件检查

将仓库同步到服务器并进入项目根目录：

```bash
python scripts/probe_biren.py
python scripts/probe_biren.py --json results/biren_environment.json
# 扩展构建完成后的正式验收
python scripts/probe_biren.py --require-extension
```

探针会检查 `brsmi`、Python、PyTorch、`torch_br`、SUPA 设备、复数张量、FFT、反向传播及项目扩展；某项失败会保留诊断。普通模式区分“平台能力”和“项目扩展能力”，`--require-extension` 会在扩展不可用时返回失败。生成的 JSON 不包含密钥，但提交前仍应人工检查主机名等环境字段。

已验证的赛事环境为 Biren106M 32512 MiB、SUPA/Driver 1.11、SDK
`1.11.0.0.rc2`、Python 3.10.12、PyTorch 2.9.0 和
`torch_br 1.10.0.20900+br1xx`。该镜像通过 `import torch_br` 注册
`torch.supa`，不依赖 `torch_supa`。

该环境的 native `torch.fft.rfft2` 实测会遗漏 H 维变换，但对应 native
`irfft2` 仍可产生自洽 roundtrip，不能据此判断二维 FFT 正确。相同输入与
CPU 对照时 native relative L2 约为 `1`，改用
`rfft(dim=-1) → fft(dim=-2)` 后最终环境探针实测降至
`4.860923384338554e-8`。因此
`SupaSpectralConv2d` 在 SUPA 上固定使用顺序 FFT，逆变换使用
`ifft(dim=-2) → irfft(dim=-1)`；CPU/CUDA 仍使用 native 二维接口。

首次构建扩展前设置动态库顺序，再运行构建脚本：

```bash
TORCH_BR_BASE="$(python -c 'import importlib.util; from pathlib import Path; s=importlib.util.find_spec("torch_br"); print(Path(s.origin).resolve().parent)')"
TORCH_LIB_PATHS="$(python -c 'from torch.utils.cpp_extension import library_paths; print(":".join(library_paths()))')"
SUPA_PATH="${SUPA_PATH:-/usr/local/birensupa/sdk/1.11.0.0.rc2/supa}"
export LD_LIBRARY_PATH="${TORCH_BR_BASE}/lib:${TORCH_LIB_PATHS}:${SUPA_PATH}/lib:${LD_LIBRARY_PATH:-}"
bash src/supa/csrc/build.sh
```

`torch_br/lib` 必须位于 `torch/lib` 之前；顺序相反会让 `torch_br` 加载到
不匹配的 `libbr_common`，典型表现是 `undefined symbol`，并非 Python 包缺失。

### 统一验证

```bash
python tests/test_correctness.py
python tests/test_backward.py
python -m pytest tests/test_supa_operator.py
# BIREN 正式前向/输入梯度/两组权重梯度验收，并保存结构化结果
python scripts/validate_supa_correctness.py \
  --json results/spectralconv_correctness_biren.json
python tests/test_performance.py
python tests/run_all_tests.py
```

BIREN 性能结果在设备同步后测量。必选题固定配置
`B=4, C_in=32, C_out=64, modes=16` 的实测结果：

| 分辨率 | median ms | P90 ms | 峰值显存 MB | samples/s |
|---:|---:|---:|---:|---:|
| 64×64 | 91.104 | 97.488 | 124.6 | 43.9 |
| 128×128 | 103.867 | 115.187 | 473.1 | 38.5 |
| 256×256 | 293.657 | 300.174 | 1290.2 | 13.6 |

### Navier–Stokes 正式数据

本项目采用 FNO 论文公开的
`NavierStokes_V1e-5_N1200_T20.mat`：1200 个 64×64 样本、20 个时间点。
来源页是 NeuralOperator 组织的
[graph-pde 数据目录](https://github.com/neuraloperator/graph-pde#datasets)，
脚本使用可校验 SHA256 的公开镜像下载同名文件。

```bash
python scripts/prepare_ns_data.py
python -m pytest tests/test_prepare_ns_data.py
```

输出 `data/navier_stokes_64x64_n1200.npz`，取 `a` 为 `ω₀`、`u[..., 19]`
为 `ω_T`。原文件前 1000 个样本再固定划分为 900 train / 100 validation，
最后 200 个样本作为独立 test；不使用 test 指标选 checkpoint。metadata
JSON 记录官方来源、镜像、源文件/转换文件 SHA256、shape、时间索引和
原始 `1000/200` 边界。数据文件本身不提交 Git。

进阶题训练与评估入口：

```bash
# 不依赖正式数据的本地流程冒烟
python fno_ns/train_fno.py --smoke-test
python fno_ns/evaluate_fno.py --smoke-test

# 正式数据与 BIREN backend（数据路径按服务器实际位置替换）
python fno_ns/train_fno.py \
  --data data/navier_stokes_64x64_n1200.npz --backend supa --device supa
python fno_ns/evaluate_fno.py --checkpoint fno_ns/outputs/best_fno.pt \
  --data data/navier_stokes_64x64_n1200.npz --backend supa --device supa
```

正式训练完成 100 epoch，按 validation relative L2 选择 epoch 90：

- best validation relative L2：`0.6736211919784546`；
- epoch 100 train / validation relative L2：
  `0.6829287846883138 / 0.6736404609680176`；
- 200 个独立测试样本 relative L2：`0.6849088621139526`。

正式 FNO 评测使用 BIREN 单卡、64×64、batch size 16、`backend=supa`：
`38.461118329 samples/s`、`26.000284013 ms/sample`、
`157536.7407 grid_points/s`，峰值显存 `218.541015625 MB`。计时采用
20 次 warmup 和 100 次同步 forward repeat。

模型、训练日志和指标已回传至 `results/hardware/`：

- 模型：
  [best_fno.pt](results/hardware/fno_ns/outputs/ns64_supa/best_fno.pt)
  （SHA256
  `296c22f8f50a4ca97a66e431b8411d6749f798b88ca7d1caa77a8c149fd24b79`）
- 训练/测试指标：
  [train_metrics.jsonl](results/hardware/fno_ns/outputs/ns64_supa/train_metrics.jsonl)、
  [final_test_metrics.json](results/hardware/fno_ns/outputs/ns64_supa/final_test_metrics.json)
- 评估指标：
  [evaluation_metrics.json](results/hardware/fno_ns/outputs/ns64_supa_eval/evaluation_metrics.json)
- 环境、正确性与算子性能：
  [results/hardware/results](results/hardware/results)

已生成并逐张视觉核验 3 个独立测试样本的
Ground truth / Prediction / Absolute error 对比图，无乱码或元素重叠：

- [prediction_000.png](results/hardware/fno_ns/outputs/ns64_supa_eval/prediction_000.png)
- [prediction_001.png](results/hardware/fno_ns/outputs/ns64_supa_eval/prediction_001.png)
- [prediction_002.png](results/hardware/fno_ns/outputs/ns64_supa_eval/prediction_002.png)

## 验收标准

- 自定义算子与 PyTorch 参考实现的前向相对误差不超过 `1e-4`；
- 若提交 backward，加测输入与权重梯度相对误差；
- FNO 至少包含 4 层 Fourier Layer，并复用本项目自定义谱卷积；
- 报告 Navier–Stokes 公开数据集来源、64×64 数据版本、训练/测试划分与 relative L2；
- BIREN 单卡完成模型单次前向，并保留环境探针、运行命令和原始日志；
- `agent_logs/` 至少包含 5 段真实有效交互，覆盖赛题要求的至少 3 类场景。
- 本提交实际包含 7 段对话摘录和 8 份问题闭环日志；入口见
  [Agent 对话证据索引](agent_logs/chat_exports/README.md)。

## 提交前检查

1. 将探针输出、正确性结果和 benchmark 原始日志保存到 `results/`。
2. 更新 [results/REPORT.md](results/REPORT.md)，只加入可追溯的实测结果。
3. 核对图表坐标、单位、设备型号、batch size、warmup/repeat 和同步方式。
4. 根据 [agent_logs/README.md](agent_logs/README.md) 整理真实交互，不补写不存在的对话。
5. 更新 `skill.md` 的实现状态，并确认所有命令可从仓库根目录复现。

赛题原文整理见 [competition.md](competition.md)，技术方案调研见 [technology_survey.md](technology_survey.md)。
