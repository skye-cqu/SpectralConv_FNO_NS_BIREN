# SpectralConv 与 FNO-NS 验证报告

## 结果状态

| 项目 | 状态 | 当前结论 |
|---|---|---|
| BIREN 环境与扩展加载 | 已实测 | Biren106M 单卡可加载自定义 extension |
| SpectralConv 前向 | 已实测 | relative L2 约 `2e-7`，通过 `1e-4` 门槛 |
| 输入及两组权重梯度 | 已实测 | relative L2 均约 `2e-7`，通过 `1e-4` 门槛 |
| 四层 FNO 严格 SUPA 冒烟 | 已实测 | 合成数据训练、checkpoint、评估链路成功 |
| 64/128/256 算子性能 | 已实测 | 报告 median、P90、吞吐和峰值显存 |
| 正式 Navier–Stokes 训练 | 已实测 | 100 epoch，独立 test relative L2 `0.6849088621139526` |
| 正式 FNO 性能 | 已实测 | `157536.7407 grid_points/s` |
| 预测/真值/误差图 | 已实测 | 3 张独立测试样本对比图已生成并视觉核验 |

“严格 SUPA”表示 FNO 的谱卷积强制使用自定义 extension backend；没有用
Python fallback 替代核心频域复数乘。合成数据冒烟只验证工程链路，不构成
公开数据集精度成绩。

## BIREN 实测环境

| 项目 | 实测值 |
|---|---|
| GPU | Biren106M，32512 MiB，单卡 |
| BR-SMI / Driver | 1.11.0 |
| SUPA | 1.11 |
| SDK | `birensupa-sdk 1.11.0.0.rc2` |
| Python | 3.10.12 |
| PyTorch | `2.9.0+cu128` |
| BIREN PyTorch 插件 | `torch_br 1.10.0.20900+br1xx` |
| 设备接口 | `import torch_br` 后使用 `torch.supa` |

环境没有独立 `torch_supa` 包，且 `torch.cuda.is_available()` 为 false。
PyTorch 版本字符串中的 `cu128` 不表示本次计算使用 NVIDIA CUDA。

## 算子实现与正确性

计算流程为实数输入 FFT、上下低频模态截断、自定义复数通道缩并、逆 FFT。
自定义 SUPA extension 分别实现 forward、输入梯度和权重梯度 kernel。

正式验收入口：

```bash
python scripts/probe_biren.py --require-extension
python scripts/validate_supa_correctness.py \
  --json results/spectralconv_correctness_biren.json
```

默认验收配置为 `B=2, C_in=3, C_out=5, H=W=64, modes=12,
dtype=float32, seed=20260731`。与 CPU PyTorch reference 对比，forward、
input gradient、`weights1` gradient、`weights2` gradient 的 relative L2
均约为 `2e-7`，低于赛题阈值 `1e-4`。

前向与三个梯度的 relative L2 均约为 `2e-7`；最终归档仍应保留服务器端
结构化 JSON，以提供每项未舍入的精确值。

## 算子性能

Biren106M 单卡，固定配置
`B=4, C_in=32, C_out=64, modes1=modes2=16`：

| 分辨率 | median ms | P90 ms | 峰值显存 MB | samples/s |
|---:|---:|---:|---:|---:|
| 64×64 | 91.104 | 97.488 | 124.6 | 43.9 |
| 128×128 | 103.867 | 115.187 | 473.1 | 38.5 |
| 256×256 | 293.657 | 300.174 | 1290.2 | 13.6 |

该表只报告自定义 SUPA backend 的实测绝对性能；没有提供同机 reference
延迟，因此不计算或声称加速比。

## 平台 FFT 排障

赛事环境 native `torch.fft.rfft2` 实测遗漏 H 维变换；其 native
`irfft2` 与该错误行为自洽，因此仅做设备内 roundtrip 会产生误判。与 CPU
逐元素对照时 native relative L2 约为 `1`。

SUPA 模块改用 `rfft(dim=-1) → fft(dim=-2)`，逆变换使用
`ifft(dim=-2) → irfft(dim=-1)`。该顺序 FFT 与 CPU 对照的 relative L2
在最终环境探针中实测为 `4.860923384338554e-8`。

## FNO-NS 正式训练与评估

模型包含 4 层 Fourier Layer，并在 `backend=supa` 时将每层绑定到自定义
extension。Biren106M 上先完成合成 16×16 数据的严格 SUPA 冒烟，随后在
公开 64×64 Navier–Stokes 数据上完成正式训练与独立测试。

初次硬件运行发现 PyTorch Adam 的 foreach 路径调用平台不支持的
`torch._foreach_lerp_`；改用项目的 SUPA 兼容逐参数优化器后复验成功。
合成数据 loss 未作为比赛精度指标。

正式数据准备脚本已固定 FNO 公开数据
`NavierStokes_V1e-5_N1200_T20.mat` 的 SHA256、64×64 shape、时间切片和
原顺序 `1000/200` 划分：

```bash
python scripts/prepare_ns_data.py
```

原文件前 1000 个样本固定拆分为 900 train / 100 validation；最后 200 个
样本作为独立 test。训练共 100 epoch，checkpoint 只依据 validation 指标
选择，不使用 test 指标参与选模。

| 指标 | 实测值 |
|---|---:|
| selected epoch | 90 |
| best validation relative L2 | 0.6736211919784546 |
| epoch 100 train relative L2 | 0.6829287846883138 |
| epoch 100 validation relative L2 | 0.6736404609680176 |
| independent test relative L2 | 0.6849088621139526 |

正式 evaluate 使用 `backend=supa`、Biren106M 单卡、64×64、batch size 16，
20 次 warmup 后执行 100 次同步 forward repeat：

| 指标 | 实测值 |
|---|---:|
| evaluate elapsed seconds | 41.600454420316964 |
| samples/s | 38.461118329 |
| ms/sample | 26.000284013 |
| grid_points/s | 157536.7407 |
| peak memory MB | 218.541015625 |

上述性能对应已训练 checkpoint 的前向评测。模型及指标位于：

- `results/hardware/fno_ns/outputs/ns64_supa/best_fno.pt`
- `results/hardware/fno_ns/outputs/ns64_supa/train_metrics.jsonl`
- `results/hardware/fno_ns/outputs/ns64_supa/final_test_metrics.json`
- `results/hardware/fno_ns/outputs/ns64_supa_eval/evaluation_metrics.json`
- `results/hardware/results/` 下的环境、正确性和三档算子性能 JSON

## FNO 预测可视化

已生成并逐张视觉检查以下 3 个独立测试样本。每张图均包含 Ground truth、
Prediction 和 Absolute error，未发现中文乱码、标签重叠或图像元素遮挡：

1. [prediction_000.png](hardware/fno_ns/outputs/ns64_supa_eval/prediction_000.png)
2. [prediction_001.png](hardware/fno_ns/outputs/ns64_supa_eval/prediction_001.png)
3. [prediction_002.png](hardware/fno_ns/outputs/ns64_supa_eval/prediction_002.png)

## 提交证据完整性

- 服务器环境、正确性、三档 benchmark、100-epoch 训练和独立评估产物均已
  回传并保存到 `results/hardware/`。
- `agent_logs/chat_exports/` 包含 7 段真实、已脱敏的 Codex 主任务及并行
  Agent 对话摘录，覆盖赛题指定的全部 6 类场景；`agent_logs/01` 至 `08`
  另记录问题—行动—证据—结论闭环。
- 正式 checkpoint SHA256：
  `296c22f8f50a4ca97a66e431b8411d6749f798b88ca7d1caa77a8c149fd24b79`。
- 服务器原始结果归档 SHA256：
  `1594128a967578769e5e6fa8fe21713d41a1cd0f59df55df473c518de6e9845`。
- 本地最终复验：`pytest` 为 `33 passed, 2 skipped`，统一测试入口为
  `5/5` 通过，`compileall` 与 `git diff --check` 均通过。
