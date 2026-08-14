# SpectralConv-FNO-BIREN Skill

## 功能

本 Skill 用于开发、验证和评测 BIREN GPU 上的二维 Spectral Convolution，并将其复用于四层以上 Fourier Layer 的 FNO-NS 模型。

核心能力：

1. 运行 PyTorch SpectralConv2d 参考实现；
2. 调用 SUPA / `torch.extension` 自定义频域复数乘核心；
3. 比较自定义实现与参考实现的前向及反向相对误差；
4. 在 64、128、256 分辨率下测量延迟、吞吐和峰值显存；
5. 训练或评估 64×64 Navier–Stokes FNO，并生成 relative L2 与误差空间图；
6. 采集 BIREN 硬件、软件栈和关键 API 支持情况。

## 前置条件

- 从项目根目录运行所有命令；
- 本地调试优先使用现有 `.venv`；
- 正式结果必须来自已导入 `torch_br` 并注册 `torch.supa` 的 BIREN
  单卡环境；本次赛事镜像不要求独立 `torch_supa` 包；
- BIREN 环境不得被 PyPI 标准版 `torch` 覆盖；
- Navier–Stokes 数据集、划分与 checkpoint 必须记录来源和哈希。

## 调用入口

### 1. 环境探针

```bash
python scripts/probe_biren.py
python scripts/probe_biren.py --json results/biren_environment.json
python scripts/probe_biren.py --require-extension
```

平台成功条件：探针识别 SUPA 设备，并通过张量创建、complex64、
`rfft2/irfft2` 和 backward 冒烟测试。构建后正式验收还必须使用
`--require-extension` 确认项目扩展可加载。

### 2. 算子正确性与反向

```bash
python tests/test_correctness.py
python tests/test_backward.py
python -m pytest tests/test_supa_operator.py
python scripts/validate_supa_correctness.py \
  --json results/spectralconv_correctness_biren.json
```

最后一条命令是正式 BIREN 验收入口，强制使用 extension backend，并把前向、
输入梯度、`weights1` 和 `weights2` 梯度的误差写入 JSON。验收条件：所有
relative L2 均不超过 `1e-4`。

### 3. 算子性能

```bash
python tests/test_performance.py
```

报告条件：注明设备、dtype、`B/C_in/C_out/modes`、分辨率、warmup、repeat、同步方式、median/P90 和峰值显存，不把 CPU fallback 结果标记为 BIREN。

### 4. 完整回归

```bash
python tests/run_all_tests.py
```

### 5. FNO-NS

```bash
python fno_ns/train_fno.py --smoke-test
python fno_ns/evaluate_fno.py --smoke-test

python scripts/prepare_ns_data.py
python fno_ns/train_fno.py \
  --data data/navier_stokes_64x64_n1200.npz --backend supa --device supa
python fno_ns/evaluate_fno.py --checkpoint fno_ns/outputs/best_fno.pt \
  --data data/navier_stokes_64x64_n1200.npz --backend supa --device supa
```

最终模型须复用自定义 SpectralConv，包含至少 4 层 Fourier Layer。正式性能口径为 BIREN 单卡、64×64、batch size 16，主指标 `grid_points/s`，辅指标 `samples/s`、`ms/sample` 和峰值显存。

## 输入与输出约定

- SpectralConv 输入：实数浮点张量 `[B, C_in, H, W]`；
- SpectralConv 输出：实数浮点张量 `[B, C_out, H, W]`；
- 要求 `2 * modes1 <= H` 且 `modes2 <= W // 2 + 1`；
- FNO-NS 输入/目标：64×64 涡度场，具体通道布局以数据加载器说明为准；
- 报告输出：环境 JSON、测试日志、CSV 性能表、Markdown 报告和精选 PNG/PDF 图表。

## 失败处理

- SUPA 扩展不可用时应明确报错；参考 backend 仅用于开发诊断，不得静默冒充自定义实现；
- FFT、complex64 或 backward 任一能力失败时，保留完整探针错误和软件版本；
- 输出出现 NaN/Inf、shape 不一致或 relative L2 超阈值时停止性能结论，先完成正确性定位；
- 缺少数据或 checkpoint 时只运行合成数据冒烟，并在报告中标注“未完成正式精度评测”。

## 提交产物

- 源码、构建脚本和完整运行命令；
- 正确性、反向、三档性能与 FNO 正式训练/评估记录；
- `results/hardware/` 下的 checkpoint、训练/评估指标及 3 张
  Ground truth / Prediction / Absolute error 对比图；
- 至少 5 段真实 Agent 交互，覆盖至少 3 类赛题指定场景；
- 本作品实际提交 7 段已脱敏真实对话摘录和 8 份问题闭环日志，覆盖赛题
  指定的全部 6 类场景；
- 所有结果可追溯到 Git commit、环境探针和原始运行日志。

## 当前状态

- 已完成：Biren106M 上 extension 构建与加载；SpectralConv 前向、输入梯度和
  两组权重梯度相对误差均约 `2e-7`，通过 `1e-4` 门槛。
- 已完成：`64/128/256` 自定义算子 BIREN 性能；median 分别为
  `91.104 / 103.867 / 293.657 ms`，峰值显存分别为
  `124.6 / 473.1 / 1290.2 MB`。
- 已完成：四层 FNO 使用严格 extension backend 完成正式数据 100 epoch
  训练；900 train / 100 validation / 200 independent test，选中 epoch 90，
  测试 relative L2 为 `0.6849088621139526`。
- 已完成：FNO BIREN 单卡、64×64、batch size 16 正式评测，
  `157536.7407 grid_points/s`、`38.461118329 samples/s`、
  `26.000284013 ms/sample`，峰值显存 `218.541015625 MB`；计时采用
  20 次 warmup 和 100 次同步 forward repeat。
- 已完成：生成并视觉核验
  `results/hardware/fno_ns/outputs/ns64_supa_eval/prediction_000.png`、
  `prediction_001.png`、`prediction_002.png`，每张均包含 Ground truth、
  Prediction 和 Absolute error，未发现乱码或元素重叠。
- 正式模型及指标统一保存在 `results/hardware/`。
- 正式 checkpoint SHA256 为
  `296c22f8f50a4ca97a66e431b8411d6749f798b88ca7d1caa77a8c149fd24b79`。
- 已知平台限制：赛事环境 native `rfft2` 遗漏 H 维变换，SUPA 路径使用
  `rfft → fft` 和 `ifft → irfft` 顺序实现规避。
