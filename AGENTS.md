# AGENTS.md — 项目惯例与开发指南

## 项目概览

2026 飞翔杯 · 赛道五：模型与算子开发。
必选题：Spectral Convolution 算子开发（SUPA/torch.extension 实现）。
进阶题 C：FNO 求解 Navier-Stokes 方程（复用必选题算子）。

## 项目结构

```
SpectralConv/
├── src/
│   ├── reference/     # PyTorch 参考实现（正确性验证 baseline）
│   ├── supa/          # SUPA kernel + torch.extension 封装
│   └── utils/         # benchmark、可视化等工具
├── tests/             # 正确性验证 + 性能测试
├── fno_ns/            # 进阶题 C：FNO-NS
├── results/           # 报告与图表
├── agent_logs/        # Agent 开发日志（必须提交）
├── skill.md           # 功能描述入口（必须提交）
├── AGENTS.md          # 本文件
└── .gitignore
```

## 开发环境

- 硬件：首选壁仞 BIREN GPU；开发阶段可用 CPU/CUDA
- 语言：Python（模型层）、SUPA/C++（算子 kernel 层）
- 框架：壁仞 br_pytorch（`torch_br`/`torch_supa`）
- 辅助工具：Claude Code（本 Agent）

## 代码风格

- Python：遵循 PEP 8，不使用分号，不使用尾随逗号
- 导入顺序：标准库 → 第三方库（torch等）→ 本地模块
- 类型标注：函数签名必须包含类型注解
- 注释：关键数学公式和维度变换处加 docstring；不写无关注释
- 命名：类 PascalCase，函数/变量 snake_case，常量 UPPER_CASE

## 验证命令

```bash
# 正确性验证（PyTorch 参考 vs 自定义实现）
python tests/test_correctness.py

# 性能测试（64/128/256 多分辨率）
python tests/test_performance.py

# 反向传播验证
python tests/test_backward.py

# 完整运行（全部测试）
python tests/run_all_tests.py
```

## 进阶 C 训练

```bash
python fno_ns/train_fno.py
python fno_ns/evaluate_fno.py
```

## 关键依赖

- `torch_br` / `torch_supa` — BIREN GPU 设备插件
- `torch.fft.rfft2 / irfft2` — FFT 操作族（通过 BIREN FFT 等价库映射）
- `torch.einsum` — 复数张量缩并
- `neuraloperator`（可选参考）— NeuralOperator 库用于对比

## 提交规范

- 提交前运行全部测试，确保正确性
- 每个 Agent 交互记录存入 `agent_logs/`，至少 5 段有效记录
- `skill.md` 在提交前必须更新为最终版本
