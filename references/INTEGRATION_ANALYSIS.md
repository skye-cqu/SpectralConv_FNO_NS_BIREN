# 外部代码集成分析

> 分析 4 个开源项目的代码如何接入现有项目，给出具体集成策略。

---

## 1. 总览：现有项目状态 vs 外部参考

| 项目模块 | 现有代码 | 问题/缺失 | 外部参考源 | 集成方式 |
|---------|---------|----------|-----------|---------|
| `src/reference/SpectralConv2d` | 2D 专用，两权重 `weights1/weights2` | 代码重复（2D/3D 各一套）；einsum 依赖 | **DeepChem SpectralConv** | 用 `dims` 参数化统一 2D/3D |
| `src/reference/SpectralConv3d` | 3D 专用，四权重 | 同上 | 同上 | 合并为单一 `SpectralConvND` |
| `src/supa/` （待实现） | 空 | 无任何 SUPA kernel | **TurboFNO** 融合策略 + **fft-conv-pytorch** 复数乘 | 分级实现：Baseline → 分离 kernel → 融合 kernel |
| `tests/test_correctness.py` | 无相对误差对比 | 缺少定量 ground-truth | **DeepChem** 作为基线 | 创建独立参考实现，计算相对 L2 误差 |
| `fno_ns/` （待实现） | 空 | 无 FNO 模型 | **DeepChem FNO** FNOBlock | 组装 ≥4 层 FNO |

---

## 2. 逐项目集成方案

### 2.1 DeepChem SpectralConv → `src/reference/`

**对接点**：

| DeepChem 特性 | 接入方式 | 修改内容 |
|--------------|---------|---------|
| `modes` 自动处理 rfft 非对称维度 (`modes[-1]//2+1`) | 直接复用 | 替换 `SpectralConv2d.__init__` 中手动处理 W//2+1 的逻辑 |
| `torch.complex(real, imag)` 初始化权重 | 直接复用 | 替换 `nn.Parameter(torch.randn(..., dtype=torch.cfloat))` |
| `dims` 参数化 1D/2D/3D | **重构** | 将 `SpectralConv2d` + `SpectralConv3d` 合并为 `SpectralConv2d(dims=2)` 和 `SpectralConv3d(dims=3)` |

**修改后代码比现有代码减少约 40%**（约 30 行 → 约 18 行核心逻辑）。

```python
# 现有：2D 需要 weights1/weights2 + 正负频率分别处理
# 改为：统一 einsum("b i ..., i o ... -> b o ...")，modes tuple 自动处理各维度
```

### 2.2 TurboFNO 融合策略 → `src/supa/`

**对接点**：

| TurboFNO 策略 | 接入方式 | 在 SUPA 中的对应 |
|--------------|---------|-----------------|
| 内置截断的 FFT | **架构参考** | 在 SUPA kernel 中合并 `rfft2` 截断操作，减少一次 global write |
| FFT→CGEMM 共享内存传递 | **架构参考** | 利用 `__shared__` 或 SUPA 的本地数据共享机制 |
| thread block 沿隐藏维度遍历 | **架构参考** | thread block 调度策略映射 |
| GEMM tile 参数（64/64/8） | **参数参考** | 调优 SUPA 的 block/grid 大小 |

**实施路径**（分级融合，从易到难）：

```
阶段 0（当前）：纯 PyTorch, 无自定义 kernel
阶段 1（先做）：torch.extension 封装复数乘 kernel
  → 把 einsum 替换为手写复数矩阵乘 CGEMM
  → 参考 fft-conv-pytorch 的 real/imag 分离算法

阶段 2（加分）：FFT + 截断 + CGEMM 融合 kernel
  → 参考 TurboFNO Variant B 策略
  → 在 shared memory 中完成 FFT 输出→CGEMM 输入的数据传递

阶段 3（挑战）：FFT + CGEMM + iFFT 全融合
  → 参考 TurboFNO Variant D 策略
  → 单 kernel 内完成完整计算流程
```

### 2.3 fft-conv-pytorch complex_matmul → `src/supa/`

**对接点**：

```python
# 现有代码（依赖 torch.einsum，在 SUPA 上可能不支持）：
out = torch.einsum("bixy,ioxy->boxy", x_ft_trunc, weights)

# 替代方案（不依赖 einsum，仅用 matmul/add）：
def complex_matmul_fallback(x, w):
    """纯 matmul 实现的复数乘，可在任何后端运行"""
    # x: [B, C_in, M, M], w: [C_in, C_out, M, M]
    x = x.permute(1, 0, 2, 3)          # [C_in, B, M, M]
    w = w.unsqueeze(1)                  # [C_in, 1, C_out, M, M]
    real = x.real @ w.real - x.imag @ w.imag
    imag = x.imag @ w.real + x.real @ w.imag
    return real.sum(0).permute(1, 0, 2, 3)
```

**策略**：在 Python 层优先尝试 einsum，如果 SUPA 不支持则回退到该方案。

### 2.4 DeepChem FNOBlock → `fno_ns/`

**对接点**：

```python
# DeepChem FNOBlock 结构：SpectralConv + 1x1 Conv + ReLU
# 复用本项目 src/reference/ 中的 SpectralConv
# 不需修改，直接在 fno_ns/ 中 import 后组装

class FourierLayer(nn.Module):
    def __init__(self, width, modes1, modes2):
        super().__init__()
        self.spectral_conv = SpectralConv2d(width, width, modes1, modes2)
        self.conv = nn.Conv2d(width, width, 1)

    def forward(self, x):
        return F.gelu(self.spectral_conv(x) + self.conv(x))

# 然后 ≥4 层堆叠为 FNO 模型
```

---

## 3. 文件级别改动清单

### 立即可以做的（直接复用外部代码）

| 文件 | 操作 | 参考源 | 改动量 |
|------|------|-------|--------|
| `src/reference/spectral_conv2d.py` | 重构为 `SpectralConv2d` 保留兼容性，内部用 dims 参数化逻辑 | DeepChem | 小改（~5 行） |
| `tests/test_correctness.py` | 增加相对 L2 误差对比 | DeepChem 作为独立参考 | 新增 ~20 行 |
| `fno_ns/fno_model.py` | 用 FNOBlock 模式搭建 ≥4 层 FNO | DeepChem FNO | 新建 ~60 行 |

### 需要适配的（策略参考，不可直接复制）

| 文件 | 参考源 | 说明 |
|------|--------|------|
| `src/supa/spectral_conv_kernel.supa` | TurboFNO + fft-conv | 需手写 SUPA CGEMM kernel |
| `src/supa/setup.py` | Biren ModelZoo + DeepSpeed 适配 | 需参考 torch_supa_ext 构建模式 |

---

## 4. 风险与缓解

| 风险 | 概率 | 缓解措施 |
|------|------|---------|
| SUPA 不支持 `torch.einsum` | 中 | 用 `complex_matmul_fallback` 替代（来自 fft-conv-pytorch） |
| SUPA 不支持 `torch.fft.rfft2` | 中 | 确认 BIREN cuFFT 等价库支持；否则手写 SUPA FFT |
| TurboFNO 策略无法直接移植到 SUPA | 低 | 保留阶段 1（CGEMM kernel 封装）作为保底 |

---

## 5. 推荐执行顺序

```
Step 1: 重构 reference（DeepChem 参数化） + 增加相对 L2 测试
                               ↓
Step 2: torch.extension 封装复数乘 kernel（阶段 1，参考 fft-conv）
                               ↓
Step 3: 搭建 FNO 模型（fno_ns/，参考 DeepChem FNOBlock）
                               ↓
Step 4: 逐步融合优化（阶段 2→3，参考 TurboFNO）
                               ↓
Step 5: 填写 agent_logs + skill.md
```
