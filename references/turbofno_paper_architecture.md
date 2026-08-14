# TurboFNO Kernel Architecture（提取自 arXiv:2504.11681）

## 核心创新：全融合 FFT-GEMM-iFFT

传统 FNO kernel 调用链路：
```
rfft2 → 截断 → 零填充 → CGEMM → irfft2 （多个独立 kernel，多次 global mem 读写）
```

TurboFNO 全融合 kernel：
```
[FFT → 截断 → CGEMM → iFFT] 单 kernel，全部在 shared memory 内完成
```

## 三步优化策略

### 1. 内置截断的 FFT kernel
- FFT 输出时只写保留的低频分量到 global memory（75% 写入减少）
- GPU-side FFT 剪枝：跳过截断频段的冗余蝶形运算
- 输入零填充也集成在 FFT 内，无额外 kernel

### 2. GEMM 兼容的 FFT 变体
- 每个 thread block 沿 HiddenDim 方向取一片数据做 FFT
- FFT 输出直接写入 shared memory，格式匹配 CGEMM 的 A 操作数
- 同一个 thread block 继续做 CGEMM 的 MAC 运算

```
thread block (M, N, K):
  for k in range(K):
    FFT(A_slice)  →  shared_mem  // 替代 A 的 global load
    C += A @ B                     // CGEMM
  shared_mem → iFFT               // 替代 C 的 global store
```

### 3. 共享内存 swizzling
- FFT→GEMM：解决 FFT 输出布局与 GEMM 输入布局不匹配问题
- GEMM→iFFT：解决 GEMM 输出与 iFFT 输入布局不匹配问题
- Bank utilization 从 25% 提升到 100%

## Kernel 配置参数

```c
// 从 TurboFNO.h 提取
#define THREADBLOCK_M 64  // GEMM M 维度 tile 大小
#define THREADBLOCK_N 64  // GEMM N 维度 tile 大小
#define THREADBLOCK_K 8   // GEMM K 维度 tile 大小
#define WARP_M 32
#define WARP_N 16
#define THREAD_M 4
#define THREAD_N 4
```

## 限制
- 仅支持 C2C FFT（非 R2C）
- 截断后 size ≤ 64

## 对我们项目的意义
1. **架构参考**：在 SUPA 中实现分级融合（A/B/C/D 四阶段渐进）
2. **共享内存策略**：SUPA 的 SUPA_SHARED_MEM 使用方式参考
3. **thread block 调度**："沿隐藏维度遍历"的策略直接适用于我们的复数乘
