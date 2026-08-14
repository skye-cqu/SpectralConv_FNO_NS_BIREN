# 外部可复用参考代码

本目录存放从开源项目提取的相关源码片段，用于指导 SUPA kernel 实现和参考实现优化。

| 来源 | 文件 | 核心价值 |
|------|------|---------|
| TurboFNO | `turbofno_paper_architecture.md` | FFT-GEMM-iFFT 全融合策略 + 共享内存 swizzle |
| DeepChem | `deepchem_spectral_conv.py` | 通用 n 维 SpectralConv（dim 参数化） |
| DeepChem | `deepchem_fno.py` | FNOBlock + FNO 完整模型 |
| fft-conv-pytorch | `fft_conv_complex_matmul.py` | 复数乘 real/imag 分离实现（逃避 einsum 依赖） |
| TurboFNO | `turbofno_gemm_config.h` | GEMM tile 配置和 kernel 参数 |
