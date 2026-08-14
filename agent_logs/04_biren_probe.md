# 交互 04：BIREN 硬件与关键 API 探针

- 时间：2026-07-31
- 场景：BIREN GPU 平台适配

## 探针内容

检查 `brsw`、`brsmi`、Python、PyTorch、`torch_br`、设备命名空间、complex64 FFT 和 backward。

## 实测环境

- GPU：Biren106M
- 显存：32512 MiB
- Driver / SUPA：1.11.0 / 1.11
- SDK：`1.11.0.0.rc2`
- Python：3.10.12
- PyTorch：2.9.0+cu128
- `torch_br`：`1.10.0.20900+br1xx`
- `torch_supa`：未安装，实际接口为 `import torch_br` 后的 `torch.supa`

## 结果

`torch.supa.is_available()` 为真，检测到一张 Biren106M；SUPA 上
complex64 张量和反向传播可运行。后续与 CPU 逐元素对照发现 native
`rfft2` 遗漏 H 维变换，单纯的设备内 `rfft2/irfft2` roundtrip 会掩盖该
问题。最终 SUPA 模块改用顺序一维 FFT，CPU 对照 relative L2 为
`4.860923384338554e-8`。
