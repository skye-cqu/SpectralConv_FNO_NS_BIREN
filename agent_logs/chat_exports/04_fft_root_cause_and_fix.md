# 交互 04：BIREN native rfft2 根因定位

- 日期：2026-07-31
- 来源：Codex 主任务真实回合、BIREN 探针输出
- 场景：性能/正确性瓶颈定位、BIREN GPU 平台适配

## 对话摘录

**Agent**

> 完整 SpectralConv 验收暴露了一个新的真实硬件问题：扩展复数乘单测通过，
> 但加入 `torch.supa` FFT 后整体误差接近 1，说明问题已缩小到 CPU 与
> SUPA 的 FFT 频谱约定/实现差异，而不是 kernel 乘法。

进一步诊断：

> 已定位根因：BIREN 当前软件栈里的 `torch.fft.rfft2` 实际没有完成二维
> 变换——它的 DC 值很像只对第一行做了最后一维 FFT。这解释了
> “round-trip 能过但跨设备正确性失败”。

修复决定：

> 把 SUPA 的二维 FFT 改成“宽度 `rfft` + 高度 `fft`”，逆变换相应改成
> “高度 `ifft` + 宽度 `irfft`”。

## 实测闭环

- native `rfft2` 相对 CPU：`1.027866201678142`
- 顺序 `rfft(W) → fft(H)` 相对 CPU：`4.860923384338554e-8`
- 最终完整 SpectralConv 前向和三项梯度：均约 `2e-7`

## 证据

- 适配实现：[fft.py](../../src/supa/fft.py)
- 探针源码：[probe_biren.py](../../scripts/probe_biren.py)
- 原始探针 JSON：
  [biren_environment.json](../../results/hardware/results/biren_environment.json)
- 完整正确性 JSON：
  [spectralconv_correctness_biren.json](../../results/hardware/results/spectralconv_correctness_biren.json)

这段记录保留了“早期性能数字因 FFT 错误而作废、修复后全部重测”的失败
与迭代过程。

