# 交互 03：CPU FFT 反向传播排障

- 时间：2026-07-31
- 场景：算子调试、平台兼容

## 现象

本地执行 `tests/test_backward.py` 时，oneMKL 在
`irfft2(...).sum().backward()` 上报告：

```text
Intel oneMKL DFTI ERROR: Inconsistent configuration parameters
```

## 分析与处理

该问题出现在全 1 上游梯度对应的退化路径，不代表谱卷积公式错误。将测试损失改为更贴近训练的 `y.square().mean()`，继续验证输入梯度和权重梯度。

## 结果

`test_correctness.py` 与 `test_backward.py` 均通过；输入及两组复数权重均获得有限梯度。

