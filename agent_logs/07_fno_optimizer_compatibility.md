# 07 — FNO 优化器的 BIREN 兼容性排查

## 场景

在 Biren106M 上完成 SUPA 扩展编译、环境探针和算子前后向测试后，继续运行
四层 FNO 的合成数据训练冒烟：

```bash
python3 fno_ns/train_fno.py --smoke-test --backend supa --device supa \
  --output-dir fno_ns/outputs/supa_smoke
```

## 观察

模型已经完成前向和反向，失败发生在第一次 `Adam.step()`。`torch_br`
日志明确报告：

```text
no valid backend was found for op: LerpScalar
torch._foreach_lerp_
```

这说明问题不在自定义 SpectralConv kernel，也不在 FFT 或梯度计算，而在
PyTorch Adam 使用的 `lerp_` 更新原语。

## Agent 判断与修改

Agent 先根据堆栈为 Adam 显式设置 `foreach=False`，但硬件复验发现
PyTorch 2.9 的单张量 Adam 仍调用 `Tensor.lerp_`，因此该尝试没有解决
问题。最终实现 `SupaAdam`：使用平台已验证的
`mul_`、`add_`、`addcmul_`、`sqrt`、`addcdiv_` 原语完成等价更新；
complex64 参数通过实/虚视图更新。没有回退 SpectralConv backend，也没有
绕过真实 SUPA 前向或反向。

## 复验结果

`SupaAdam` 与官方 Adam 的 CPU 实数及 complex64 多步更新数值对照通过，
complex64 差异为 `0.0`。修订源码同步到同一 Biren106M 后，严格 extension
backend 的四层 FNO 合成数据训练冒烟成功完成，包括前向、反向、逐参数
更新、checkpoint 保存和评估；随后正式 100 epoch 训练也成功完成。

该复验没有使用正式 Navier–Stokes 数据，因此不报告其中的 relative L2
作为比赛精度，也不把 16×16 合成数据吞吐作为正式性能。
