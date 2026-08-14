# 交互 05：FNO 优化器兼容性排障

- 日期：2026-07-31
- 来源：Codex 主任务与 `biren_optimizer` 并行 Agent
- 场景：模型训练调试、BIREN GPU 平台适配

## 对话摘录

**Agent**

> 已定位到精确兼容点：FNO 的前向和反向都已经跑过，失败发生在
> `Adam.step()` 默认启用的多张量 `_foreach_lerp_`；当前 `torch_br`
> 不支持 `LerpScalar`。

关闭 foreach 后再次失败：

> 即使关闭 foreach，PyTorch 2.9 的单张量 Adam 内部仍调用
> `Tensor.lerp_`，所以官方 Adam 在当前 `torch_br` 上不可用；
> 自定义 SpectralConv 前后向仍然正常。

**Agent 决定**

> 使用 `mul_ + add_ + addcmul_ + sqrt + addcdiv_` 实现不依赖
> `lerp_` 的等价 Adam，并对 complex64 参数使用实/虚视图。

复验结果：

> 基础 Adam 原语和复数参数路径均在 Biren106M 通过；本地与官方 Adam 的
> complex64 多步更新差异为 `0.0`。

## 证据

- 优化器实现：[optim.py](../../fno_ns/optim.py)
- 数值对照测试：[test_optim.py](../../fno_ns/test_optim.py)
- 原始阶段日志：[07_fno_optimizer_compatibility.md](../07_fno_optimizer_compatibility.md)
- 正式 100 epoch 训练日志：
  [train_console.log](../../results/hardware/fno_ns/outputs/ns64_supa/train_console.log)

