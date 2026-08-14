# 交互 05：官方 SUPA 扩展示例分析

- 时间：2026-07-31
- 场景：kernel 设计、BIREN 扩展构建

## 参考来源

服务器预装示例：

```text
/workspace/ai4s/gemv/torch_extension/
```

## 确认的构建范式

1. `.su` 文件使用类似 CUDA 的 `__global__`、block/thread 索引和 launch 语法。
2. C++ binding 包含 `torch/extension.h` 与 `supa.h`，通过 pybind 暴露 Python 接口。
3. `brcc` 使用 `--supa-gpu-arch=br100` 编译和链接 SUPA 对象。
4. 链接 `torch`、`torch_br` 与 `supa-runtime`。
5. 官方示例将 `nullptr` 作为默认 stream 传给 launch wrapper。

## 决策

自定义复数乘以 interleaved float32 实部/虚部实现；forward、输入梯度和权重梯度分别使用独立 kernel，避免猜测不存在的 `torch_supa` 构建接口。

