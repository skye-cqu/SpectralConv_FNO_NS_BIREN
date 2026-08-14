# 交互 06：正式数据、训练、评估与可视化

- 日期：2026-07-31
- 来源：Codex 主任务真实回合、BIREN 训练输出
- 场景：数据预处理、模型超参、结果分析与可视化

## 对话摘录

**用户**

> 需要 huggingface 的什么数据，如果只是网络问题，我可以手动下载下来，
> 放到指定位置。

用户随后确认：

> 已放好

**Agent**

> 原文件校验通过：大小 `412,877,192` 字节，SHA256 与项目锁定值完全
> 一致。现在开始解析形状并转换为只包含 `ω₀ → ω(T)` 的训练数据。

> 数据转换也已通过：生成的训练文件只有 `36.05 MB`，SHA256 为
> `cd52e739e6bde4f7d5caf8694502e2235a0a24f6fdba656d30223f9134787fd2`。

一轮硬件校准后：

> 完整一轮仅 `65.50 秒`，没有 OOM。按此速度，100 轮约 109 分钟，
> 适合一次性完成正式训练。

训练完成后：

> 第 90 轮最佳验证 relative L2 为 `0.673621`；第 100 轮训练/验证为
> `0.682929 / 0.673640`；独立测试为 `0.684909`。

独立评估后：

> 推理 `38.46 samples/s`、`26.00 ms/sample`、峰值显存 `218.54 MB`。

## 证据

- 数据元信息：[metadata.json](../../results/hardware/data/navier_stokes_64x64_n1200.metadata.json)
- 100 epoch 指标：[train_metrics.jsonl](../../results/hardware/fno_ns/outputs/ns64_supa/train_metrics.jsonl)
- 独立测试：[final_test_metrics.json](../../results/hardware/fno_ns/outputs/ns64_supa/final_test_metrics.json)
- 推理性能：[evaluation_metrics.json](../../results/hardware/fno_ns/outputs/ns64_supa_eval/evaluation_metrics.json)
- 预测/真值/误差图：
  [000](../../results/hardware/fno_ns/outputs/ns64_supa_eval/prediction_000.png)、
  [001](../../results/hardware/fno_ns/outputs/ns64_supa_eval/prediction_001.png)、
  [002](../../results/hardware/fno_ns/outputs/ns64_supa_eval/prediction_002.png)

三张图已逐张视觉检查，无乱码、标签重叠或图像元素遮挡。

