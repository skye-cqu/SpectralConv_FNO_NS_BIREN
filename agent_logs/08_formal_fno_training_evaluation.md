# 08 — 正式 FNO 训练、独立测试与性能评估

## 场景

- 时间：2026-07-31
- 平台：Biren106M 单卡，`torch.supa`
- 场景类别：模型训练、结果验证、性能评估

在严格 extension backend 的合成数据冒烟成功后，继续使用公开 64×64
Navier–Stokes 数据完成正式训练。谱卷积没有切换到 Python fallback。

## 数据与选模

1200 个样本按原始顺序保留独立测试边界：

- 前 1000 个样本内部固定拆分为 900 train / 100 validation；
- 最后 200 个样本只用于 independent test；
- checkpoint 依据 validation relative L2 选择，不使用 test 指标选模。

训练 100 epoch，选中 epoch 90。best validation relative L2 为
`0.6736211919784546`。epoch 100 的 train / validation relative L2 分别为
`0.6829287846883138` 和 `0.6736404609680176`。

## 独立测试结果

选中 checkpoint 在 200 个独立测试样本上的 relative L2 为：

```text
0.6849088621139526
```

该值来自正式数据，不是合成冒烟指标。

## BIREN 推理评估

评测配置为 `backend=supa`、64×64、batch size 16；计时采用 20 次
warmup 和 100 次同步 forward repeat：

| 指标 | 实测值 |
|---|---:|
| evaluate elapsed seconds | 41.600454420316964 |
| samples/s | 38.461118329 |
| ms/sample | 26.000284013 |
| grid_points/s | 157536.7407 |
| peak memory MB | 218.541015625 |

## 结论与边界

正式训练、validation 选模、独立测试和 BIREN 前向性能链路已完成。模型、
训练指标、评估指标和硬件 JSON 已回传到 `results/hardware/`。
正式 checkpoint SHA256 为
`296c22f8f50a4ca97a66e431b8411d6749f798b88ca7d1caa77a8c149fd24b79`。

同时生成并逐张视觉核验 3 个独立测试样本的 Ground truth / Prediction /
Absolute error 图，未发现乱码或元素重叠：

- `results/hardware/fno_ns/outputs/ns64_supa_eval/prediction_000.png`
- `results/hardware/fno_ns/outputs/ns64_supa_eval/prediction_001.png`
- `results/hardware/fno_ns/outputs/ns64_supa_eval/prediction_002.png`
