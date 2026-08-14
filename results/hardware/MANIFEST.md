# BIREN 硬件结果清单

本目录来自服务器原始归档
`spectralconv_hardware_results_final.tar.gz`，归档 SHA256 为：

```text
1594128a967578769e5e6fa8fe21713d41a1cd0f59df55df473c518de6e9845
```

提交前仅对 `biren_environment.json` 的实例 hostname 和瞬时进程 PID 做了
隐私脱敏，替换为 `[REDACTED_INSTANCE]` / `[REDACTED]`；设备、软件版本、
探针结果及全部实验数值未修改。

关键产物：

- `results/biren_environment.json`：Biren106M、驱动、SUPA、PyTorch 与 FFT
  探针；
- `results/spectralconv_correctness_biren.json`：前向、输入梯度及两组权重
  梯度正确性；
- `results/spectralconv_extension_benchmark.json`：64/128/256 三档算子性能；
- `fno_ns/outputs/ns64_supa/train_metrics.jsonl`：100 个 epoch 的正式训练记录；
- `fno_ns/outputs/ns64_supa/final_test_metrics.json`：validation 选模与独立测试
  结果；
- `fno_ns/outputs/ns64_supa_eval/evaluation_metrics.json`：20 次 warmup、100 次
  repeat 的 BIREN 前向性能；
- `fno_ns/outputs/ns64_supa_eval/prediction_000.png` 至
  `prediction_002.png`：预测、真值及绝对误差空间分布；
- `fno_ns/outputs/ns64_supa/best_fno.pt`：正式 checkpoint，SHA256 为
  `296c22f8f50a4ca97a66e431b8411d6749f798b88ca7d1caa77a8c149fd24b79`。

公开数据集没有重复收录在结果目录中；来源、原始文件及转换文件 SHA256
见 `data/navier_stokes_64x64_n1200.metadata.json`。
