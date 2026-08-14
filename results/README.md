# 结果目录

本目录只提交可追溯的精选结果。

- `REPORT.md`：已验证结果、证据状态与未完成项；
- `figures/`：预测、真值、误差图和性能图；
- `*.csv`：结构化 benchmark 结果；
- `*.json`：环境探针或小型指标摘要；
- `*.log`：必要的运行日志。

临时图片放在 `results/tmp/`，大体量原始输出放在 `results/raw/`，两者默认不提交。模型权重和数据集默认不进入 Git；最终报告中提供来源、哈希及可复现获取方式。

只有 `REPORT.md` 中明确标为“已实测”的条目可作为当前成绩。发布前检查：

1. 每个数字都有对应命令和原始日志；
2. BIREN 结果附 `biren_environment.json`；
3. 表格注明单位、batch、dtype、shape、modes、warmup 和 repeat；
4. CPU/CUDA 结果不得标为 BIREN；
5. 图片无中文乱码、元素遮挡或不可辨认色标；
6. 未完成项保持明确标注，不用估算值填充正式成绩。
