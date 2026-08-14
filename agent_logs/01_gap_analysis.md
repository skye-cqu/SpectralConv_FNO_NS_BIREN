# 交互 01：赛题与仓库缺口分析

- 时间：2026-07-31（Asia/Shanghai；对应 UTC 2026-07-30）
- 场景：模型架构选型、提交风险分析
- 用户目标：阅读赛题并根据现有项目制定完成方案。

## Agent 操作

读取 `competition.md`、现有 reference、测试、外部调研和 Git 历史，对照评分项检查仓库。

## 关键发现

1. 已有 2D/3D PyTorch reference 和测试骨架。
2. `src/supa/`、`fno_ns/`、`skill.md`、正式报告和 Agent 日志尚未形成。
3. 原正确性测试只验证 shape/有限值，没有自定义实现与 reference 的相对误差。
4. 原性能脚本混用了 `perf_counter()` 和 `time.time()`。
5. `.venv` 存在，但最初无法导入 PyTorch。

## 决策

三天版本优先完成 2D 自定义算子、反向、四层 FNO-NS、BIREN 实测和提交材料；3D 自定义 kernel 与全融合优化延后。
