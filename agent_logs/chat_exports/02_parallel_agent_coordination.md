# 交互 02：多 Agent 并行开发拆分

- 日期：2026-07-31
- 来源：Codex 主任务与并行 Agent 活动记录
- 场景：算子、模型和提交材料并行协作

## 对话摘录

**用户**

> 我需要你边创造边在我给你提供的这个 codex 终端窗口进行硬件设施验证，
> 通过多个 agent 协同方式快速推进，多个可并行任务同时进行，代码以及软件
> 编写好之后，在我提供给你的硬件服务器上进行验证，其他 agent 接着推进
> 后续任务。

**主 Agent**

> 我会按“算子、FNO、工程材料”三条线并行推进，我负责集成、测试与服务器
> 验证。每条线写入互不重叠的模块；完成一批就立即合并检查，并在 BIREN
> 端做能力探针、编译和数值/性能验证，再把结果反馈给后续 Agent 继续优化。

## 实际 Agent 分工

- `supa_operator`：自定义复数乘 forward/backward、C++ 绑定、SUPA kernel、
  严格 backend 与顺序 FFT 适配；
- `fno_ns`：数据层、四层 FNO、训练、评估、relative L2 与可视化；
- `submission_docs`：依赖、探针、报告、`skill.md` 与提交证据审计；
- `biren_optimizer`：定位 `LerpScalar` 不支持并实现/验证 `SupaAdam`；
- 主 Agent：集成、BIREN 终端验证、正式训练监控、结果回传和最终归档。

## 代码落点

- 算子：[src/supa](../../src/supa)
- 模型：[fno_ns](../../fno_ns)
- 硬件证据：[results/hardware](../../results/hardware)
- 汇总日志：[agent_logs](..)

该分工来自真实子 Agent 启动与回传记录；没有将事后文件分类伪装成并行开发。

