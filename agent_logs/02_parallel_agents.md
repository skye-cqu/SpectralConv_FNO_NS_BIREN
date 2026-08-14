# 交互 02：多 Agent 并行开发

- 时间：2026-07-31
- 场景：算子设计、模型架构、工程化
- 用户要求：多个 Agent 协同并行推进，完成后在 BIREN 服务器验证。

## 并行分工

1. `supa_operator`：实现 backend、扩展 ABI、`SupaSpectralConv2d` 和专属测试。
2. `fno_ns`：实现数据、FNO2d、训练、评估、可视化和冒烟测试。
3. `submission_docs`：实现 README、依赖说明、环境探针、报告和日志模板。
4. 主 Agent：修正 reference/性能测试、集成、硬件探针和验收。

## 阶段结果

- SUPA Python fallback 与 extension 严格加载接口完成。
- FNO-NS 合成数据训练、checkpoint 和评估链路完成。
- README、`skill.md`、依赖说明及报告模板完成。
- 本地 reference、SUPA 接口和 FNO 测试形成并行闭环。

