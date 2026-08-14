# 交互 06：集成审查与修复

- 时间：2026-07-31
- 场景：正确性、性能与数据流程审查

## 审查发现

1. FNO `backend=supa` 最初未强制 `backend=extension`。
2. FNO runtime 最初未同步 `torch.supa`。
3. 统一测试入口漏跑 SUPA 接口和 FNO smoke。
4. 训练过程按测试集指标选择 checkpoint，存在评估偏差。
5. 性能脚本缺少 median、P90 和结构化输出。

## 已执行修复

- FNO 的 `supa` factory 显式绑定 extension。
- FNO runtime 复用统一 SUPA 设备、同步和显存接口。
- `run_all_tests.py` 纳入 SUPA 接口与 FNO smoke。
- 算子 benchmark 增加 mean、median、P90、设备、backend、warmup、repeat 和 JSON 输出。
- 数据划分、独立测试集和 MATLAB v7.3 支持交由 FNO Agent 继续修复。

