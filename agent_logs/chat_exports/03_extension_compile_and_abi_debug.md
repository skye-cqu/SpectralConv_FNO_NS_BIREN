# 交互 03：SUPA 扩展编译与 ABI 路径排障

- 日期：2026-07-31
- 来源：Codex 主任务真实回合、BIREN 终端输出
- 场景：kernel 调试、BIREN GPU 平台适配

## 对话摘录

**Agent**

> SUPA 扩展已在 Biren106M 的官方 SDK 上一次编译成功，已经产出可导入的
> `.so`。接下来验证扩展可加载、前后向数值正确性和真实硬件性能。

首次导入失败后：

> 编译本身成功；刚才失败不是算子代码问题，而是 `LD_LIBRARY_PATH` 顺序
> 让 PyTorch 自带的 `libbr_common.so` 抢先加载，造成了 `torch_br` ABI
> 冲突。把 `torch_br/lib` 放到最前。

复验后：

> 加载顺序修正已验证成功：`torch_br`、Biren106M 和项目 `.so` 都能同时
> 导入。

## 实际动作

```bash
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch_br/lib:/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/birensupa/sdk/1.11.0.0.rc2/supa/lib:$LD_LIBRARY_PATH
bash src/supa/csrc/build.sh
python3 scripts/probe_biren.py --require-extension
```

远程账号和地址已脱敏；命令本身不包含认证信息。

## 证据

- 构建脚本：[build.sh](../../src/supa/csrc/build.sh)
- ABI 说明：[EXTENSION_ABI.md](../../src/supa/EXTENSION_ABI.md)
- 环境探针：[biren_environment.json](../../results/hardware/results/biren_environment.json)
- 最终设备：Biren106M，SUPA/Driver 1.11，扩展 `available=true`

