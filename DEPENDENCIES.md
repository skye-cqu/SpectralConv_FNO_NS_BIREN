# 依赖与环境说明

## 1. 平台分层

本项目有两套用途不同的环境：

| 环境 | 用途 | PyTorch 来源 |
|---|---|---|
| 本地 CPU/CUDA | 参考实现、静态检查、小规模调试 | 官方 PyTorch |
| BIREN SUPA | 自定义 kernel、正式正确性/性能、FNO 前向与训练 | 平台预装 `br_pytorch` / `torch_br` |

禁止在 BIREN 官方环境中直接执行 `pip install torch`，否则可能覆盖平台适配版本。先运行 `python scripts/probe_biren.py` 并保存结果。

## 2. 本地开发环境

根据项目约定，优先复用根目录 `.venv`：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

标准 PyTorch 请根据本机 CPU/CUDA 条件，使用 PyTorch 官方对应安装命令单独安装。建议记录：

```powershell
python --version
python -m pip freeze
python -c "import torch; print(torch.__version__); print(torch.__config__.show())"
```

## 3. BIREN 环境

BIREN 侧依赖应由赛事镜像或平台模块提供：

- BIREN 驱动及 `brsmi`；
- SUPA runtime、编译器及扩展构建工具链；
- BIREN 适配版 PyTorch；
- `torch_br`，导入后注册 `torch.supa`；
- FFT 等价库对 `torch.fft.rfft2` / `irfft2` 的映射；
- C++ 编译器、Python headers、构建工具。

本次赛事服务器实测环境：Biren106M 32512 MiB，BR-SMI/Driver 1.11.0，
SUPA 1.11，`birensupa-sdk 1.11.0.0.rc2`，Python 3.10.12，
PyTorch `2.9.0+cu128`，`torch_br 1.10.0.20900+br1xx`。此处版本字符串中的
`cu128` 不代表使用 NVIDIA CUDA；正式设备接口是 `torch.supa`，
`torch.cuda.is_available()` 为 false。

推荐验证顺序：

```bash
python scripts/probe_biren.py --json results/biren_environment.json
python scripts/probe_biren.py --require-extension
python -m pip list
python tests/test_correctness.py
python tests/test_backward.py
python -m pytest tests/test_supa_operator.py
python tests/test_performance.py
```

若赛事镜像提供专用环境激活脚本，应先执行该脚本，并把脚本路径、镜像名和容器标签写入最终报告。不要把 token、密码、私有仓库地址或完整环境变量写入日志。

构建和导入扩展前，动态库路径必须按以下优先级排列：

```text
torch_br/lib : torch/lib : SUPA/lib : 原 LD_LIBRARY_PATH
```

项目的 `src/supa/csrc/build.sh` 会在构建阶段把 `torch_br/lib` 和 SUPA
runtime 置于已有路径之前，并在完成后打印正式运行所需的完整
`LD_LIBRARY_PATH`。复制该行到当前 shell 后再运行探针和测试。

## 4. 可复现性记录

每次正式测试至少保存以下信息：

- 日期与时区、Git commit、是否存在未提交改动；
- 主机/容器标识、BIREN GPU 型号、驱动和 runtime 版本；
- Python、PyTorch、`torch_br`、`torch.supa` 命名空间和 SUPA 编译器版本；
- 测试命令、随机种子、dtype、输入 shape、modes；
- warmup、repeat、同步方式、耗时统计口径与峰值显存口径；
- 数据集来源、文件哈希、划分、归一化参数和 checkpoint 哈希。

正式 FNO 数据用标准库下载，转换依赖 `numpy`、`scipy`，仅 MATLAB v7.3
文件需要 `h5py`：

```bash
python scripts/prepare_ns_data.py
```

脚本固定校验公开文件 SHA256，并将 1200 个 64×64 样本按原顺序切分为
1000 个训练样本和 200 个测试样本；转换产物及 metadata 位于 `data/`。

## 5. 常见问题

- `torch.supa` 不存在：确认已激活赛事环境并先 `import torch_br`。本次赛事
  镜像没有独立 `torch_supa` 包，不应将其作为必需依赖安装。
- `import torch_br` 出现 `libbr_common` 的 `undefined symbol`：检查
  `LD_LIBRARY_PATH`，确保 `torch_br/lib` 在 `torch/lib` 之前；不要通过重装
  PyTorch 处理这一动态库冲突。
- `rfft2` 报设备不支持：记录完整错误，确认 FFT 映射库和输入 dtype；不要静默退回 CPU 后仍标记为 BIREN 结果。
- SUPA native `rfft2/irfft2` roundtrip 正常但与 CPU 不一致：不能把自洽
  roundtrip 当作正确性证据。运行 `scripts/probe_biren.py` 检查 native 和
  sequential 两条路径；本项目在 SUPA 上使用
  `rfft(W) → fft(H)` / `ifft(H) → irfft(W)` 兼容层。
- 扩展导入失败：保存构建命令、编译器输出、PyTorch ABI 与动态库搜索路径。
- 性能异常：确认 warmup 后执行、计时前后设备同步，且没有把首次编译或数据加载时间计入 kernel 延迟。
