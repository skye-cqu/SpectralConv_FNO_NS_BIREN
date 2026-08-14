# SUPA 扩展 ABI 与远端确认项

Python 层加载模块名默认为 `src.supa.spectralconv_supa_ext`，也可通过
`SPECTRALCONV_SUPA_EXTENSION_MODULE` 指定。扩展应提供以下任一入口：

- pybind 导出 `complex_mul_modes(x, weight)`
- 注册 `torch.ops.spectralconv_supa.complex_mul_modes`

固定张量 ABI：

| 张量 | shape | dtype | 内存 |
|---|---|---|---|
| `x` | `[B, C_in, K]` | complex64/complex128 | contiguous |
| `weight` | `[C_in, C_out, K]` | 与 `x` 相同 | contiguous |
| 输出 | `[B, C_out, K]` | 与 `x` 相同 | contiguous |

计算语义为
`out[b,o,k] = sum_i x[b,i,k] * weight[i,o,k]`。kernel 只对
`C_in` 归约，`B/C_out/K` 可并行。扩展必须支持 autograd；可注册显式 backward，
也可用 PyTorch dispatcher 的 Autograd key 提供反向实现。

当前扩展源码位于 `src/supa/csrc`，在目标服务器项目根目录运行：

```bash
bash src/supa/csrc/build.sh
python -m pytest -q tests/test_supa_operator.py
```

构建结果写入 `src/supa/spectralconv_supa_ext*.so`。运行时必须按
`torch_br/lib : torch/lib : SUPA/lib` 的顺序设置 `LD_LIBRARY_PATH`；
`torch/lib` 在前会触发 `libbr_common` 的 `undefined symbol`。构建脚本完成后
会打印可直接执行的 `export` 命令。远端测试
`test_real_supa_extension_forward_and_backward` 会将 SUPA kernel 的前向和两组
梯度与 CPU PyTorch 结果比较。

## 服务器探针结果

截至 2026-07-31，目标服务器已确认：

- GPU：Biren106M 32 GB
- BR-SMI / driver / SUPA：1.11
- SDK：`birensupa-sdk 1.11.0.0.rc2`
- Python：3.10.12
- PyTorch：`2.9.0+cu128`
- BIREN 插件：`torch_br 1.10.0.20900+br1xx`
- `torch_supa` 不存在，`torch.cuda.is_available()` 为 false
- native `torch.fft.rfft2` 疑似只执行 W 维，虽可与 native `irfft2`
  自洽 roundtrip，但相对 CPU 误差约为 `1`
- 顺序执行 `rfft(W) → fft(H)` 后相对 CPU 误差为 `1.3775e-7`

因此不得从 PyTorch 的 `+cu128` 版本字符串推断 CUDA 可用，也不得硬编码
`torch_supa` 导入。模块通过输入张量的真实 `device.type` 选择 backend，并
在 SUPA 上使用 `src/supa/fft.py` 的顺序二维 FFT 兼容层；CPU/CUDA 继续调用
native `rfft2/irfft2`。

## 尚待远端确认

在 BIREN 服务器上编写构建文件前还必须确认：

1. `import torch_br` 后设备报告为 `supa`、`privateuseone` 还是其他名称。
2. 平台使用兼容的 `torch.utils.cpp_extension`、`torch_br` 自带构建入口，
   还是独立 SUPA 编译工具链。
3. 若设备兼容层报告为 `cuda`，运行前设置
   `SPECTRALCONV_DEVICE_BACKEND=supa`，保证
   `backend="auto"` 不会静默使用 Python fallback。
4. C++ ABI、SUPA 源文件扩展名、编译参数、架构参数以及自定义 op 注册方式。
5. complex64/complex128 的设备侧类型定义和 `torch.fft.rfft2/irfft2` 支持情况。
6. 自定义 op 的反向注册 API；正式性能与正确性测试必须使用
   `backend="extension"`。
