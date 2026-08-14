"""性能测试：不同分辨率下的前向执行时间和显存占用"""

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.reference import SpectralConv2d
from src.supa import SupaSpectralConv2d
from src.supa import is_supa_device
from src.supa import max_memory_allocated
from src.supa import reset_peak_memory_stats
from src.supa import select_device
from src.supa import synchronize_device


def make_operator(
    backend: str,
    in_channels: int,
    out_channels: int,
    modes1: int,
    modes2: int
) -> nn.Module:
    """构建明确标注 backend 的待测算子。"""
    if backend == "reference":
        return SpectralConv2d(in_channels, out_channels, modes1, modes2)
    if backend in {"python", "extension"}:
        return SupaSpectralConv2d(
            in_channels,
            out_channels,
            modes1,
            modes2,
            backend=backend
        )
    raise ValueError(f"未知 backend={backend!r}")


def benchmark_resolution(
    height: int,
    width: int,
    batch_size: int = 4,
    in_channels: int = 32,
    out_channels: int = 64,
    modes1: int = 16,
    modes2: int = 16,
    warmup: int = 10,
    repeat: int = 100,
    backend: str = "reference",
    device: torch.device | None = None
) -> dict[str, str | float | int]:
    """测试指定分辨率下的前向性能"""
    if warmup < 0 or repeat <= 0:
        raise ValueError("warmup 必须非负，repeat 必须为正")
    if batch_size <= 0:
        raise ValueError("batch_size 必须为正")
    if device is None:
        device = select_device(strict_accelerator=backend == "extension")
    if backend == "extension" and not is_supa_device(device):
        raise RuntimeError("extension 性能测试必须在 BIREN SUPA 设备上运行")
    if backend == "reference" and is_supa_device(device):
        raise RuntimeError(
            "SUPA native rfft2 不能作为有效 reference；"
            "请使用 --backend python 测量顺序 FFT + PyTorch 复数乘基线"
        )
    model = make_operator(
        backend,
        in_channels,
        out_channels,
        modes1,
        modes2
    ).to(device)
    model.eval()
    x = torch.randn(
        batch_size,
        in_channels,
        height,
        width,
        device=device
    )

    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
    synchronize_device(device)

    reset_peak_memory_stats(device)
    iteration_times_ms: list[float] = []
    with torch.inference_mode():
        for _ in range(repeat):
            start = time.perf_counter()
            model(x)
            synchronize_device(device)
            iteration_times_ms.append(
                (time.perf_counter() - start) * 1000
            )
    mean_time_ms = statistics.fmean(iteration_times_ms)
    median_time_ms = statistics.median(iteration_times_ms)
    sorted_times = sorted(iteration_times_ms)
    p90_index = max(0, math.ceil(0.9 * len(sorted_times)) - 1)
    p90_time_ms = sorted_times[p90_index]

    memory_mb = max_memory_allocated(device) / 1024**2

    mean_batches_per_sec = 1000 / mean_time_ms
    median_batches_per_sec = 1000 / median_time_ms
    mean_samples_per_sec = batch_size * mean_batches_per_sec
    median_samples_per_sec = batch_size * median_batches_per_sec

    return {
        "backend": backend,
        "device": str(device),
        "resolution": f"{height}x{width}",
        "batch_size": batch_size,
        "mean_time_ms": round(mean_time_ms, 3),
        "median_time_ms": round(median_time_ms, 3),
        "p90_time_ms": round(p90_time_ms, 3),
        "mean_batches_per_sec": round(mean_batches_per_sec, 2),
        "median_batches_per_sec": round(median_batches_per_sec, 2),
        "mean_samples_per_sec": round(mean_samples_per_sec, 1),
        "throughput_samples_per_sec": round(median_samples_per_sec, 1),
        "median_ms_per_sample": round(median_time_ms / batch_size, 4),
        "memory_mb": round(memory_mb, 1),
        "warmup": warmup,
        "repeat": repeat
    }


def run_benchmark(
    backend: str = "reference",
    strict_accelerator: bool = False,
    warmup: int = 10,
    repeat: int = 100
) -> list[dict[str, str | float | int]]:
    """运行所有分辨率测试"""
    resolutions = [(64, 64), (128, 128), (256, 256)]
    device = select_device(
        strict_accelerator=strict_accelerator or backend == "extension"
    )
    results: list[dict[str, str | float | int]] = []
    for height, width in resolutions:
        result = benchmark_resolution(
            height,
            width,
            backend=backend,
            device=device,
            warmup=warmup,
            repeat=repeat
        )
        results.append(result)
        print(
            f"  {result['resolution']:>8s}:  "
            f"{result['median_time_ms']:>8.3f} ms  |  "
            f"P90 {result['p90_time_ms']:>8.3f} ms  |  "
            f"{result['memory_mb']:>6.1f} MB  |  "
            f"{result['throughput_samples_per_sec']:>8.1f} samples/s"
        )
    return results


def parse_args() -> argparse.Namespace:
    """解析性能测试参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("reference", "python", "extension"),
        default="reference"
    )
    parser.add_argument(
        "--strict-accelerator",
        action="store_true",
        help="未检测到 SUPA/CUDA 时直接失败，正式硬件测试必须启用"
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument(
        "--json",
        type=Path,
        help="可选：保存完整 benchmark 结果"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print(
        "SpectralConv2d 性能测试 "
        f"(backend={args.backend}, B=4, C_in=32, C_out=64, modes=16)"
    )
    print("=" * 60)
    print(f"{'分辨率':>8s}  {'前向时间':>10s}  {'显存':>8s}  {'吞吐':>12s}")
    print("-" * 60)
    results = run_benchmark(
        backend=args.backend,
        strict_accelerator=args.strict_accelerator,
        warmup=args.warmup,
        repeat=args.repeat
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8"
        )
        print(f"结果已写入: {args.json}")
    print("-" * 60)
    print("性能测试完成")
