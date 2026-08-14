"""Shared runtime helpers for training and evaluation commands."""

import random
import time
from collections.abc import Callable

import numpy as np
import torch

from src.supa.runtime import max_memory_allocated
from src.supa.runtime import select_device
from src.supa.runtime import synchronize_device


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto`` while allowing explicit BIREN device strings."""
    if requested != "auto":
        if requested.startswith("supa"):
            select_device(strict_accelerator=True)
        return torch.device(requested)
    return select_device()


def synchronize(device: torch.device) -> None:
    """Synchronize accelerators before wall-clock measurements."""
    synchronize_device(device)


def benchmark_forward(
    function: Callable[[], torch.Tensor],
    device: torch.device,
    batch_size: int,
    height: int,
    width: int,
    warmup: int,
    repeats: int
) -> dict[str, float]:
    """Benchmark a no-argument model closure with synchronized timing."""
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    with torch.inference_mode():
        for _ in range(warmup):
            function()
        synchronize(device)
        start = time.perf_counter()
        for _ in range(repeats):
            function()
        synchronize(device)
    elapsed = time.perf_counter() - start
    samples = batch_size * repeats
    return {
        "elapsed_seconds": elapsed,
        "grid_points_per_second": samples * height * width / elapsed,
        "samples_per_second": samples / elapsed,
        "milliseconds_per_sample": elapsed * 1000.0 / samples,
        "peak_memory_mb": max_memory_allocated(device) / 1024.0 / 1024.0
    }
