"""Device and measurement helpers shared by CPU, CUDA and BIREN SUPA."""

from __future__ import annotations

import importlib
from typing import Any

import torch

from .backend import is_supa_device


def _load_torch_br_if_installed() -> None:
    try:
        importlib.import_module("torch_br")
    except ModuleNotFoundError as error:
        if error.name == "torch_br":
            return
        raise RuntimeError(f"torch_br 依赖导入失败: {error}") from error
    except Exception as error:
        raise RuntimeError(
            f"torch_br 初始化失败: {type(error).__name__}: {error}"
        ) from error


def _api_is_available(api: Any) -> bool:
    is_available = getattr(api, "is_available", None)
    if callable(is_available):
        return bool(is_available())
    device_count = getattr(api, "device_count", None)
    if callable(device_count):
        return int(device_count()) > 0
    return False


def select_device(strict_accelerator: bool = False) -> torch.device:
    """Select SUPA first, then CUDA, and finally CPU.

    ``torch_br`` is imported lazily so that its custom device is registered
    before probing ``torch.supa``.  Benchmark entrypoints should pass
    ``strict_accelerator=True`` to avoid accidentally publishing CPU results.
    """
    _load_torch_br_if_installed()
    supa = getattr(torch, "supa", None)
    if supa is not None and _api_is_available(supa):
        return torch.device("supa")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if strict_accelerator:
        raise RuntimeError("未检测到可用的 BIREN SUPA 或 CUDA 加速设备")
    return torch.device("cpu")


def _accelerator_api(device: torch.device) -> Any | None:
    if is_supa_device(device):
        api = getattr(torch, "supa", None)
        if api is None:
            raise RuntimeError(
                f"设备 {device} 需要 torch.supa API，请先 import torch_br"
            )
        return api
    if device.type == "cuda":
        return torch.cuda
    return None


def synchronize_device(device: torch.device) -> None:
    """Synchronize the active accelerator; CPU is a no-op."""
    api = _accelerator_api(device)
    if api is None:
        return
    synchronize = getattr(api, "synchronize", None)
    if not callable(synchronize):
        raise RuntimeError(f"{device.type} runtime 缺少 synchronize()")
    synchronize()


def reset_peak_memory_stats(device: torch.device) -> None:
    """Reset peak allocated-memory statistics; CPU is a no-op."""
    api = _accelerator_api(device)
    if api is None:
        return
    reset = getattr(api, "reset_peak_memory_stats", None)
    if not callable(reset):
        raise RuntimeError(
            f"{device.type} runtime 缺少 reset_peak_memory_stats()"
        )
    reset()


def max_memory_allocated(device: torch.device) -> int:
    """Return peak allocated bytes for an accelerator, or zero for CPU."""
    api = _accelerator_api(device)
    if api is None:
        return 0
    get_max_memory = getattr(api, "max_memory_allocated", None)
    if not callable(get_max_memory):
        raise RuntimeError(
            f"{device.type} runtime 缺少 max_memory_allocated()"
        )
    return int(get_max_memory())

