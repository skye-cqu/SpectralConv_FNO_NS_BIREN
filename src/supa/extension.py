"""Strict loader for the separately built BIREN SUPA extension."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib
import os
from types import ModuleType
from typing import Callable

import torch

ExtensionOperator = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
ExtensionBackwardOperator = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor]
]

DEFAULT_EXTENSION_MODULE = "src.supa.spectralconv_supa_ext"
EXTENSION_MODULE_ENV = "SPECTRALCONV_SUPA_EXTENSION_MODULE"


class ExtensionUnavailableError(RuntimeError):
    """Raised when the requested SUPA extension cannot be loaded."""


@dataclass(frozen=True)
class ExtensionStatus:
    """Diagnostic result for a SUPA extension probe."""

    available: bool
    module_name: str
    detail: str


@dataclass(frozen=True)
class ExtensionOperators:
    """Forward and optional explicit-backward extension entrypoints."""

    forward: ExtensionOperator
    backward: ExtensionBackwardOperator | None


def _module_name() -> str:
    return os.environ.get(EXTENSION_MODULE_ENV, DEFAULT_EXTENSION_MODULE)


def _operators_from_module(module: ModuleType) -> ExtensionOperators:
    forward = getattr(module, "complex_mul_modes_forward", None)
    if forward is None:
        forward = getattr(module, "complex_mul_modes", None)
    if forward is None or not callable(forward):
        raise AttributeError(
            "扩展模块必须导出 callable "
            "complex_mul_modes_forward(x, weight)"
        )
    backward = getattr(module, "complex_mul_modes_backward", None)
    if backward is not None and not callable(backward):
        raise AttributeError("complex_mul_modes_backward 必须为 callable")
    return ExtensionOperators(forward, backward)


def _operators_from_torch_library() -> ExtensionOperators | None:
    namespace = getattr(torch.ops, "spectralconv_supa", None)
    if namespace is None:
        return None
    forward = getattr(namespace, "complex_mul_modes_forward", None)
    if forward is None:
        forward = getattr(namespace, "complex_mul_modes", None)
    if forward is None or not callable(forward):
        return None
    backward = getattr(namespace, "complex_mul_modes_backward", None)
    if backward is not None and not callable(backward):
        backward = None
    return ExtensionOperators(forward, backward)


@lru_cache(maxsize=None)
def _load_extension_operators(module_name: str) -> ExtensionOperators:
    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        registered_operators = _operators_from_torch_library()
        if registered_operators is not None:
            return registered_operators
        raise ExtensionUnavailableError(
            f"无法加载 SUPA 扩展 {module_name!r}: "
            f"{type(error).__name__}: {error}。请先在 BIREN 环境编译扩展，"
            f"或通过 {EXTENSION_MODULE_ENV} 指定已构建模块名。"
        ) from error

    try:
        return _operators_from_module(module)
    except AttributeError as error:
        registered_operators = _operators_from_torch_library()
        if registered_operators is not None:
            return registered_operators
        raise ExtensionUnavailableError(
            f"SUPA 扩展 {module_name!r} 已导入，但未注册所需算子: {error}"
        ) from error


def load_extension_operators() -> ExtensionOperators:
    """Load the configured forward and explicit-backward extension ops."""
    return _load_extension_operators(_module_name())


def load_extension_operator() -> ExtensionOperator:
    """Load and cache the configured SUPA complex multiplication op.

    The extension may either export a pybind function or register
    ``torch.ops.spectralconv_supa.complex_mul_modes``.  Import errors are never
    hidden because a performance run on BIREN must not be mislabeled as a
    custom-kernel result.
    """
    return load_extension_operators().forward


def clear_extension_cache() -> None:
    """Clear a loaded operator after rebuilding an extension in-place."""
    _load_extension_operators.cache_clear()


def probe_extension() -> ExtensionStatus:
    """Probe extension availability without running a GPU kernel."""
    module_name = _module_name()
    try:
        operators = load_extension_operators()
    except ExtensionUnavailableError as error:
        return ExtensionStatus(False, module_name, str(error))
    backward_status = "显式 backward 已注册"
    if operators.backward is None:
        backward_status = "依赖扩展自身 autograd"
    return ExtensionStatus(
        True,
        module_name,
        f"complex_mul_modes forward 已注册，{backward_status}"
    )
