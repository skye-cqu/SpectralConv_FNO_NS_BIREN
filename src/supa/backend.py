"""Backend selection for the custom spectral-convolution operator."""

from __future__ import annotations

import os
from typing import Literal

import torch

BackendName = Literal["auto", "python", "extension"]

_VALID_BACKENDS = frozenset({"auto", "python", "extension"})
_SUPA_DEVICE_TYPES = frozenset({"supa", "privateuseone"})


def validate_backend(backend: str) -> BackendName:
    """Validate and narrow a user-provided backend name."""
    if backend not in _VALID_BACKENDS:
        choices = ", ".join(sorted(_VALID_BACKENDS))
        raise ValueError(f"未知 backend={backend!r}，可选值为: {choices}")
    return backend  # type: ignore[return-value]


def is_supa_device(device: torch.device) -> bool:
    """Return whether a tensor device should require the SUPA extension.

    Native BIREN builds may expose either ``supa`` or PyTorch's
    ``privateuseone`` device type.  Some compatibility builds reuse the CUDA
    device spelling; those deployments can set
    ``SPECTRALCONV_DEVICE_BACKEND=supa`` to make ``auto`` strict.
    """
    if device.type in _SUPA_DEVICE_TYPES:
        return True
    forced_backend = os.environ.get("SPECTRALCONV_DEVICE_BACKEND", "").lower()
    return forced_backend == "supa"


def resolve_backend(backend: str, device: torch.device) -> Literal[
    "python",
    "extension"
]:
    """Resolve ``auto`` without silently falling back on a SUPA device."""
    validated = validate_backend(backend)
    if validated != "auto":
        return validated
    if is_supa_device(device):
        return "extension"
    return "python"

