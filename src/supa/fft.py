"""Portable 2D real FFT with a sequential workaround for BIREN SUPA."""

from __future__ import annotations

import torch

from .backend import is_supa_device


def sequential_rfft2(x: torch.Tensor) -> torch.Tensor:
    """Compute rFFT over W, followed by complex FFT over H."""
    width_frequency = torch.fft.rfft(x, dim=-1)
    return torch.fft.fft(width_frequency, dim=-2)


def sequential_irfft2(
    x_ft: torch.Tensor,
    size: tuple[int, int]
) -> torch.Tensor:
    """Invert H with complex IFFT, followed by W with real IFFT."""
    _, width = size
    height_spatial = torch.fft.ifft(x_ft, dim=-2)
    return torch.fft.irfft(height_spatial, n=width, dim=-1)


def rfft2_compat(x: torch.Tensor) -> torch.Tensor:
    """Use the sequential transform on SUPA and native rfft2 elsewhere."""
    if is_supa_device(x.device):
        return sequential_rfft2(x)
    return torch.fft.rfft2(x)


def irfft2_compat(
    x_ft: torch.Tensor,
    size: tuple[int, int]
) -> torch.Tensor:
    """Use the sequential inverse on SUPA and native irfft2 elsewhere."""
    if is_supa_device(x_ft.device):
        return sequential_irfft2(x_ft, size)
    return torch.fft.irfft2(x_ft, s=size)
