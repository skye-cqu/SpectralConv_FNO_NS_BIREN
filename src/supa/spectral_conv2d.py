"""SpectralConv2d module with a selectable custom multiplication backend."""

from __future__ import annotations

import torch
import torch.nn as nn

from .backend import validate_backend
from .complex_mul import complex_mul_modes
from .fft import irfft2_compat, rfft2_compat


class SupaSpectralConv2d(nn.Module):
    """2D spectral convolution backed by Python or a BIREN SUPA extension.

    FFT and inverse FFT use the active PyTorch device implementation.  The two
    retained Fourier bands are contracted by ``complex_mul_modes``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        backend: str = "auto"
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels 和 out_channels 必须为正整数")
        if modes1 <= 0 or modes2 <= 0:
            raise ValueError("modes1 和 modes2 必须为正整数")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.backend = validate_backend(backend)

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(
                in_channels,
                out_channels,
                modes1,
                modes2,
                dtype=torch.cfloat
            )
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(
                in_channels,
                out_channels,
                modes1,
                modes2,
                dtype=torch.cfloat
            )
        )

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(
                f"输入必须为 [B, C, H, W]，实际 shape={tuple(x.shape)}"
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"输入通道应为 {self.in_channels}，实际为 {x.shape[1]}"
            )
        if not x.is_floating_point():
            raise TypeError(f"输入必须为实数浮点张量，实际 dtype={x.dtype}")

        height, width = x.shape[-2:]
        if 2 * self.modes1 > height:
            raise ValueError(
                f"modes1={self.modes1} 过大，要求 2*modes1 <= H={height}"
            )
        if self.modes2 > width // 2 + 1:
            raise ValueError(
                f"modes2={self.modes2} 过大，要求 "
                f"modes2 <= W//2+1={width // 2 + 1}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply FFT, two low-frequency contractions and inverse FFT."""
        self._validate_input(x)
        batch_size, _, height, width = x.shape
        x_ft = rfft2_compat(x)
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=x_ft.dtype,
            device=x.device
        )

        out_ft[:, :, :self.modes1, :self.modes2] = complex_mul_modes(
            x_ft[:, :, :self.modes1, :self.modes2],
            self.weights1,
            backend=self.backend
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = complex_mul_modes(
            x_ft[:, :, -self.modes1:, :self.modes2],
            self.weights2,
            backend=self.backend
        )
        return irfft2_compat(out_ft, (height, width))

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"modes1={self.modes1}, modes2={self.modes2}, "
            f"backend={self.backend!r}"
        )
