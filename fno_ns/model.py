"""Fourier Neural Operator for 2D Navier-Stokes vorticity prediction."""

from collections.abc import Callable
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

SpectralLayerFactory = Callable[[int, int, int, int], nn.Module]


def get_spectral_layer_factory(backend: str) -> SpectralLayerFactory:
    """Resolve a reference or SUPA spectral convolution implementation."""
    if backend == "reference":
        from src.reference.spectral_conv2d import SpectralConv2d

        return SpectralConv2d
    if backend == "supa":
        from src.supa import SupaSpectralConv2d

        return partial(
            SupaSpectralConv2d,
            backend="extension"
        )
    raise ValueError(
        f"Unknown spectral backend: {backend!r}; "
        "expected 'reference' or 'supa'"
    )


class FourierLayer2d(nn.Module):
    """A spectral branch plus a learnable pointwise residual branch."""

    def __init__(
        self,
        width: int,
        modes1: int,
        modes2: int,
        spectral_layer_factory: SpectralLayerFactory
    ) -> None:
        super().__init__()
        self.spectral = spectral_layer_factory(width, width, modes1, modes2)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.spectral(values) + self.pointwise(values)


class FNO2d(nn.Module):
    """FNO mapping an initial 2D vorticity field to a target field.

    The input is ``[B,C,H,W]``. Two normalized coordinate channels are
    concatenated before lifting. At least four Fourier layers are enforced to
    satisfy the competition requirement.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 32,
        depth: int = 4,
        backend: str = "reference",
        spectral_layer_factory: SpectralLayerFactory | None = None
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0 or width <= 0:
            raise ValueError("Channel counts and width must be positive")
        if modes1 <= 0 or modes2 <= 0:
            raise ValueError("Fourier mode counts must be positive")
        if depth < 4:
            raise ValueError("FNO2d requires depth >= 4 for the competition")
        if spectral_layer_factory is None:
            spectral_layer_factory = get_spectral_layer_factory(backend)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.depth = depth
        self.backend = backend
        self.lifting = nn.Conv2d(in_channels + 2, width, kernel_size=1)
        self.layers = nn.ModuleList(
            [
                FourierLayer2d(
                    width,
                    modes1,
                    modes2,
                    spectral_layer_factory
                )
                for _ in range(depth)
            ]
        )
        self.projection1 = nn.Conv2d(width, 128, kernel_size=1)
        self.projection2 = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4:
            raise ValueError(
                f"Input must be [B,C,H,W], got {tuple(values.shape)}"
            )
        if values.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, got {values.shape[1]}"
            )
        coordinates = self._coordinate_grid(values)
        hidden = self.lifting(torch.cat((values, coordinates), dim=1))
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if index + 1 < self.depth:
                hidden = F.gelu(hidden)
        hidden = F.gelu(self.projection1(hidden))
        return self.projection2(hidden)

    def config(self) -> dict[str, int | str]:
        """Return constructor settings stored with checkpoints."""
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "modes1": self.modes1,
            "modes2": self.modes2,
            "width": self.width,
            "depth": self.depth,
            "backend": self.backend
        }

    @staticmethod
    def _coordinate_grid(values: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = values.shape
        y_axis = torch.linspace(
            0.0,
            1.0,
            height,
            dtype=values.dtype,
            device=values.device
        )
        x_axis = torch.linspace(
            0.0,
            1.0,
            width,
            dtype=values.dtype,
            device=values.device
        )
        grid_y = y_axis.reshape(1, 1, height, 1).expand(
            batch_size,
            1,
            height,
            width
        )
        grid_x = x_axis.reshape(1, 1, 1, width).expand(
            batch_size,
            1,
            height,
            width
        )
        return torch.cat((grid_y, grid_x), dim=1)
