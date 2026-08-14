"""
从 deepchem/models/torch_models/layers.py 提取的 SpectralConv
核心亮点：dim 参数化，支持 1D/2D/3D 统一接口
"""

import torch
import torch.nn as nn
from typing import Union, Tuple, List


class SpectralConv(nn.Module):
    """n-Dimensional Fourier layer.

    用 dims 参数统一 1D/2D/3D，消除了代码重复。
    权重初始化方式不同：用 torch.complex(real, imag) 而非 nn.Parameter(complex64)
    包含可选的实值偏置项。
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 modes: Union[int, Tuple[int, ...], List[int]],
                 dims: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dims = dims

        if isinstance(modes, int):
            self.modes = (modes,) * dims
        elif isinstance(modes, (tuple, list)):
            if len(modes) != dims:
                raise ValueError("Length of modes must equal dims.")
            self.modes = tuple(modes)

        modes = list(self.modes)
        modes[-1] = modes[-1] // 2 + 1  # rfft 最后一维非对称
        self.modes = tuple(modes)

        weight_shape = (in_channels, out_channels) + self.modes
        self.scale = (1 / (in_channels + out_channels))**0.5
        real = torch.randn(*weight_shape) * self.scale
        imag = torch.randn(*weight_shape) * self.scale
        self.weights = nn.Parameter(torch.complex(real, imag))

        bias_shape = (1, out_channels) + (1,) * self.dims
        self.bias = nn.Parameter(torch.randn(*bias_shape) * self.scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.ndim == self.dims + 2:
            raise ValueError(...)

        fft_dims = tuple(range(2, x.ndim))
        x_ft = torch.fft.rfftn(x, dim=fft_dims)

        for num_modes, size in zip(self.modes, x_ft.shape[2:]):
            if num_modes > size:
                raise ValueError(...)

        out_ft = torch.zeros(x.shape[0], self.out_channels, *x_ft.shape[2:],
                             dtype=torch.cfloat, device=x.device)

        slices = tuple(slice(0, m) for m in self.modes)
        idx = (slice(None), slice(None)) + slices
        out_ft[idx] = torch.einsum(
            "b i ... , i o ... -> b o ...", x_ft[idx], self.weights)

        x_out = torch.fft.irfftn(out_ft, s=x.shape[2:],
                                  dim=tuple(range(2, x.ndim))).real
        return x_out + self.bias
