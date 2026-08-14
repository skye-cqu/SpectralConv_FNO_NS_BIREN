"""
从 deepchem/models/torch_models/fno.py 提取的 FNOBlock + FNO 模型
核心亮点：FNOBlock 结构（SpectralConv + 1x1 Conv skip）可直接复用至 fno_ns/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple


class FNOBlock(nn.Module):
    """FNO block: SpectralConv(x) + Conv(x) + ReLU

    对于必选题，SpectralConv 在 torch.extension 中实现后，
    此 FNOBlock 可直接在 Python 层组装使用。

    Args:
        width: 隐藏层通道数
        modes: 傅里叶模态数
        dims: 空间维度 (1, 2, 或 3)
    """
    def __init__(self, width: int, modes: Union[int, Tuple[int, ...]],
                 dims: int) -> None:
        super().__init__()
        self.spectral_conv = SpectralConv(width, width, modes, dims=dims)
        if dims == 1:
            self.w = nn.Conv1d(width, width, 1)
        elif dims == 2:
            self.w = nn.Conv2d(width, width, 1)
        elif dims == 3:
            self.w = nn.Conv3d(width, width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.spectral_conv(x)
        x2 = self.w(x)
        return F.relu(x1 + x2)


class FNO(nn.Module):
    """完整 FNO 模型

    结构: lifting → [FNOBlock × depth] → projection
    包含可选的位置编码（网格坐标）
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 modes: Union[int, Tuple[int, ...]],
                 width: int,
                 dims: int,
                 depth: int = 4,
                 positional_encoding: bool = False):
        super().__init__()
        self.dims = dims
        self.in_channels = in_channels + dims if positional_encoding else in_channels
        self.positional_encoding = positional_encoding

        self.lifting = nn.Sequential(
            nn.Linear(self.in_channels, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width))

        self.fno_blocks = nn.Sequential(
            *[FNOBlock(width, modes, dims=dims) for _ in range(depth)])

        self.projection = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, out_channels))

    def forward(self, x):
        # ... (详见原文件完整实现)
        pass
