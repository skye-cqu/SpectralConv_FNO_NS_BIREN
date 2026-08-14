import torch
import torch.nn as nn
import torch.fft


def compl_mul2d(input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """复数批量矩阵乘: (B, C_in, H, W) x (C_in, C_out, H, W) -> (B, C_out, H, W)"""
    return torch.einsum("bixy,ioxy->boxy", input, weights)


class SpectralConv2d(nn.Module):
    """2D Spectral Convolution — FNO 核心算子

    计算流程: FFT → 频域复数矩阵乘（低频截断）→ IFFT

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        modes1: H 维傅里叶模态截断数
        modes2: W 维傅里叶模态截断数
    """
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels 和 out_channels 必须为正整数")
        if modes1 <= 0 or modes2 <= 0:
            raise ValueError("modes1 和 modes2 必须为正整数")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"输入必须为 [B, C, H, W]，实际 shape={tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"输入通道应为 {self.in_channels}，实际为 {x.shape[1]}"
            )
        if not x.is_floating_point():
            raise TypeError(f"输入必须为实数浮点张量，实际 dtype={x.dtype}")

        batch_size, _, height, width = x.shape
        if 2 * self.modes1 > height:
            raise ValueError(
                f"modes1={self.modes1} 过大，要求 2*modes1 <= H={height}"
            )
        if self.modes2 > width // 2 + 1:
            raise ValueError(
                f"modes2={self.modes2} 过大，要求 modes2 <= W//2+1={width // 2 + 1}"
            )

        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=x_ft.dtype,
            device=x.device
        )

        out_ft[:, :, :self.modes1, :self.modes2] = compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2],
            self.weights1
        )

        out_ft[:, :, -self.modes1:, :self.modes2] = compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2],
            self.weights2
        )

        return torch.fft.irfft2(out_ft, s=(height, width))
