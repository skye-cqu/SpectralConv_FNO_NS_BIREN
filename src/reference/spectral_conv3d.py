import torch
import torch.nn as nn
import torch.fft


def compl_mul3d(input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """复数批量矩阵乘: (B, C_in, D, H, W) x (C_in, C_out, D, H, W) -> (B, C_out, D, H, W)"""
    return torch.einsum("bixyz,ioxyz->boxyz", input, weights)


class SpectralConv3d(nn.Module):
    """3D Spectral Convolution — FNO 核心算子（进阶）

    计算流程: rfftn → 频域复数矩阵乘（四象限低频截断）→ irfftn

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        modes1: D 维傅里叶模态截断数
        modes2: H 维傅里叶模态截断数
        modes3: W 维傅里叶模态截断数
    """
    def __init__(self, in_channels: int, out_channels: int,
                 modes1: int, modes2: int, modes3: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.rand(
            in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(
            in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(scale * torch.rand(
            in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(scale * torch.rand(
            in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        out_ft = torch.zeros(B, self.out_channels, D, H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)

        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        return torch.fft.irfftn(out_ft, s=(D, H, W))
