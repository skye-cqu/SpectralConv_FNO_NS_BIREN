"""
从 fkodom/fft-conv-pytorch 提取的复数矩阵乘实现
关键价值：用 real/imag 分离运算替代 torch.einsum，降低对 SUPA einsum 的依赖
"""

import torch
from torch import Tensor


def complex_matmul(a: Tensor, b: Tensor, groups: int = 1) -> Tensor:
    """复数矩阵乘：用 real/imag 分离运算实现，不依赖 einsum

    Args:
        a: [B, C_in, H, W] 复数张量
        b: [C_in, C_out, H, W] 复数权重
    Returns:
        [B, C_out, H, W] 复数张量
    """
    a = a.view(a.size(0), groups, -1, *a.shape[2:])
    b = b.view(groups, -1, *b.shape[1:])

    a = torch.movedim(a, 2, a.dim() - 1).unsqueeze(-2)
    b = torch.movedim(b, (1, 2), (b.dim() - 1, b.dim() - 2))

    # 复数乘： (a_r + i*a_i) * (b_r + i*b_i)
    #   real = a_r @ b_r - a_i @ b_i
    #   imag = a_i @ b_r + a_r @ b_i
    real = a.real @ b.real - a.imag @ b.imag
    imag = a.imag @ b.real + a.real @ b.imag
    real = torch.movedim(real, real.dim() - 1, 2).squeeze(-1)
    imag = torch.movedim(imag, imag.dim() - 1, 2).squeeze(-1)

    c = torch.zeros(real.shape, dtype=torch.complex64, device=a.device)
    c.real, c.imag = real, imag

    return c.view(c.size(0), -1, *c.shape[3:])


# ============================================================
# SpectralConv2d 中使用 complex_matmul 的等价替换：
# ============================================================
#
# 原版（依赖 einsum）：
#   out_ft[idx] = torch.einsum("bixy,ioxy->boxy", x_ft[idx], self.weights)
#
# 等价替换（不依赖 einsum）：
#   w = self.weights.unsqueeze(0).expand(...)  # [1, C_in, C_out, H, W]
#   out = complex_matmul(x_ft[idx], w)
#
# 或者更直接地（针对 SpectralConv2d 精简版）：
#   x_in = x_ft[:, :, :modes1, :modes2]    # [B, C_in, M, M]
#   w = self.weights                        # [C_in, C_out, M, M]
#   x_in = x_in.permute(1, 0, 2, 3)        # [C_in, B, M, M]
#   w = w.unsqueeze(1)                      # [C_in, 1, C_out, M, M]
#   out = x_in.real @ w.real - x_in.imag @ w.imag  # [C_in, B, C_out, M, M]
#   out = out.sum(dim=0).permute(1, 0, 2, 3)        # [B, C_out, M, M]
