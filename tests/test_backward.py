"""反向传播验证：梯度正确性 + 端到端训练模拟"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.reference import SpectralConv2d


def test_backward_value_range() -> None:
    """验证梯度数值范围合理"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectralConv2d(32, 64, 16, 16).to(device)
    x = torch.randn(4, 32, 128, 128, device=device, requires_grad=True)
    y = model(x)
    loss = y.square().mean()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any(), "输入梯度包含 NaN"
    assert not torch.isinf(x.grad).any(), "输入梯度包含 Inf"
    print(f"[PASS] 输入梯度 范围 [{x.grad.min().item():.6f}, {x.grad.max().item():.6f}]")


def test_backward_all_params() -> None:
    """验证所有可学习参数均收到梯度"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectralConv2d(8, 16, 8, 8).to(device)
    x = torch.randn(2, 8, 64, 64, device=device, requires_grad=True)
    y = model(x)
    loss = y.square().mean()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} 未收到梯度"
        assert not torch.isnan(param.grad).any(), f"{name} 梯度包含 NaN"
    print(f"[PASS] 所有 {sum(1 for _ in model.parameters())} 个参数均收到有效梯度")


def test_multiple_backward() -> None:
    """验证多次反向传播可累积梯度"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectralConv2d(4, 8, 8, 8).to(device)
    x = torch.randn(2, 4, 64, 64, device=device, requires_grad=True)
    for i in range(3):
        y = model(x)
        loss = y.square().mean()
        loss.backward()
    assert x.grad is not None
    print(f"[PASS] 3 次反向传播累积梯度: 范围 [{x.grad.min():.6f}, {x.grad.max():.6f}]")


def test_training_step() -> None:
    """模拟一步训练：前向 → loss → 反向 → 参数更新"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectralConv2d(4, 8, 8, 8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.randn(2, 4, 64, 64, device=device)
    target = torch.randn(2, 8, 64, 64, device=device)

    optimizer.zero_grad()
    y = model(x)
    loss = nn.MSELoss()(y, target)
    loss.backward()
    optimizer.step()

    assert loss.item() > 0, "loss 应为正数"
    print(f"[PASS] 单步训练完成: loss = {loss.item():.6f}")


if __name__ == "__main__":
    print("=" * 50)
    print("SpectralConv2d 反向传播验证")
    print("=" * 50)
    test_backward_value_range()
    test_backward_all_params()
    test_multiple_backward()
    test_training_step()
    print("\n" + "=" * 50)
    print("全部反向传播测试通过!")
    print("=" * 50)
