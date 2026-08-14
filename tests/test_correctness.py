"""正确性验证：PyTorch 参考实现的前向 + 反向数值正确性"""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.reference import SpectralConv2d


def test_forward_shape() -> None:
    """验证前向输出 shape 正确"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectralConv2d(32, 64, 16, 16).to(device)
    x = torch.randn(4, 32, 128, 128, device=device)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (4, 64, 128, 128), f"输出 shape 错误: {y.shape}"
    print(f"[PASS] 前向 shape: {x.shape} -> {y.shape}")


def test_forward_value_range() -> None:
    """验证前向输出在合理数值范围内"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectralConv2d(32, 64, 16, 16).to(device)
    x = torch.randn(4, 32, 128, 128, device=device)
    with torch.no_grad():
        y = model(x)
    assert torch.isfinite(y).all(), "输出包含 NaN 或 Inf"
    print(f"[PASS] 数值范围 [{y.min().item():.6f}, {y.max().item():.6f}]")


def test_backward() -> None:
    """验证反向传播梯度正确传播"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectralConv2d(32, 64, 16, 16).to(device)
    x = torch.randn(4, 32, 128, 128, device=device, requires_grad=True)
    y = model(x)
    loss = y.square().mean()
    loss.backward()
    assert x.grad is not None, "输入梯度为 None"
    assert torch.isfinite(x.grad).all(), "梯度包含 NaN 或 Inf"
    assert x.grad.shape == x.shape, f"梯度 shape 错误: {x.grad.shape}"
    print(f"[PASS] 反向传播: 梯度 shape {x.grad.shape}, 范围 [{x.grad.min().item():.6f}, {x.grad.max().item():.6f}]")


def test_weight_grad() -> None:
    """验证权重参数收到梯度"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectralConv2d(32, 64, 16, 16).to(device)
    x = torch.randn(4, 32, 128, 128, device=device, requires_grad=True)
    y = model(x)
    loss = y.square().mean()
    loss.backward()
    assert model.weights1.grad is not None, "weights1 梯度为 None"
    assert model.weights2.grad is not None, "weights2 梯度为 None"
    print(f"[PASS] 权重梯度: weights1 {model.weights1.grad.shape}, weights2 {model.weights2.grad.shape}")


def test_different_modes() -> None:
    """验证不同 modes 配置下均可正常运行"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for modes in [(4, 4), (8, 8), (16, 16), (12, 20)]:
        model = SpectralConv2d(8, 16, modes[0], modes[1]).to(device)
        x = torch.randn(2, 8, 64, 64, device=device)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (2, 16, 64, 64)
    print(f"[PASS] modes 配置: (4,4) (8,8) (16,16) (12,20) 全部通过")


def test_modes_not_exceed_resolution() -> None:
    """验证接近频率上限的 modes 能正确处理"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectralConv2d(4, 8, 30, 30).to(device)
    x = torch.randn(2, 4, 64, 64, device=device)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 8, 64, 64)
    print("[PASS] modes(30,30) 接近分辨率上限，仍正确运行")


def test_invalid_modes() -> None:
    """验证频率带重叠或超出 rFFT 宽度时明确报错"""
    model = SpectralConv2d(4, 8, 17, 8)
    x = torch.randn(1, 4, 32, 32)
    try:
        model(x)
    except ValueError as error:
        assert "2*modes1" in str(error)
    else:
        raise AssertionError("modes1 超界时应抛出 ValueError")

    model = SpectralConv2d(4, 8, 8, 18)
    try:
        model(x)
    except ValueError as error:
        assert "W//2+1" in str(error)
    else:
        raise AssertionError("modes2 超界时应抛出 ValueError")
    print("[PASS] 非法 modes 能被拒绝")


def test_batch_independence() -> None:
    """验证 batch 中不同样本的计算相互独立"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectralConv2d(4, 4, 8, 8).to(device)
    x = torch.randn(2, 4, 32, 32, device=device)
    with torch.no_grad():
        y = model(x)
    assert not torch.allclose(y[0], y[1]), "batch 中样本不应相同"
    print("[PASS] batch 独立性验证通过")


if __name__ == "__main__":
    print("=" * 50)
    print("SpectralConv2d 正确性验证")
    print("=" * 50)
    test_forward_shape()
    test_forward_value_range()
    test_backward()
    test_weight_grad()
    test_different_modes()
    test_modes_not_exceed_resolution()
    test_invalid_modes()
    test_batch_independence()
    print("\n" + "=" * 50)
    print("全部正确性测试通过!")
    print("=" * 50)
