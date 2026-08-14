"""Focused tests for the selectable SUPA spectral-convolution operator."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import src.supa.fft as fft_compat
from src.reference import SpectralConv2d
from src.supa import (
    ExtensionUnavailableError,
    SupaSpectralConv2d,
    complex_mul_modes,
    max_memory_allocated,
    probe_extension,
    reset_peak_memory_stats,
    sequential_irfft2,
    sequential_rfft2,
    select_device,
    synchronize_device
)


def _copy_weights(
    source: SpectralConv2d,
    target: SupaSpectralConv2d
) -> None:
    with torch.no_grad():
        target.weights1.copy_(source.weights1)
        target.weights2.copy_(source.weights2)


def _supa_extension_available() -> bool:
    try:
        __import__("torch_br")
    except Exception:
        return False
    supa = getattr(torch, "supa", None)
    is_available = getattr(supa, "is_available", None)
    return bool(
        callable(is_available)
        and is_available()
        and probe_extension().available
    )


def test_complex_mul_python_matches_einsum_and_backward() -> None:
    torch.manual_seed(0)
    x = torch.randn(
        2,
        3,
        4,
        5,
        dtype=torch.cfloat,
        requires_grad=True
    )
    weight = torch.randn(
        3,
        6,
        4,
        5,
        dtype=torch.cfloat,
        requires_grad=True
    )
    expected = torch.einsum("bixy,ioxy->boxy", x, weight)
    actual = complex_mul_modes(x, weight, backend="python")
    torch.testing.assert_close(actual, expected)

    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    actual_x_grad = x.grad.detach().clone()
    actual_weight_grad = weight.grad.detach().clone()
    x.grad = None
    weight.grad = None
    expected.backward(gradient)
    torch.testing.assert_close(x.grad, actual_x_grad)
    torch.testing.assert_close(weight.grad, actual_weight_grad)


def test_sequential_rfft2_matches_native_forward_and_backward() -> None:
    torch.manual_seed(11)
    native_input = torch.randn(
        2,
        3,
        9,
        10,
        dtype=torch.double,
        requires_grad=True
    )
    sequential_input = native_input.detach().clone().requires_grad_(True)
    native = torch.fft.rfft2(native_input)
    sequential = sequential_rfft2(sequential_input)
    torch.testing.assert_close(sequential, native, rtol=1e-12, atol=1e-12)

    gradient = torch.randn_like(native)
    native.backward(gradient)
    sequential.backward(gradient)
    torch.testing.assert_close(
        sequential_input.grad,
        native_input.grad,
        rtol=1e-12,
        atol=1e-12
    )


def test_sequential_irfft2_matches_native_forward_and_backward() -> None:
    torch.manual_seed(12)
    size = (9, 10)
    native_input = torch.randn(
        2,
        3,
        size[0],
        size[1] // 2 + 1,
        dtype=torch.cdouble,
        requires_grad=True
    )
    sequential_input = native_input.detach().clone().requires_grad_(True)
    native = torch.fft.irfft2(native_input, s=size)
    sequential = sequential_irfft2(sequential_input, size)
    torch.testing.assert_close(sequential, native, rtol=1e-12, atol=1e-12)

    gradient = torch.randn_like(native)
    native.backward(gradient)
    sequential.backward(gradient)
    torch.testing.assert_close(
        sequential_input.grad,
        native_input.grad,
        rtol=1e-12,
        atol=1e-12
    )


def test_python_module_matches_reference_forward_and_backward() -> None:
    torch.manual_seed(1)
    reference = SpectralConv2d(3, 5, 4, 6)
    model = SupaSpectralConv2d(3, 5, 4, 6, backend="python")
    _copy_weights(reference, model)

    reference_x = torch.randn(2, 3, 16, 18, requires_grad=True)
    model_x = reference_x.detach().clone().requires_grad_(True)
    reference_output = reference(reference_x)
    model_output = model(model_x)
    torch.testing.assert_close(model_output, reference_output)

    gradient = torch.randn_like(reference_output)
    reference_output.backward(gradient)
    model_output.backward(gradient)
    torch.testing.assert_close(model_x.grad, reference_x.grad)
    torch.testing.assert_close(model.weights1.grad, reference.weights1.grad)
    torch.testing.assert_close(model.weights2.grad, reference.weights2.grad)


def test_module_sequential_fft_path_matches_reference(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fft_compat, "is_supa_device", lambda device: True)
    torch.manual_seed(13)
    reference = SpectralConv2d(2, 3, 3, 4)
    model = SupaSpectralConv2d(2, 3, 3, 4, backend="python")
    _copy_weights(reference, model)

    reference_x = torch.randn(2, 2, 10, 12, requires_grad=True)
    model_x = reference_x.detach().clone().requires_grad_(True)
    reference_output = reference(reference_x)
    model_output = model(model_x)
    torch.testing.assert_close(
        model_output,
        reference_output,
        rtol=1e-5,
        atol=1e-6
    )

    gradient = torch.randn_like(reference_output)
    reference_output.backward(gradient)
    model_output.backward(gradient)
    torch.testing.assert_close(
        model_x.grad,
        reference_x.grad,
        rtol=1e-5,
        atol=1e-6
    )
    torch.testing.assert_close(
        model.weights1.grad,
        reference.weights1.grad,
        rtol=1e-5,
        atol=1e-6
    )
    torch.testing.assert_close(
        model.weights2.grad,
        reference.weights2.grad,
        rtol=1e-5,
        atol=1e-6
    )


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((2, 3, 7, 16), "2\\*modes1"),
        ((2, 3, 16, 8), "modes2")
    ]
)
def test_module_rejects_modes_outside_resolution(
    shape: tuple[int, int, int, int],
    message: str
) -> None:
    model = SupaSpectralConv2d(3, 5, 4, 6, backend="python")
    with pytest.raises(ValueError, match=message):
        model(torch.randn(*shape))


def test_extension_backend_fails_loudly_when_unavailable(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "SPECTRALCONV_SUPA_EXTENSION_MODULE",
        "_definitely_missing_spectralconv_extension"
    )
    x = torch.randn(1, 2, 3, dtype=torch.cfloat)
    weight = torch.randn(2, 4, 3, dtype=torch.cfloat)
    with pytest.raises(ExtensionUnavailableError, match="无法加载 SUPA 扩展"):
        complex_mul_modes(x, weight, backend="extension")


def test_extension_adapter_uses_flat_contiguous_abi(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    module_name = "_fake_spectralconv_extension"
    module = ModuleType(module_name)

    def fake_complex_mul_modes(
        x_flat: torch.Tensor,
        weight_flat: torch.Tensor
    ) -> torch.Tensor:
        assert x_flat.is_contiguous()
        assert weight_flat.is_contiguous()
        assert x_flat.shape == (2, 3, 20)
        assert weight_flat.shape == (3, 6, 20)
        return torch.einsum("bik,iok->bok", x_flat, weight_flat)

    module.complex_mul_modes = fake_complex_mul_modes  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setenv("SPECTRALCONV_SUPA_EXTENSION_MODULE", module_name)

    x = torch.randn(2, 3, 4, 5, dtype=torch.cfloat)
    weight = torch.randn(3, 6, 4, 5, dtype=torch.cfloat)
    expected = complex_mul_modes(x, weight, backend="python")
    actual = complex_mul_modes(x, weight, backend="extension")
    torch.testing.assert_close(actual, expected)


def test_extension_adapter_uses_explicit_backward(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    module_name = "_fake_spectralconv_extension_with_backward"
    module = ModuleType(module_name)

    def fake_forward(
        x_flat: torch.Tensor,
        weight_flat: torch.Tensor
    ) -> torch.Tensor:
        return torch.einsum("bik,iok->bok", x_flat, weight_flat)

    def fake_backward(
        grad_output: torch.Tensor,
        x_flat: torch.Tensor,
        weight_flat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grad_x = torch.einsum(
            "bok,iok->bik",
            grad_output,
            weight_flat.conj()
        )
        grad_weight = torch.einsum(
            "bik,bok->iok",
            x_flat.conj(),
            grad_output
        )
        return grad_x, grad_weight

    module.complex_mul_modes_forward = fake_forward  # type: ignore[attr-defined]
    module.complex_mul_modes_backward = fake_backward  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setenv("SPECTRALCONV_SUPA_EXTENSION_MODULE", module_name)

    x = torch.randn(2, 3, 4, 5, dtype=torch.cfloat, requires_grad=True)
    weight = torch.randn(
        3,
        6,
        4,
        5,
        dtype=torch.cfloat,
        requires_grad=True
    )
    output = complex_mul_modes(x, weight, backend="extension")
    output.abs().square().mean().backward()
    extension_x_grad = x.grad.detach().clone()
    extension_weight_grad = weight.grad.detach().clone()

    x.grad = None
    weight.grad = None
    reference = complex_mul_modes(x, weight, backend="python")
    reference.abs().square().mean().backward()
    torch.testing.assert_close(extension_x_grad, x.grad)
    torch.testing.assert_close(extension_weight_grad, weight.grad)


def test_complex_mul_rejects_incompatible_inputs() -> None:
    x = torch.randn(1, 2, 3, dtype=torch.cfloat)
    wrong_channels = torch.randn(4, 5, 3, dtype=torch.cfloat)
    with pytest.raises(ValueError, match="输入通道不匹配"):
        complex_mul_modes(x, wrong_channels, backend="python")

    real_weight = torch.randn(2, 5, 3)
    with pytest.raises(TypeError, match="必须都是复数张量"):
        complex_mul_modes(x, real_weight, backend="python")


def test_cpu_runtime_measurement_helpers_are_safe_noops() -> None:
    device = torch.device("cpu")
    synchronize_device(device)
    reset_peak_memory_stats(device)
    assert max_memory_allocated(device) == 0


def test_privateuse_runtime_uses_torch_supa_api(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    fake_supa = SimpleNamespace(
        synchronize=lambda: calls.append("synchronize"),
        reset_peak_memory_stats=lambda: calls.append("reset"),
        max_memory_allocated=lambda: 4096
    )
    monkeypatch.setattr(torch, "supa", fake_supa, raising=False)
    device = torch.device("privateuseone")

    synchronize_device(device)
    reset_peak_memory_stats(device)
    assert max_memory_allocated(device) == 4096
    assert calls == ["synchronize", "reset"]


@pytest.mark.skipif(
    not _supa_extension_available(),
    reason="需要已构建的 SUPA 扩展和可用 BIREN GPU"
)
def test_real_supa_extension_forward_and_backward() -> None:
    torch.manual_seed(7)
    cpu_x = torch.randn(
        2,
        3,
        20,
        dtype=torch.cfloat,
        requires_grad=True
    )
    cpu_weight = torch.randn(
        3,
        5,
        20,
        dtype=torch.cfloat,
        requires_grad=True
    )
    cpu_gradient = torch.randn(2, 5, 20, dtype=torch.cfloat)
    reference = complex_mul_modes(cpu_x, cpu_weight, backend="python")
    reference.backward(cpu_gradient)

    device = select_device(strict_accelerator=True)
    supa_x = cpu_x.detach().to(device).requires_grad_(True)
    supa_weight = cpu_weight.detach().to(device).requires_grad_(True)
    actual = complex_mul_modes(supa_x, supa_weight, backend="extension")
    actual.backward(cpu_gradient.to(device))
    synchronize_device(device)

    torch.testing.assert_close(
        actual.cpu(),
        reference.detach(),
        rtol=1e-4,
        atol=1e-5
    )
    torch.testing.assert_close(
        supa_x.grad.cpu(),
        cpu_x.grad,
        rtol=1e-4,
        atol=1e-5
    )
    torch.testing.assert_close(
        supa_weight.grad.cpu(),
        cpu_weight.grad,
        rtol=1e-4,
        atol=1e-5
    )
