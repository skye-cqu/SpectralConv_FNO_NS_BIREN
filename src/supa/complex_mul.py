"""Complex mode multiplication shared by fallback and SUPA backends."""

from __future__ import annotations

import torch

from .backend import resolve_backend
from .extension import (
    ExtensionBackwardOperator,
    ExtensionOperator,
    ExtensionUnavailableError,
    load_extension_operators
)


def _validate_inputs(
    x_ft: torch.Tensor,
    weight: torch.Tensor
) -> tuple[int, int, int, tuple[int, ...]]:
    if x_ft.ndim < 3:
        raise ValueError(
            f"x_ft 必须为 [B, C_in, ...modes]，实际 shape={tuple(x_ft.shape)}"
        )
    if weight.ndim != x_ft.ndim:
        raise ValueError(
            "weight 必须为 [C_in, C_out, ...modes]，且维数与 x_ft 相同"
        )
    if not x_ft.is_complex() or not weight.is_complex():
        raise TypeError("x_ft 和 weight 必须都是复数张量")
    if x_ft.device != weight.device:
        raise ValueError(
            f"x_ft 与 weight 必须位于同一设备，实际为 "
            f"{x_ft.device} 和 {weight.device}"
        )
    if x_ft.dtype != weight.dtype:
        raise TypeError(
            f"x_ft 与 weight dtype 必须一致，实际为 "
            f"{x_ft.dtype} 和 {weight.dtype}"
        )

    batch_size, in_channels = x_ft.shape[:2]
    weight_in_channels, out_channels = weight.shape[:2]
    mode_shape = tuple(x_ft.shape[2:])
    if weight_in_channels != in_channels:
        raise ValueError(
            f"输入通道不匹配: x_ft={in_channels}, weight={weight_in_channels}"
        )
    if tuple(weight.shape[2:]) != mode_shape:
        raise ValueError(
            f"频率模态 shape 不匹配: x_ft={mode_shape}, "
            f"weight={tuple(weight.shape[2:])}"
        )
    return batch_size, in_channels, out_channels, mode_shape


def complex_mul_modes_python(
    x_ft: torch.Tensor,
    weight: torch.Tensor
) -> torch.Tensor:
    """Multiply complex Fourier modes using differentiable PyTorch ops.

    Shapes are ``[B, C_in, ...modes]`` and
    ``[C_in, C_out, ...modes]``.  All mode axes are independent; reduction is
    only over ``C_in``.
    """
    _validate_inputs(x_ft, weight)
    return torch.einsum("bi...,io...->bo...", x_ft, weight)


class _ComplexMulModesExtension(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: object,
        x_flat: torch.Tensor,
        weight_flat: torch.Tensor,
        forward_operator: ExtensionOperator,
        backward_operator: ExtensionBackwardOperator
    ) -> torch.Tensor:
        ctx.save_for_backward(x_flat, weight_flat)  # type: ignore[attr-defined]
        ctx.backward_operator = backward_operator  # type: ignore[attr-defined]
        return forward_operator(x_flat, weight_flat)

    @staticmethod
    def backward(
        ctx: object,
        grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        x_flat, weight_flat = ctx.saved_tensors  # type: ignore[attr-defined]
        backward_operator = ctx.backward_operator  # type: ignore[attr-defined]
        grad_x, grad_weight = backward_operator(
            grad_output.contiguous(),
            x_flat,
            weight_flat
        )
        return grad_x, grad_weight, None, None


def _run_extension(
    x_ft: torch.Tensor,
    weight: torch.Tensor
) -> torch.Tensor:
    batch_size, _, out_channels, mode_shape = _validate_inputs(x_ft, weight)
    mode_count = weight.numel() // (weight.shape[0] * out_channels)
    x_flat = x_ft.reshape(batch_size, x_ft.shape[1], mode_count).contiguous()
    weight_flat = weight.reshape(
        weight.shape[0],
        out_channels,
        mode_count
    ).contiguous()

    operators = load_extension_operators()
    if operators.backward is None:
        output = operators.forward(x_flat, weight_flat)
    else:
        output = _ComplexMulModesExtension.apply(
            x_flat,
            weight_flat,
            operators.forward,
            operators.backward
        )
    expected_shape = (batch_size, out_channels, mode_count)
    if not isinstance(output, torch.Tensor):
        raise RuntimeError(
            "SUPA complex_mul_modes 必须返回 torch.Tensor，实际返回 "
            f"{type(output).__name__}"
        )
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(
            f"SUPA complex_mul_modes 输出 shape 错误: "
            f"期望 {expected_shape}，实际 {tuple(output.shape)}"
        )
    if output.device != x_ft.device or output.dtype != x_ft.dtype:
        raise RuntimeError(
            "SUPA complex_mul_modes 必须保持输入 device/dtype，实际输出为 "
            f"{output.device}/{output.dtype}"
        )
    return output.reshape(batch_size, out_channels, *mode_shape)


def complex_mul_modes(
    x_ft: torch.Tensor,
    weight: torch.Tensor,
    backend: str = "auto"
) -> torch.Tensor:
    """Dispatch complex mode multiplication to Python or the SUPA extension.

    ``auto`` uses PyTorch on ordinary CPU/CUDA devices, but requires the
    extension on a SUPA device.  ``extension`` is always strict and never
    falls back after an extension import or execution failure.
    """
    resolved_backend = resolve_backend(backend, x_ft.device)
    if resolved_backend == "python":
        return complex_mul_modes_python(x_ft, weight)
    try:
        return _run_extension(x_ft, weight)
    except ExtensionUnavailableError:
        raise
    except Exception as error:
        raise RuntimeError(
            f"SUPA complex_mul_modes 执行失败: {type(error).__name__}: {error}"
        ) from error
