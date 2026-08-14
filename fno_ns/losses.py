"""Metrics and objectives for FNO Navier-Stokes."""

import torch
import torch.nn as nn


def relative_l2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-12
) -> torch.Tensor:
    """Return mean sample-wise ``||prediction-target||_2 / ||target||_2``."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}"
        )
    residual_norm = torch.linalg.vector_norm(
        (prediction - target).reshape(prediction.shape[0], -1),
        dim=1
    )
    target_norm = torch.linalg.vector_norm(
        target.reshape(target.shape[0], -1),
        dim=1
    ).clamp_min(eps)
    return (residual_norm / target_norm).mean()


class RelativeL2Loss(nn.Module):
    """Module wrapper around :func:`relative_l2`."""

    def __init__(self, eps: float = 1e-12) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        return relative_l2(prediction, target, self.eps)
