"""Optimizers compatible with the BIREN SUPA PyTorch backend."""

import math
from collections.abc import Callable
from collections.abc import Iterable
from typing import Any

import torch
from torch.optim import Optimizer


class SupaAdam(Optimizer):
    """Adam implemented without ``lerp_`` for the SUPA PrivateUse1 backend.

    Complex parameters follow PyTorch Adam semantics: the real and imaginary
    components maintain independent first and second moments.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], torch.Tensor] | None = None
    ) -> torch.Tensor | None:
        """Perform one Adam update using SUPA-supported tensor primitives."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            learning_rate = float(group["lr"])
            beta1, beta2 = group["betas"]
            epsilon = float(group["eps"])
            weight_decay = float(group["weight_decay"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("SupaAdam does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        parameter,
                        memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        parameter,
                        memory_format=torch.preserve_format
                    )
                state["step"] += 1
                step = int(state["step"])
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                if weight_decay != 0.0:
                    gradient = gradient.add(
                        parameter,
                        alpha=weight_decay
                    )
                parameter_view = parameter
                if torch.is_complex(parameter):
                    parameter_view = torch.view_as_real(parameter)
                    gradient = torch.view_as_real(gradient)
                    exp_avg = torch.view_as_real(exp_avg)
                    exp_avg_sq = torch.view_as_real(exp_avg_sq)
                exp_avg.mul_(beta1)
                exp_avg.add_(gradient, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2)
                squared_gradient = gradient * gradient
                exp_avg_sq.add_(
                    squared_gradient,
                    alpha=1.0 - beta2
                )
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                step_size = learning_rate / bias_correction1
                denominator = exp_avg_sq.sqrt()
                denominator.div_(math.sqrt(bias_correction2))
                denominator.add_(epsilon)
                update = exp_avg / denominator
                parameter_view.add_(update, alpha=-step_size)
        return loss
