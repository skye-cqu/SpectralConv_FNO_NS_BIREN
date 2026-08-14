"""Validate the complete SpectralConv2d forward and backward on BIREN."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.reference import SpectralConv2d
from src.supa import SupaSpectralConv2d
from src.supa import probe_extension
from src.supa import select_device
from src.supa import synchronize_device


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Return ||actual - expected||_2 / max(||expected||_2, eps)."""
    difference = torch.linalg.vector_norm(actual - expected)
    denominator = torch.linalg.vector_norm(expected)
    epsilon = torch.finfo(expected.real.dtype).eps
    return float((difference / denominator.clamp_min(epsilon)).item())


def error_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor
) -> dict[str, float]:
    """Calculate portable correctness metrics on CPU."""
    actual_cpu = actual.detach().cpu()
    expected_cpu = expected.detach().cpu()
    return {
        "relative_l2": relative_l2(actual_cpu, expected_cpu),
        "max_absolute": float((actual_cpu - expected_cpu).abs().max().item())
    }


def copy_weights(
    source: SpectralConv2d,
    target: SupaSpectralConv2d
) -> None:
    """Copy the two complex Fourier weights between equivalent modules."""
    with torch.no_grad():
        target.weights1.copy_(source.weights1.to(target.weights1.device))
        target.weights2.copy_(source.weights2.to(target.weights2.device))


def validate(
    batch_size: int,
    in_channels: int,
    out_channels: int,
    resolution: int,
    modes: int,
    seed: int,
    tolerance: float
) -> dict[str, object]:
    """Run deterministic end-to-end forward and backward comparisons."""
    extension = probe_extension()
    if not extension.available:
        raise RuntimeError(extension.detail)
    device = select_device(strict_accelerator=True)

    torch.manual_seed(seed)
    reference = SpectralConv2d(
        in_channels,
        out_channels,
        modes,
        modes
    )
    custom = SupaSpectralConv2d(
        in_channels,
        out_channels,
        modes,
        modes,
        backend="extension"
    ).to(device)
    copy_weights(reference, custom)

    reference_input = torch.randn(
        batch_size,
        in_channels,
        resolution,
        resolution,
        requires_grad=True
    )
    custom_input = (
        reference_input.detach().to(device).requires_grad_(True)
    )
    output_gradient = torch.randn(
        batch_size,
        out_channels,
        resolution,
        resolution
    )

    expected = reference(reference_input)
    actual = custom(custom_input)
    expected.backward(output_gradient)
    actual.backward(output_gradient.to(device))
    synchronize_device(device)

    metrics = {
        "forward": error_metrics(actual, expected),
        "input_gradient": error_metrics(
            custom_input.grad,
            reference_input.grad
        ),
        "weights1_gradient": error_metrics(
            custom.weights1.grad,
            reference.weights1.grad
        ),
        "weights2_gradient": error_metrics(
            custom.weights2.grad,
            reference.weights2.grad
        )
    }
    passed = all(
        item["relative_l2"] <= tolerance
        for item in metrics.values()
    )
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "device_name": torch.supa.get_device_name(0),
        "dtype": str(reference_input.dtype),
        "shape": {
            "batch_size": batch_size,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "height": resolution,
            "width": resolution,
            "modes1": modes,
            "modes2": modes
        },
        "extension": extension.detail,
        "tolerance": tolerance,
        "metrics": metrics,
        "passed": passed
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--out-channels", type=int, default=5)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("results/spectralconv_correctness_biren.json")
    )
    return parser.parse_args()


def main() -> int:
    """Run validation, save JSON and return a CI-friendly exit status."""
    args = parse_args()
    result = validate(
        batch_size=args.batch_size,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        resolution=args.resolution,
        modes=args.modes,
        seed=args.seed,
        tolerance=args.tolerance
    )
    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(output + "\n", encoding="utf-8")
    print(f"结果已写入: {args.json}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
