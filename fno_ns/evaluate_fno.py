"""Evaluate, benchmark and visualize a trained Navier-Stokes FNO."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fno_ns.data import ChannelGaussianNormalizer
from fno_ns.data import NavierStokesDataset
from fno_ns.data import load_ns_fields
from fno_ns.data import make_synthetic_ns_fields
from fno_ns.losses import relative_l2
from fno_ns.model import FNO2d
from fno_ns.runtime import benchmark_forward
from fno_ns.runtime import resolve_device


def parse_args() -> argparse.Namespace:
    """Parse evaluation command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("fno_ns/outputs/best_fno.pt")
    )
    parser.add_argument("--data", type=Path)
    parser.add_argument("--test-data", type=Path)
    parser.add_argument("--input-key", default="a")
    parser.add_argument("--target-key", default="u")
    parser.add_argument("--input-time", type=int, default=0)
    parser.add_argument("--target-time", type=int, default=-1)
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--backend", choices=("reference", "supa"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--figures", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("fno_ns/outputs"))
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Load the best checkpoint and emit metrics plus comparison figures."""
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = dict(checkpoint["model_config"])
    if args.backend is not None:
        model_config["backend"] = args.backend
    model = FNO2d(**model_config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    input_normalizer = ChannelGaussianNormalizer()
    target_normalizer = ChannelGaussianNormalizer()
    input_normalizer.load_state_dict(checkpoint["input_normalizer"])
    target_normalizer.load_state_dict(checkpoint["target_normalizer"])
    test_inputs, test_targets = _prepare_test_data(args)
    test_dataset = NavierStokesDataset(
        input_normalizer.encode(test_inputs),
        target_normalizer.encode(test_targets)
    )
    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda"
    )
    metric, examples = _run_evaluation(
        model,
        loader,
        target_normalizer,
        device,
        args.figures
    )
    benchmark_batch = next(iter(loader))[0].to(device)
    timing = benchmark_forward(
        lambda: model(benchmark_batch),
        device,
        benchmark_batch.shape[0],
        benchmark_batch.shape[-2],
        benchmark_batch.shape[-1],
        args.warmup,
        args.repeats
    )
    results: dict[str, Any] = {
        "relative_l2": metric,
        "samples": len(test_dataset),
        "batch_size": benchmark_batch.shape[0],
        "configured_batch_size": args.batch_size,
        "device": str(device),
        "backend": model_config["backend"],
        **timing
    }
    if device.type == "cuda":
        results["peak_memory_mb"] = (
            torch.cuda.max_memory_allocated(device) / 1024.0 / 1024.0
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "evaluation_metrics.json"
    metrics_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    if examples:
        _save_figures(examples, args.output_dir)
    print(json.dumps(results, ensure_ascii=False, indent=2))


def _prepare_test_data(
    args: argparse.Namespace
) -> tuple[torch.Tensor, torch.Tensor]:
    if args.smoke_test:
        inputs, targets = make_synthetic_ns_fields(
            min(args.n_test, 4),
            resolution=16,
            seed=1
        )
        return inputs, targets
    if args.data is None:
        if args.test_data is None:
            raise ValueError(
                "--data or --test-data is required unless --smoke-test is used"
            )
    if (
        args.data is not None
        and args.test_data is not None
        and args.data.resolve() == args.test_data.resolve()
    ):
        raise ValueError(
            "--test-data must be a different file from --data to avoid leakage"
        )
    source = args.test_data if args.test_data is not None else args.data
    assert source is not None
    inputs, targets = load_ns_fields(
        source,
        args.input_key,
        args.target_key,
        args.input_time,
        args.target_time
    )
    start = 0 if args.test_data is not None else args.n_train
    stop = start + args.n_test
    if inputs.shape[0] < stop:
        raise ValueError(f"Dataset has {inputs.shape[0]} samples, need {stop}")
    return inputs[start:stop], targets[start:stop]


def _run_evaluation(
    model: FNO2d,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    normalizer: ChannelGaussianNormalizer,
    device: torch.device,
    figure_count: int
) -> tuple[float, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    weighted_metric = 0.0
    sample_count = 0
    examples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            predictions = normalizer.decode(model(inputs))
            physical_targets = normalizer.decode(targets)
            batch_metric = relative_l2(predictions, physical_targets)
            weighted_metric += batch_metric.item() * inputs.shape[0]
            sample_count += inputs.shape[0]
            for prediction, target in zip(predictions, physical_targets):
                if len(examples) >= figure_count:
                    break
                error = (prediction - target).abs()
                examples.append(
                    (prediction.cpu(), target.cpu(), error.cpu())
                )
    return weighted_metric / sample_count, examples


def _save_figures(
    examples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    output_dir: Path
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; comparison figures were skipped")
        return
    for index, (prediction, target, error) in enumerate(examples):
        figure, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
        panels = (
            (target[0], "Ground truth", "viridis"),
            (prediction[0], "Prediction", "viridis"),
            (error[0], "Absolute error", "magma")
        )
        for axis, (image, title, color_map) in zip(axes, panels):
            plot = axis.imshow(image.numpy(), origin="lower", cmap=color_map)
            axis.set_title(title)
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            figure.colorbar(plot, ax=axis, fraction=0.046, pad=0.04)
        figure.savefig(output_dir / f"prediction_{index:03d}.png", dpi=180)
        plt.close(figure)


if __name__ == "__main__":
    main()
