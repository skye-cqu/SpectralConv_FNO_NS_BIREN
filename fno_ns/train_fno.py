"""Train a 2D FNO on Navier-Stokes vorticity fields."""

import argparse
import json
import sys
import time
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
from fno_ns.data import split_ns_fields
from fno_ns.losses import RelativeL2Loss
from fno_ns.model import FNO2d
from fno_ns.optim import SupaAdam
from fno_ns.runtime import resolve_device
from fno_ns.runtime import seed_everything
from fno_ns.runtime import synchronize


def parse_args() -> argparse.Namespace:
    """Parse training command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--test-data", type=Path)
    parser.add_argument("--input-key", default="a")
    parser.add_argument("--target-key", default="u")
    parser.add_argument("--input-time", type=int, default=0)
    parser.add_argument("--target-time", type=int, default=-1)
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-validation", type=int, default=100)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--modes1", type=int, default=12)
    parser.add_argument("--modes2", type=int, default=12)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--backend", choices=("reference", "supa"), default="reference")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("fno_ns/outputs"))
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Train, evaluate each epoch and save a reproducible best checkpoint."""
    args = parse_args()
    _apply_smoke_defaults(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    (
        train_inputs,
        train_targets,
        validation_inputs,
        validation_targets,
        test_inputs,
        test_targets
    ) = _prepare_data(args)
    input_normalizer = ChannelGaussianNormalizer().fit(train_inputs)
    target_normalizer = ChannelGaussianNormalizer().fit(train_targets)
    train_dataset = NavierStokesDataset(
        input_normalizer.encode(train_inputs),
        target_normalizer.encode(train_targets)
    )
    validation_dataset = NavierStokesDataset(
        input_normalizer.encode(validation_inputs),
        target_normalizer.encode(validation_targets)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda"
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda"
    )
    model = FNO2d(
        in_channels=train_inputs.shape[1],
        out_channels=train_targets.shape[1],
        modes1=args.modes1,
        modes2=args.modes2,
        width=args.width,
        depth=args.depth,
        backend=args.backend
    ).to(device)
    if device.type in {"privateuseone", "supa"}:
        optimizer: torch.optim.Optimizer = SupaAdam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            foreach=False
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )
    objective = RelativeL2Loss()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train_metrics.jsonl"
    checkpoint_path = args.output_dir / "best_fno.pt"
    best_validation_l2 = float("inf")
    total_steps = 0
    log_path.write_text("", encoding="utf-8")
    print(
        f"device={device} backend={args.backend} "
        f"train={len(train_dataset)} validation={len(validation_dataset)} "
        f"test={test_inputs.shape[0]}"
    )
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_l2, steps = _train_epoch(
            model,
            train_loader,
            optimizer,
            objective,
            target_normalizer,
            device
        )
        total_steps += steps
        validation_l2 = _evaluate(
            model,
            validation_loader,
            objective,
            target_normalizer,
            device
        )
        scheduler.step()
        synchronize(device)
        record = {
            "epoch": epoch,
            "step": total_steps,
            "train_relative_l2": train_l2,
            "validation_relative_l2": validation_l2,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - epoch_start
        }
        _append_json(log_path, record)
        if validation_l2 < best_validation_l2:
            best_validation_l2 = validation_l2
            _save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                input_normalizer,
                target_normalizer,
                args,
                epoch,
                total_steps,
                best_validation_l2
            )
        if epoch == 1 or epoch % args.log_interval == 0 or epoch == args.epochs:
            print(json.dumps(record, ensure_ascii=False))
    best_checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(best_checkpoint["model_state"])
    test_dataset = NavierStokesDataset(
        input_normalizer.encode(test_inputs),
        target_normalizer.encode(test_targets)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda"
    )
    final_test_l2 = _evaluate(
        model,
        test_loader,
        objective,
        target_normalizer,
        device
    )
    final_metrics = {
        "selected_epoch": best_checkpoint["epoch"],
        "best_validation_relative_l2": best_validation_l2,
        "final_test_relative_l2": final_test_l2,
        "test_samples": len(test_dataset)
    }
    (args.output_dir / "final_test_metrics.json").write_text(
        json.dumps(final_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(json.dumps(final_metrics, ensure_ascii=False))
    print(f"checkpoint={checkpoint_path}")


def _train_epoch(
    model: FNO2d,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    objective: RelativeL2Loss,
    normalizer: ChannelGaussianNormalizer,
    device: torch.device
) -> tuple[float, int]:
    model.train()
    metric_sum = 0.0
    sample_count = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        loss = objective(
            normalizer.decode(predictions),
            normalizer.decode(targets)
        )
        loss.backward()
        optimizer.step()
        batch_size = inputs.shape[0]
        metric_sum += loss.detach().item() * batch_size
        sample_count += batch_size
    return metric_sum / sample_count, len(loader)


def _evaluate(
    model: FNO2d,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    objective: RelativeL2Loss,
    normalizer: ChannelGaussianNormalizer,
    device: torch.device
) -> float:
    model.eval()
    metric_sum = 0.0
    sample_count = 0
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            predictions = model(inputs)
            loss = objective(
                normalizer.decode(predictions),
                normalizer.decode(targets)
            )
            batch_size = inputs.shape[0]
            metric_sum += loss.item() * batch_size
            sample_count += batch_size
    return metric_sum / sample_count


def _prepare_data(
    args: argparse.Namespace
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor
]:
    if args.smoke_test:
        inputs, targets = make_synthetic_ns_fields(
            args.n_train + args.n_test,
            resolution=16,
            seed=args.seed
        )
        train_inputs, train_targets, test_inputs, test_targets = split_ns_fields(
            inputs,
            targets,
            args.n_train,
            args.n_test
        )
        return _split_validation(
            train_inputs,
            train_targets,
            test_inputs,
            test_targets,
            args.n_validation
        )
    if args.data is None:
        raise ValueError("--data is required unless --smoke-test is used")
    inputs, targets = load_ns_fields(
        args.data,
        args.input_key,
        args.target_key,
        args.input_time,
        args.target_time
    )
    if args.test_data is None:
        train_inputs, train_targets, test_inputs, test_targets = split_ns_fields(
            inputs,
            targets,
            args.n_train,
            args.n_test
        )
        return _split_validation(
            train_inputs,
            train_targets,
            test_inputs,
            test_targets,
            args.n_validation
        )
    if args.data.resolve() == args.test_data.resolve():
        raise ValueError(
            "--test-data must be a different file from --data to avoid leakage"
        )
    test_inputs, test_targets = load_ns_fields(
        args.test_data,
        args.input_key,
        args.target_key,
        args.input_time,
        args.target_time
    )
    if inputs.shape[0] < args.n_train or test_inputs.shape[0] < args.n_test:
        raise ValueError("Separate train/test files do not contain enough samples")
    return _split_validation(
        inputs[:args.n_train],
        targets[:args.n_train],
        test_inputs[:args.n_test],
        test_targets[:args.n_test],
        args.n_validation
    )


def _split_validation(
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    test_inputs: torch.Tensor,
    test_targets: torch.Tensor,
    n_validation: int
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor
]:
    if n_validation <= 0 or n_validation >= train_inputs.shape[0]:
        raise ValueError("n_validation must be between zero and n_train")
    split = train_inputs.shape[0] - n_validation
    return (
        train_inputs[:split],
        train_targets[:split],
        train_inputs[split:],
        train_targets[split:],
        test_inputs,
        test_targets
    )


def _apply_smoke_defaults(args: argparse.Namespace) -> None:
    if not args.smoke_test:
        return
    args.n_train = min(args.n_train, 16)
    args.n_test = min(args.n_test, 4)
    args.n_validation = min(args.n_validation, 4)
    args.batch_size = min(args.batch_size, 4)
    args.epochs = 1
    args.width = min(args.width, 8)
    args.modes1 = min(args.modes1, 4)
    args.modes2 = min(args.modes2, 4)
    args.workers = 0


def _append_json(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _save_checkpoint(
    path: Path,
    model: FNO2d,
    optimizer: torch.optim.Optimizer,
    input_normalizer: ChannelGaussianNormalizer,
    target_normalizer: ChannelGaussianNormalizer,
    args: argparse.Namespace,
    epoch: int,
    step: int,
    best_validation_l2: float
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": model.config(),
            "input_normalizer": input_normalizer.state_dict(),
            "target_normalizer": target_normalizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "best_validation_relative_l2": best_validation_l2,
            "train_args": vars(args)
        },
        path
    )


if __name__ == "__main__":
    main()
