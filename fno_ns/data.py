"""Navier-Stokes data loading, normalization and deterministic splitting."""

from pathlib import Path
from typing import Any
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import Dataset


class ChannelGaussianNormalizer:
    """Normalize each channel using statistics from the training split only."""

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

    def fit(self, values: torch.Tensor) -> "ChannelGaussianNormalizer":
        """Fit channel statistics for a tensor shaped ``[N, C, H, W]``."""
        _validate_fields(values, "values")
        self.mean = values.mean(dim=(0, 2, 3), keepdim=True)
        self.std = values.std(dim=(0, 2, 3), keepdim=True, unbiased=False)
        self.std = self.std.clamp_min(self.eps)
        return self

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        """Return normalized values without mutating the input."""
        self._check_fitted()
        assert self.mean is not None
        assert self.std is not None
        return (values - self.mean.to(values.device)) / self.std.to(values.device)

    def decode(self, values: torch.Tensor) -> torch.Tensor:
        """Restore values to their physical scale."""
        self._check_fitted()
        assert self.mean is not None
        assert self.std is not None
        return values * self.std.to(values.device) + self.mean.to(values.device)

    def state_dict(self) -> dict[str, Any]:
        """Serialize statistics with a checkpoint."""
        self._check_fitted()
        return {
            "eps": self.eps,
            "mean": self.mean,
            "std": self.std
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore statistics from a checkpoint."""
        self.eps = float(state["eps"])
        self.mean = torch.as_tensor(state["mean"])
        self.std = torch.as_tensor(state["std"])

    def _check_fitted(self) -> None:
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer must be fitted before use")


class NavierStokesDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """In-memory pairs of initial and target vorticity fields."""

    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        _validate_fields(inputs, "inputs")
        _validate_fields(targets, "targets")
        if inputs.shape != targets.shape:
            raise ValueError(
                "inputs and targets must have identical shapes, got "
                f"{tuple(inputs.shape)} and {tuple(targets.shape)}"
            )
        self.inputs = inputs.contiguous().float()
        self.targets = targets.contiguous().float()

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index]


def load_ns_fields(
    path: str | Path,
    input_key: str = "a",
    target_key: str = "u",
    input_time: int = 0,
    target_time: int = -1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load NS fields from ``.pt``, ``.npz`` or MATLAB ``.mat`` files.

    Arrays may be ``[N,H,W]``, ``[N,H,W,T]``, ``[N,T,H,W]`` or already
    channel-first. When only a trajectory named by ``target_key`` is present,
    the selected initial and target time slices form the training pair.
    """
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"NS data file does not exist: {data_path}")
    suffix = data_path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        raw = torch.load(data_path, map_location="cpu", weights_only=False)
    elif suffix == ".npz":
        with np.load(data_path, allow_pickle=False) as archive:
            raw = {key: archive[key] for key in archive.files}
    elif suffix == ".mat":
        raw = _load_mat(data_path)
    else:
        raise ValueError(f"Unsupported NS data format: {suffix}")
    return _extract_pairs(raw, input_key, target_key, input_time, target_time)


def split_ns_fields(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    n_train: int = 1000,
    n_test: int = 200
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Take a reproducible official-order train/test split."""
    if n_train <= 0 or n_test <= 0:
        raise ValueError("n_train and n_test must be positive")
    required = n_train + n_test
    if inputs.shape[0] < required:
        raise ValueError(
            f"Dataset has {inputs.shape[0]} samples, but {required} are required"
        )
    return (
        inputs[:n_train],
        targets[:n_train],
        inputs[n_train:required],
        targets[n_train:required]
    )


def make_synthetic_ns_fields(
    samples: int,
    resolution: int = 64,
    seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create smooth deterministic fields for smoke tests, not model scoring."""
    if samples <= 0 or resolution <= 0:
        raise ValueError("samples and resolution must be positive")
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(samples, 1, resolution, resolution, generator=generator)
    spectrum = torch.fft.rfft2(noise)
    frequencies_y = torch.fft.fftfreq(resolution).reshape(1, 1, -1, 1)
    frequencies_x = torch.fft.rfftfreq(resolution).reshape(1, 1, 1, -1)
    radius_squared = frequencies_x.square() + frequencies_y.square()
    smooth_filter = torch.exp(-80.0 * radius_squared)
    initial = torch.fft.irfft2(spectrum * smooth_filter, s=(resolution, resolution))
    shifted = torch.roll(initial, shifts=(1, -1), dims=(-2, -1))
    target = 0.92 * initial + 0.08 * shifted
    return initial.float(), target.float()


def _extract_pairs(
    raw: Any,
    input_key: str,
    target_key: str,
    input_time: int,
    target_time: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(raw, (tuple, list)) and len(raw) == 2:
        return _as_field(raw[0]), _as_field(raw[1])
    if not isinstance(raw, Mapping):
        raise TypeError("Data must be a mapping or an (inputs, targets) pair")
    if input_key in raw and target_key in raw:
        inputs_raw = torch.as_tensor(raw[input_key])
        targets_raw = torch.as_tensor(raw[target_key])
        if inputs_raw.ndim == 4 and inputs_raw.shape[1] != 1:
            inputs_raw = _select_time(inputs_raw, input_time)
        if targets_raw.ndim == 4 and targets_raw.shape[1] != 1:
            targets_raw = _select_time(targets_raw, target_time)
        return _as_field(inputs_raw), _as_field(targets_raw)
    if target_key in raw:
        trajectory = torch.as_tensor(raw[target_key])
        return (
            _as_field(_select_time(trajectory, input_time)),
            _as_field(_select_time(trajectory, target_time))
        )
    available = ", ".join(sorted(str(key) for key in raw))
    raise KeyError(
        f"Could not find '{input_key}'/'{target_key}' in data. "
        f"Available keys: {available}"
    )


def _select_time(values: torch.Tensor, time_index: int) -> torch.Tensor:
    if values.ndim != 4:
        raise ValueError(
            "A trajectory must be [N,H,W,T] or [N,T,H,W], got "
            f"{tuple(values.shape)}"
        )
    if values.shape[1] == values.shape[2]:
        return values[..., time_index]
    if values.shape[2] == values.shape[3]:
        return values[:, time_index]
    raise ValueError(f"Cannot infer time axis from shape {tuple(values.shape)}")


def _as_field(values: Any) -> torch.Tensor:
    tensor = torch.as_tensor(values)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
    elif tensor.ndim == 4 and tensor.shape[1] != 1 and tensor.shape[-1] == 1:
        tensor = tensor.permute(0, 3, 1, 2)
    _validate_fields(tensor, "field")
    return tensor.float().contiguous()


def _validate_fields(values: torch.Tensor, name: str) -> None:
    if values.ndim != 4:
        raise ValueError(f"{name} must be [N,C,H,W], got {tuple(values.shape)}")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"{name} cannot have empty batch or channel dimensions")
    if values.shape[-2] <= 1 or values.shape[-1] <= 1:
        raise ValueError(f"{name} spatial dimensions must exceed one")
    if not values.is_floating_point():
        raise TypeError(f"{name} must contain floating-point values")


def _load_mat(path: Path) -> Mapping[str, Any]:
    try:
        import scipy.io
    except ImportError as exc:
        raise ImportError("Reading .mat files requires scipy") from exc
    try:
        return scipy.io.loadmat(path)
    except (NotImplementedError, ValueError):
        return _load_hdf5_mat(path)


def _load_hdf5_mat(path: Path) -> Mapping[str, Any]:
    """Load MATLAB v7.3 arrays and restore MATLAB-to-NumPy axis order."""
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "Reading MATLAB v7.3 files requires h5py"
        ) from exc
    arrays: dict[str, Any] = {}
    with h5py.File(path, "r") as archive:
        for key, value in archive.items():
            if not isinstance(value, h5py.Dataset):
                continue
            array = np.asarray(value)
            if array.ndim > 1:
                array = array.transpose(tuple(reversed(range(array.ndim))))
            arrays[key] = array
    return arrays
