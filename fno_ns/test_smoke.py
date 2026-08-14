"""CPU smoke tests for the FNO-NS subsystem."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fno_ns.data import ChannelGaussianNormalizer
from fno_ns.data import load_ns_fields
from fno_ns.data import make_synthetic_ns_fields
from fno_ns.losses import relative_l2
from fno_ns.model import FNO2d


class FnoSmokeTests(unittest.TestCase):
    def test_forward_backward(self) -> None:
        model = FNO2d(modes1=4, modes2=4, width=8, depth=4)
        inputs, targets = make_synthetic_ns_fields(2, resolution=16)
        predictions = model(inputs)
        self.assertEqual(predictions.shape, targets.shape)
        loss = relative_l2(predictions, targets)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_normalizer_round_trip(self) -> None:
        inputs, _ = make_synthetic_ns_fields(3, resolution=16)
        normalizer = ChannelGaussianNormalizer().fit(inputs)
        restored = normalizer.decode(normalizer.encode(inputs))
        torch.testing.assert_close(restored, inputs)

    def test_npz_trajectory_loading(self) -> None:
        trajectory = np.random.default_rng(0).normal(size=(4, 8, 8, 3))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ns.npz"
            np.savez(path, u=trajectory)
            inputs, targets = load_ns_fields(path)
        self.assertEqual(inputs.shape, (4, 1, 8, 8))
        self.assertEqual(targets.shape, (4, 1, 8, 8))
        torch.testing.assert_close(
            inputs[:, 0],
            torch.as_tensor(trajectory[..., 0]).float()
        )
        torch.testing.assert_close(
            targets[:, 0],
            torch.as_tensor(trajectory[..., -1]).float()
        )

    def test_matlab_v73_axis_order(self) -> None:
        try:
            import h5py
        except ImportError:
            self.skipTest("h5py is unavailable")
        trajectory = np.random.default_rng(1).normal(size=(4, 8, 8, 3))
        matlab_storage = trajectory.transpose(3, 2, 1, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ns_v73.mat"
            with h5py.File(path, "w") as archive:
                archive.create_dataset("u", data=matlab_storage)
            inputs, targets = load_ns_fields(path)
        self.assertEqual(inputs.shape, (4, 1, 8, 8))
        torch.testing.assert_close(
            inputs[:, 0],
            torch.as_tensor(trajectory[..., 0]).float()
        )
        torch.testing.assert_close(
            targets[:, 0],
            torch.as_tensor(trajectory[..., -1]).float()
        )


if __name__ == "__main__":
    unittest.main()
