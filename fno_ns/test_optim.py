"""CPU numerical tests for the SUPA-compatible Adam optimizer."""

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fno_ns.optim import SupaAdam


class SupaAdamTests(unittest.TestCase):
    def test_real_parameters_match_pytorch_adam(self) -> None:
        initial = torch.tensor(
            [[0.3, -0.8], [1.2, -1.7]],
            dtype=torch.float64
        )
        expected = torch.nn.Parameter(initial.clone())
        actual = torch.nn.Parameter(initial.clone())
        reference = torch.optim.Adam(
            [expected],
            lr=2e-3,
            betas=(0.8, 0.95),
            eps=1e-9,
            weight_decay=0.03,
            foreach=False
        )
        optimizer = SupaAdam(
            [actual],
            lr=2e-3,
            betas=(0.8, 0.95),
            eps=1e-9,
            weight_decay=0.03
        )
        for step in range(1, 8):
            gradient = torch.tensor(
                [[0.1 * step, -0.2], [0.05, -0.03 * step]],
                dtype=torch.float64
            )
            expected.grad = gradient.clone()
            actual.grad = gradient.clone()
            reference.step()
            optimizer.step()
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_complex_parameters_match_pytorch_adam(self) -> None:
        initial = torch.tensor(
            [0.4 + 0.7j, -1.1 + 0.2j, 0.3 - 0.6j],
            dtype=torch.complex128
        )
        expected = torch.nn.Parameter(initial.clone())
        actual = torch.nn.Parameter(initial.clone())
        reference = torch.optim.Adam(
            [expected],
            lr=4e-3,
            betas=(0.85, 0.97),
            eps=1e-10,
            weight_decay=0.02,
            foreach=False
        )
        optimizer = SupaAdam(
            [actual],
            lr=4e-3,
            betas=(0.85, 0.97),
            eps=1e-10,
            weight_decay=0.02
        )
        for step in range(1, 6):
            gradient = torch.tensor(
                [
                    0.03 * step - 0.1j,
                    -0.2 + 0.04j * step,
                    0.07 - 0.02j * step
                ],
                dtype=torch.complex128
            )
            expected.grad = gradient.clone()
            actual.grad = gradient.clone()
            reference.step()
            optimizer.step()
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
