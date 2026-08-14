"""FNO Navier-Stokes training and evaluation package."""

from fno_ns.data import ChannelGaussianNormalizer
from fno_ns.data import NavierStokesDataset
from fno_ns.losses import RelativeL2Loss
from fno_ns.model import FNO2d

__all__ = [
    "ChannelGaussianNormalizer",
    "FNO2d",
    "NavierStokesDataset",
    "RelativeL2Loss"
]
