"""BIREN SUPA spectral-convolution interface and Python fallback."""

from .backend import BackendName, is_supa_device, resolve_backend
from .complex_mul import complex_mul_modes, complex_mul_modes_python
from .extension import (
    ExtensionStatus,
    ExtensionUnavailableError,
    clear_extension_cache,
    probe_extension
)
from .fft import (
    irfft2_compat,
    rfft2_compat,
    sequential_irfft2,
    sequential_rfft2
)
from .runtime import (
    max_memory_allocated,
    reset_peak_memory_stats,
    select_device,
    synchronize_device
)
from .spectral_conv2d import SupaSpectralConv2d

__all__ = [
    "BackendName",
    "ExtensionStatus",
    "ExtensionUnavailableError",
    "SupaSpectralConv2d",
    "clear_extension_cache",
    "complex_mul_modes",
    "complex_mul_modes_python",
    "is_supa_device",
    "irfft2_compat",
    "max_memory_allocated",
    "probe_extension",
    "rfft2_compat",
    "reset_peak_memory_stats",
    "resolve_backend",
    "select_device",
    "sequential_irfft2",
    "sequential_rfft2",
    "synchronize_device"
]
