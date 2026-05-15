"""Dataset adapters and manifest loaders."""

from .cfc_adapter import CFCAdapter
from .nz_thermal_adapter import NzThermalAdapter

__all__ = ["CFCAdapter", "NzThermalAdapter"]
