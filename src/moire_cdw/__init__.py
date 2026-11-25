"""Moiré diffraction: detection, fitting, then independent calibration.

Coordinates are always (row, col) in pixels unless explicitly calibrated.
Import fitting or tri explicitly; package import does not load Torch/QuantEM.
"""
__version__ = "0.1.0"
__all__ = ["DiffractionMoire"]

def __getattr__(name):
    if name == "DiffractionMoire":
        from .superlattice import DiffractionMoire
        return DiffractionMoire
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
