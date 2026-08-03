"""Scanvidence: Explainable, uncertainty-aware medical imaging detection.

Unified platform for brain tumor and Alzheimer's detection with
XAI, calibration, radiomics, and quantum feature selection.

See docs/ARCHITECTURE.md for the pipeline map and README.md for setup.
"""

from importlib.metadata import version

from .global_vars import config, logger

__all__ = ["config", "logger"]
__version__ = version("scanvidence")
