"""Skull stripping for brain MRI volumes."""

from .base import BaseTransform


class SkullStripper(BaseTransform):
    """Remove non-brain tissue from MRI volumes.

    Parameters
    ----------
    method : str
        Stripping algorithm. Options: ``"bet"`` (FSL BET via SimpleITK),
        ``"threshold"`` (simple intensity thresholding).
    """

    def __init__(self, method: str = "threshold"):
        self.method = method

    def __call__(self, array, metadata=None):
        """Apply skull stripping."""
        # Stub — implementation depends on chosen method
        return array, metadata or {}
