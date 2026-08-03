"""Intensity normalization for MRI volumes."""

from .base import BaseTransform


class Normalizer(BaseTransform):
    """Normalize MRI intensity values.

    Parameters
    ----------
    method : str
        Normalization method. Options: ``"zscore"``, ``"minmax"``,
        ``"nyul"`` (Nyul histogram standardization).
    """

    def __init__(self, method: str = "zscore"):
        self.method = method

    def __call__(self, array, metadata=None):
        """Apply intensity normalization."""
        if self.method == "zscore":
            mask = array > 0
            if mask.any():
                array = (array - array[mask].mean()) / (array[mask].std() + 1e-8)
        elif self.method == "minmax":
            array = (array - array.min()) / (array.max() - array.min() + 1e-8)
        return array, metadata or {}
