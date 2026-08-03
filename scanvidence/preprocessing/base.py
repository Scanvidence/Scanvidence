"""Base class for preprocessing transforms."""

from abc import ABC, abstractmethod
from typing import Any


class BaseTransform(ABC):
    """Abstract base class for preprocessing transforms.

    All transforms follow a consistent interface: they accept a numpy
    array (and optional metadata) and return a transformed array.
    """

    @abstractmethod
    def __call__(self, array: Any, metadata: dict | None = None) -> tuple[Any, dict]:
        """Apply the transform.

        Parameters
        ----------
        array : numpy.ndarray
            Input image volume.
        metadata : dict or None
            Associated metadata (spacing, orientation, etc.).

        Returns
        -------
        array : numpy.ndarray
            Transformed image volume.
        metadata : dict
            Updated metadata.
        """
        pass
