"""Base class for data loaders."""

from abc import ABC, abstractmethod
from typing import Any


class BaseLoader(ABC):
    """Abstract base class for medical image format loaders.

    Subclasses implement format-specific loading (NIfTI, DICOM, etc.)
    and return a standardised numpy array + metadata dict.

    Parameters
    ----------
    data_root : str
        Root directory containing scan files.
    """

    def __init__(self, data_root: str):
        self.data_root = data_root

    @abstractmethod
    def load(self, path: str) -> tuple[Any, dict]:
        """Load a scan file and return (array, metadata).

        Parameters
        ----------
        path : str
            Path to the scan file, relative to ``data_root``.

        Returns
        -------
        array : numpy.ndarray
            The loaded image volume.
        metadata : dict
            File-level metadata (spacing, orientation, etc.).
        """
        pass

    @abstractmethod
    def validate(self, path: str) -> bool:
        """Check whether a file is a valid scan of this format."""
        pass
