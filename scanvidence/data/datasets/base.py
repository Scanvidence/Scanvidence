"""Base class for dataset adapters."""

from abc import ABC, abstractmethod
from typing import Any


class BaseDataset(ABC):
    """Abstract base class for medical imaging datasets.

    Each dataset adapter knows how to discover cases, resolve file paths,
    and return metadata for a specific benchmark (BraTS, ADNI, OASIS, etc.).

    Parameters
    ----------
    root : str
        Root directory where the dataset is stored.
    """

    def __init__(self, root: str):
        self.root = root

    @abstractmethod
    def discover(self) -> list[dict]:
        """Discover all cases in the dataset.

        Returns
        -------
        records : list of dict
            Each dict contains at minimum ``patient_id`` and ``path``.
        """
        pass

    @abstractmethod
    def load_case(self, record: dict) -> tuple[Any, dict]:
        """Load a single case by its record.

        Parameters
        ----------
        record : dict
            A record from :meth:`discover`.

        Returns
        -------
        array : numpy.ndarray
            Image volume.
        metadata : dict
            Case-level metadata.
        """
        pass
