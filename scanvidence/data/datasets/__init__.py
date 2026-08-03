"""Dataset adapters for specific medical imaging benchmarks."""

from .ADNIDataset import ADNIDataset
from .base import BaseDataset
from .BraTSDataset import BraTSDataset
from .OASISDataset import OASISDataset

__all__ = ["BaseDataset", "ADNIDataset", "BraTSDataset", "OASISDataset"]
