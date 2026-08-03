"""Base class for radiomics feature extractors."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseExtractor(ABC):
    """Abstract base class for radiomics feature extraction.

    Parameters
    ----------
    config : dict
        Extraction configuration (bin width, feature classes, etc.).
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    @abstractmethod
    def extract(self, image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
        """Extract radiomics features from an image-mask pair.

        Parameters
        ----------
        image : numpy.ndarray
            The image volume.
        mask : numpy.ndarray
            The binary segmentation mask.

        Returns
        -------
        features : dict[str, float]
            Feature name -> value mapping.
        """
        pass
