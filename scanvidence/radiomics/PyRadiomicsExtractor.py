"""PyRadiomics-based feature extractor."""

import numpy as np

from .base import BaseExtractor


class PyRadiomicsExtractor(BaseExtractor):
    """Extract radiomics features using the pyradiomics library.

    Parameters
    ----------
    config : dict
        PyRadiomics configuration (passed to ``RadiomicsFeatureExtractor``).
    """

    def extract(self, image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
        """Extract features using pyradiomics."""
        # Stub — requires SimpleITK image conversion
        return {}
