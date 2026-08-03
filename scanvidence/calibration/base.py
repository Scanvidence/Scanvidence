"""Base class for calibration methods."""

from abc import ABC, abstractmethod

import numpy as np


class BaseCalibrator(ABC):
    """Abstract base class for probability calibrators.

    Parameters
    ----------
    model : scanvidence.models.base.BaseModel
        The model whose outputs need calibration.
    """

    def __init__(self, model=None):
        self.model = model
        self._fitted = False

    @abstractmethod
    def fit(self, logits: np.ndarray, labels: np.ndarray) -> None:
        """Fit the calibrator on validation data.

        Parameters
        ----------
        logits : numpy.ndarray
            Raw model output logits.
        labels : numpy.ndarray
            Ground-truth labels.
        """
        pass

    @abstractmethod
    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply calibration to raw logits.

        Parameters
        ----------
        logits : numpy.ndarray
            Raw model output logits.

        Returns
        -------
        calibrated : numpy.ndarray
            Calibrated probabilities.
        """
        pass
