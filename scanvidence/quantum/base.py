"""Base class for feature selectors."""

from abc import ABC, abstractmethod

import numpy as np


class BaseSelector(ABC):
    """Abstract base class for feature selection methods.

    Parameters
    ----------
    budget : int
        Maximum number of features to select.
    """

    def __init__(self, budget: int = 50):
        self.budget = budget

    @abstractmethod
    def select(self, features: np.ndarray, labels: np.ndarray) -> list[int]:
        """Select the best feature subset.

        Parameters
        ----------
        features : numpy.ndarray
            Feature matrix (n_samples, n_features).
        labels : numpy.ndarray
            Target labels.

        Returns
        -------
        indices : list[int]
            Indices of selected features.
        """
        pass
