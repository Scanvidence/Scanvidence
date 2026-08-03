"""Base class for evaluation metrics."""

from abc import ABC, abstractmethod
from typing import Any


class BaseMetric(ABC):
    """Abstract base class for evaluation metrics."""

    @abstractmethod
    def compute(self, predictions: Any, targets: Any) -> dict[str, float]:
        """Compute the metric.

        Parameters
        ----------
        predictions : Any
            Model predictions.
        targets : Any
            Ground-truth labels or masks.

        Returns
        -------
        results : dict[str, float]
            Metric name -> value mapping.
        """
        pass
