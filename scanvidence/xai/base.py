"""Base class for explainability methods."""

from abc import ABC, abstractmethod
from typing import Any


class BaseExplainer(ABC):
    """Abstract base class for all XAI methods.

    Follows the pgmpy inference pattern: an explainer is initialised
    with a fitted model, then queried via :meth:`explain`.

    Parameters
    ----------
    model : scanvidence.models.base.BaseModel
        A fitted model to explain.
    """

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def explain(self, scan: Any, **kwargs) -> dict:
        """Generate an explanation for a prediction.

        Parameters
        ----------
        scan : Any
            Preprocessed input (numpy array or torch tensor).

        Returns
        -------
        explanation : dict
            Keys: ``heatmap`` (numpy array), ``metrics`` (dict).
        """
        pass
