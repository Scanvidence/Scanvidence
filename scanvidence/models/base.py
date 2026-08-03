"""Base class for all scanvidence models."""

from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):
    """Abstract base class for detection/segmentation models.

    Follows the pgmpy pattern: models are declarative data structures
    that hold architecture and parameters. They validate themselves
    via :meth:`check_model` and support save/load for reproducibility.

    Parameters
    ----------
    config : dict
        Model configuration (architecture, hyperparameters).
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self._model = None

    @abstractmethod
    def build(self) -> None:
        """Construct the underlying neural network."""
        pass

    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Run a forward pass."""
        pass

    def check_model(self) -> None:
        """Validate that the model is properly configured.

        Raises
        ------
        ValueError
            If the model configuration is invalid.
        """
        if self._model is None:
            raise ValueError("Model not built. Call build() before using the model.")

    def save(self, path: str) -> None:
        """Save model weights and config to disk."""
        import torch

        self.check_model()
        assert self._model is not None
        torch.save(
            {"config": self.config, "state_dict": self._model.state_dict()},
            path,
        )

    def load(self, path: str) -> None:
        """Load model weights and config from disk."""
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.config = checkpoint["config"]
        self.build()
        assert self._model is not None
        self._model.load_state_dict(checkpoint["state_dict"])
