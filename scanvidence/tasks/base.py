"""Base class for detection tasks."""

from abc import ABC, abstractmethod
from typing import Any

from scanvidence.base.types import DetectionResult


class BaseTask(ABC):
    """Abstract base class for detection tasks.

    A task orchestrates the full pipeline: loading, preprocessing,
    inference, calibration, and explanation. Concrete tasks
    (brain tumor, Alzheimer's) specify which components to use.

    Parameters
    ----------
    config : dict
        Task configuration loaded from YAML.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    @classmethod
    def from_config(cls, config_path: str) -> "BaseTask":
        """Load a task from a YAML configuration file.

        Parameters
        ----------
        config_path : str
            Path to the YAML config file.

        Returns
        -------
        task : BaseTask
            Configured task instance.
        """
        import yaml

        with open(config_path) as f:
            config = yaml.safe_load(f)
        return cls(config=config)

    @abstractmethod
    def preprocess(self, scan_path: str) -> Any:
        """Load and preprocess a scan."""
        pass

    @abstractmethod
    def predict(self, preprocessed: Any) -> dict:
        """Run model inference."""
        pass

    @abstractmethod
    def explain(self, preprocessed: Any, prediction: dict) -> list[dict]:
        """Generate XAI explanations."""
        pass

    def run(self, scan_path: str) -> DetectionResult:
        """Execute the full detection pipeline.

        Parameters
        ----------
        scan_path : str
            Path to the input scan.

        Returns
        -------
        result : DetectionResult
            Bundled prediction, confidence, uncertainty, and explanations.
        """
        preprocessed = self.preprocess(scan_path)
        prediction = self.predict(preprocessed)
        explanations = self.explain(preprocessed, prediction)
        return DetectionResult(
            task=self.__class__.__name__,
            prediction=prediction.get("label", "unknown"),
            confidence=prediction.get("confidence", 0.0),
            uncertainty=prediction.get("uncertainty"),
            explanations=explanations,
        )
