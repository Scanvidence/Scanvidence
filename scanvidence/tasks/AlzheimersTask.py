"""Alzheimer's disease detection task."""

from .base import BaseTask


class AlzheimersTask(BaseTask):
    """End-to-end Alzheimer's disease detection pipeline.

    Orchestrates structural MRI loading, preprocessing, AD stage
    classification, calibration, and XAI explanation generation.

    Parameters
    ----------
    config : dict
        Task configuration. Expected keys: ``preprocessing``,
        ``model``, ``calibration``, ``xai``.
    """

    def preprocess(self, scan_path):
        """Load and preprocess a structural MRI scan."""
        # Stub — compose preprocessing pipeline from config
        return scan_path

    def predict(self, preprocessed):
        """Classify the Alzheimer's stage (CN/MCI/AD)."""
        # Stub — run AlzheimersClassifier
        return {"label": "unknown", "confidence": 0.0}

    def explain(self, preprocessed, prediction):
        """Generate explanations for Alzheimer's prediction."""
        # Stub — run configured explainers
        return []
