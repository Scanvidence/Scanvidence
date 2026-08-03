"""Brain tumor detection task."""

from .base import BaseTask


class BrainTumorTask(BaseTask):
    """End-to-end brain tumor detection pipeline.

    Orchestrates MRI loading, preprocessing, tumor classification,
    calibration, and XAI explanation generation.

    Parameters
    ----------
    config : dict
        Task configuration. Expected keys: ``preprocessing``,
        ``model``, ``calibration``, ``xai``.
    """

    def preprocess(self, scan_path):
        """Load and preprocess a brain MRI scan."""
        # Stub — compose preprocessing pipeline from config
        return scan_path

    def predict(self, preprocessed):
        """Classify the tumor type."""
        # Stub — run TumorClassifier
        return {"label": "unknown", "confidence": 0.0}

    def explain(self, preprocessed, prediction):
        """Generate Grad-CAM / SHAP / LIME explanations."""
        # Stub — run configured explainers
        return []
