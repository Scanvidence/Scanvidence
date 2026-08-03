"""Gradient-weighted Class Activation Mapping (Grad-CAM)."""

from .base import BaseExplainer


class GradCAM(BaseExplainer):
    """Grad-CAM explainability for CNN-based classifiers.

    Generates class-discriminative localization maps by using gradients
    flowing into the final convolutional layer.

    Parameters
    ----------
    model : scanvidence.models.base.BaseModel
        A fitted classification model.
    target_layer : str or None
        Name of the convolutional layer to target. If None, uses the
        last convolutional layer.
    """

    def __init__(self, model, target_layer: str | None = None):
        super().__init__(model)
        self.target_layer = target_layer

    def explain(self, scan, **kwargs):
        """Generate Grad-CAM heatmap."""
        # Stub — implementation uses captum or manual hook-based approach
        return {"heatmap": None, "metrics": {}}
