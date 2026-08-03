"""SHAP (SHapley Additive exPlanations) explainer."""

from .base import BaseExplainer


class SHAPExplainer(BaseExplainer):
    """SHAP-based feature attribution.

    Uses DeepSHAP or KernelSHAP to compute Shapley values for
    model predictions.

    Parameters
    ----------
    model : scanvidence.models.base.BaseModel
        A fitted model.
    method : str
        SHAP variant. Options: ``"deep"``, ``"kernel"``, ``"gradient"``.
    """

    def __init__(self, model, method: str = "deep"):
        super().__init__(model)
        self.method = method

    def explain(self, scan, **kwargs):
        """Compute SHAP values."""
        # Stub — implementation uses shap library
        return {"heatmap": None, "metrics": {}}
