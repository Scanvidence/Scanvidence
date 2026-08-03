"""Classification evaluation metrics."""

from .base import BaseMetric


class ClassificationMetrics(BaseMetric):
    """Standard classification metrics: accuracy, precision, recall,
    F1, AUROC, with bootstrap confidence intervals.
    """

    def compute(self, predictions, targets):
        """Compute classification metrics."""
        # Stub — uses scikit-learn metrics
        return {}
