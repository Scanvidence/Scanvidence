"""Statistical hypothesis tests for model comparison."""

from .base import BaseMetric


class StatisticalTests(BaseMetric):
    """Paired statistical tests: DeLong, McNemar, Wilcoxon,
    with Holm multiplicity correction.
    """

    def compute(self, predictions, targets):
        """Run statistical tests."""
        # Stub — uses scipy.stats
        return {}
