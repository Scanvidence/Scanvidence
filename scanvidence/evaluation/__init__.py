"""Statistical evaluation harness: bootstrap confidence intervals, paired
DeLong/McNemar/Wilcoxon tests, Holm multiplicity correction, and the
pre-specified non-inferiority decision rules (H1a, H1b).
"""

from .base import BaseMetric
from .ClassificationMetrics import ClassificationMetrics
from .StatisticalTests import StatisticalTests

__all__ = ["BaseMetric", "ClassificationMetrics", "StatisticalTests"]
