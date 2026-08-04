"""Tests for classification evaluation metrics."""

from scanvidence.evaluation import BaseMetric, ClassificationMetrics


def test_classification_metrics_is_base_metric_subclass():
    assert issubclass(ClassificationMetrics, BaseMetric)


def test_compute_returns_a_result_dict():
    result = ClassificationMetrics().compute([1, 0], [1, 1])
    assert isinstance(result, dict)
