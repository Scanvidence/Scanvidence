"""Tests for statistical hypothesis-testing harness."""

from scanvidence.evaluation import BaseMetric, StatisticalTests


def test_statistical_tests_is_base_metric_subclass():
    assert issubclass(StatisticalTests, BaseMetric)


def test_compute_returns_a_result_dict():
    result = StatisticalTests().compute([1, 0], [1, 1])
    assert isinstance(result, dict)
