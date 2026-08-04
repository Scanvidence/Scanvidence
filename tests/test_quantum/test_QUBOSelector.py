"""Tests for the QUBO feature selector."""

import numpy as np

from scanvidence.quantum import BaseSelector, QUBOSelector


def test_qubo_selector_is_base_selector_subclass():
    assert issubclass(QUBOSelector, BaseSelector)


def test_default_budget_and_solver():
    selector = QUBOSelector()
    assert selector.budget == 50
    assert selector.solver == "simulated_annealing"


def test_select_respects_budget():
    features = np.zeros((10, 5))
    labels = np.zeros(10)
    assert QUBOSelector(budget=2).select(features, labels) == [0, 1]


def test_select_clamps_to_available_features():
    features = np.zeros((10, 3))
    labels = np.zeros(10)
    assert QUBOSelector(budget=50).select(features, labels) == [0, 1, 2]
