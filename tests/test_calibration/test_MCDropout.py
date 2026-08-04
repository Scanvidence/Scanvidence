"""Tests for MC-Dropout uncertainty estimation."""

import numpy as np

from scanvidence.calibration import BaseCalibrator, MCDropout


def test_mc_dropout_is_base_calibrator_subclass():
    assert issubclass(MCDropout, BaseCalibrator)


def test_default_number_of_mc_samples():
    assert MCDropout().n_samples == 30


def test_custom_number_of_mc_samples():
    assert MCDropout(n_samples=10).n_samples == 10


def test_fit_marks_calibrator_as_fitted():
    dropout = MCDropout()
    dropout.fit(np.array([1.0]), np.array([1]))
    assert dropout._fitted


def test_calibrate_passes_logits_through():
    dropout = MCDropout()
    logits = np.array([0.1, 0.9])
    np.testing.assert_allclose(dropout.calibrate(logits), logits)
