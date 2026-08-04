"""Tests for temperature scaling calibration."""

import numpy as np

from scanvidence.calibration import BaseCalibrator, TemperatureScaling


def test_temperature_scaling_is_base_calibrator_subclass():
    assert issubclass(TemperatureScaling, BaseCalibrator)


def test_default_temperature_is_one():
    assert TemperatureScaling().temperature == 1.0


def test_fit_marks_calibrator_as_fitted():
    calibrator = TemperatureScaling()
    calibrator.fit(np.array([1.0, 2.0]), np.array([0, 1]))
    assert calibrator._fitted


def test_calibrate_divides_logits_by_temperature():
    calibrator = TemperatureScaling()
    calibrator.temperature = 2.0
    logits = np.array([1.0, -3.0, 5.0])
    np.testing.assert_allclose(calibrator.calibrate(logits), logits / 2.0)
