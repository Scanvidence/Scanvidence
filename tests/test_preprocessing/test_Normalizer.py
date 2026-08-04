"""Tests for MRI intensity normalization."""

import numpy as np

from scanvidence.preprocessing import Normalizer


def test_zscore_normalizes_brain_voxels():
    array = np.array([0.0, 1.0, 2.0, 3.0, 0.0, -5.0], dtype=float)
    normalized, _ = Normalizer(method="zscore")(array)
    brain = normalized[array > 0]
    np.testing.assert_allclose(brain.mean(), 0.0, atol=1e-6)
    np.testing.assert_allclose(brain.std(), 1.0, atol=1e-3)


def test_zscore_passthrough_when_no_brain_voxels():
    array = np.array([-1.0, -2.0, 0.0], dtype=float)
    normalized, _ = Normalizer(method="zscore")(array)
    np.testing.assert_allclose(normalized, array)


def test_minmax_scales_to_unit_interval():
    array = np.array([0.0, 5.0, 10.0], dtype=float)
    normalized, _ = Normalizer(method="minmax")(array)
    assert normalized.min() == 0.0
    np.testing.assert_allclose(normalized.max(), 1.0, atol=1e-6)


def test_returns_metadata():
    array = np.array([0.0, 1.0, 2.0], dtype=float)
    _, metadata = Normalizer()(array, {"patient": "p1"})
    assert metadata == {"patient": "p1"}
