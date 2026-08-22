"""Tests for BraTS 2023 label validation and region extraction."""

import numpy as np

from scanvidence.preprocessing.LabelUnifier import (
    BRATS_LABELS,
    collapse_to_binary,
    get_tumor_regions,
    validate_brats_labels,
)


class TestValidateBratsLabels:
    """Tests for validate_brats_labels."""

    def test_valid_mask_all_labels(self):
        mask = np.array([[[0, 1, 2], [3, 0, 1]]], dtype=np.uint8)
        assert validate_brats_labels(mask) is True

    def test_valid_mask_background_only(self):
        mask = np.zeros((4, 4, 4), dtype=np.uint8)
        assert validate_brats_labels(mask) is True

    def test_valid_mask_subset_of_labels(self):
        mask = np.array([[[0, 1], [2, 0]]], dtype=np.uint8)
        assert validate_brats_labels(mask) is True

    def test_invalid_mask_label_4(self):
        mask = np.array([[[0, 1, 4]]], dtype=np.uint8)
        assert validate_brats_labels(mask) is False

    def test_invalid_mask_negative_label(self):
        mask = np.array([[[-1, 0, 1]]], dtype=np.int8)
        assert validate_brats_labels(mask) is False

    def test_invalid_mask_large_label(self):
        mask = np.array([[[0, 255]]], dtype=np.uint8)
        assert validate_brats_labels(mask) is False


class TestCollapseToBinary:
    """Tests for collapse_to_binary."""

    def test_multi_class_to_binary(self):
        mask = np.array([[[0, 1, 2], [3, 0, 1]]], dtype=np.uint8)
        binary = collapse_to_binary(mask)
        expected = np.array([[[0, 1, 1], [1, 0, 1]]], dtype=np.uint8)
        np.testing.assert_array_equal(binary, expected)

    def test_empty_mask(self):
        mask = np.zeros((4, 4, 4), dtype=np.uint8)
        binary = collapse_to_binary(mask)
        np.testing.assert_array_equal(binary, mask)

    def test_dtype_is_uint8(self):
        mask = np.array([[[0, 3]]], dtype=np.float64)
        binary = collapse_to_binary(mask)
        assert binary.dtype == np.uint8


class TestGetTumorRegions:
    """Tests for get_tumor_regions."""

    def test_whole_tumor(self):
        mask = np.array([[[0, 1, 2, 3]]], dtype=np.uint8)
        regions = get_tumor_regions(mask)
        expected = np.array([[[0, 1, 1, 1]]], dtype=np.uint8)
        np.testing.assert_array_equal(regions["whole_tumor"], expected)

    def test_tumor_core(self):
        # Tumor core = labels 1 (NCR) and 3 (ET), NOT label 2 (edema)
        mask = np.array([[[0, 1, 2, 3]]], dtype=np.uint8)
        regions = get_tumor_regions(mask)
        expected = np.array([[[0, 1, 0, 1]]], dtype=np.uint8)
        np.testing.assert_array_equal(regions["tumor_core"], expected)

    def test_enhancing(self):
        mask = np.array([[[0, 1, 2, 3]]], dtype=np.uint8)
        regions = get_tumor_regions(mask)
        expected = np.array([[[0, 0, 0, 1]]], dtype=np.uint8)
        np.testing.assert_array_equal(regions["enhancing"], expected)

    def test_empty_mask_all_regions_empty(self):
        mask = np.zeros((4, 4, 4), dtype=np.uint8)
        regions = get_tumor_regions(mask)
        for name, region_mask in regions.items():
            assert region_mask.sum() == 0, f"{name} should be empty"


class TestBratsLabelsConstant:
    """Verify the label constant dictionary."""

    def test_standard_labels(self):
        assert BRATS_LABELS["background"] == 0
        assert BRATS_LABELS["necrotic_core"] == 1
        assert BRATS_LABELS["edema"] == 2
        assert BRATS_LABELS["enhancing"] == 3

    def test_exactly_four_labels(self):
        assert len(BRATS_LABELS) == 4
