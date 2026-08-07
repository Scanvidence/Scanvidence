"""Label validation for multi-cohort BraTS datasets (GLI, MEN, METS).

BraTS 2023 standardized the label map across all tracks:
    0 = Background
    1 = Non-Enhancing Tumor Core (NCR/NET)
    2 = Peritumoral Edema (ED)
    3 = GD-Enhancing Tumor (ET)

Since labels are consistent across GLI, MEN, and METS, this module
validates masks rather than remapping them.
"""

import numpy as np

# BraTS 2023 Unified Label Definitions (standardized across GLI, MEN, METS)
BRATS_LABELS = {
    "background": 0,
    "necrotic_core": 1,
    "edema": 2,
    "enhancing": 3,
}

VALID_LABELS = frozenset(BRATS_LABELS.values())


def validate_brats_labels(mask: np.ndarray) -> bool:
    """Verify that a segmentation mask contains only valid BraTS 2023 labels.

    Parameters
    ----------
    mask : np.ndarray
        3D segmentation mask to validate.

    Returns
    -------
    bool
        True if all voxel values are in {0, 1, 2, 3}.
    """
    unique_labels = set(np.unique(mask).astype(int))
    return unique_labels.issubset(VALID_LABELS)


def collapse_to_binary(mask: np.ndarray) -> np.ndarray:
    """Collapse a multi-class BraTS mask to binary (tumor vs background).

    Used by the ROI extractor to compute bounding boxes regardless
    of tumor sub-region.

    Parameters
    ----------
    mask : np.ndarray
        3D segmentation mask with BraTS 2023 labels.

    Returns
    -------
    np.ndarray
        Binary mask (0 = background, 1 = any tumor).
    """
    return (mask > 0).astype(np.uint8)


def get_tumor_regions(mask: np.ndarray) -> dict[str, np.ndarray]:
    """Extract individual tumor sub-region binary masks.

    Parameters
    ----------
    mask : np.ndarray
        3D segmentation mask with BraTS 2023 labels.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with keys 'whole_tumor', 'tumor_core', 'enhancing'
        mapping to binary masks for each standard BraTS evaluation region.
    """
    return {
        "whole_tumor": (mask > 0).astype(np.uint8),
        "tumor_core": np.isin(mask, [1, 3]).astype(np.uint8),
        "enhancing": (mask == 3).astype(np.uint8),
    }
