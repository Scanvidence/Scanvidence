"""BraTS evaluation metrics with the empty-region convention.

Patient-level reportable scores are computed from predicted label maps
using the BraTS convention: a region that is empty in the ground truth is
scored 1.0 if the prediction is also empty and 0.0 otherwise; non-empty
regions use standard Dice. Scores here are measured numbers for the
``as-metric-not-in-loss`` rule (see ``training.Loss``).
"""

from __future__ import annotations

import numpy as np

REGION_CLASSES: dict[str, tuple[int, ...]] = {
    "ET": (3,),  # enhancing tumor
    "TC": (1, 3),  # tumor core: necrotic + enhancing
    "WT": (1, 2, 3),  # whole tumor: all three tissue classes
}

REGION_ORDER: tuple[str, ...] = ("ET", "TC", "WT")


def dice_score(
    pred: np.ndarray,
    target: np.ndarray,
    classes: tuple[int, ...],
    *,
    empty_region_value: float = 1.0,
) -> float:
    """Dice for one anatomical region with the empty-region convention.

    Parameters
    ----------
    pred : np.ndarray
        Predicted integer labels.
    target : np.ndarray
        Ground-truth integer labels.
    classes : tuple of int
        Label values that make up the region.
    empty_region_value : float
        Score used when both maps are empty for the region (1.0 in BraTS).

    Returns
    -------
    float
        Dice in [0, 1].
    """
    pred_mask = np.isin(pred, classes)
    target_mask = np.isin(target, classes)
    intersection = np.logical_and(pred_mask, target_mask).sum()
    pred_sum = pred_mask.sum()
    target_sum = target_mask.sum()
    if target_sum == 0:
        return empty_region_value if pred_sum == 0 else 0.0
    return float(2.0 * intersection / (pred_sum + target_sum))


def regional_dice(
    pred: np.ndarray,
    target: np.ndarray,
    regions: tuple[str, ...] = REGION_ORDER,
) -> dict[str, float]:
    """ET/TC/WT Dice for one patch or volume.

    Parameters
    ----------
    pred : np.ndarray
        Predicted integer labels (argmax of logits).
    target : np.ndarray
        Ground-truth integer labels.
    regions : tuple of str
        Region keys to compute.

    Returns
    -------
    dict[str, float]
        One Dice score per region.
    """
    return {region: dice_score(pred, target, REGION_CLASSES[region]) for region in regions}


def mean_regional_dice(
    pred: np.ndarray,
    target: np.ndarray,
    regions: tuple[str, ...] = REGION_ORDER,
) -> float:
    """Mean of the regional ET/TC/WT Dice scores.

    This is the primary per-patch score tracked during B0 training.
    """
    scores = regional_dice(pred, target, regions)
    return float(np.mean(list(scores.values())))
