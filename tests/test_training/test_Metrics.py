"""Tests for BraTS regional Dice metrics (empty-region convention)."""

import numpy as np

from scanvidence.training import dice_score, mean_regional_dice, regional_dice


def _volume_with(labels: list[tuple[tuple[int, int, int, int, int, int], int]]) -> np.ndarray:
    vol = np.zeros((16, 16, 16), dtype=np.int64)
    for (d_lo, d_hi, h_lo, h_hi, w_lo, w_hi), label in labels:
        vol[d_lo:d_hi, h_lo:h_hi, w_lo:w_hi] = label
    return vol


def test_empty_region_both_empty_scores_one():
    target = np.zeros((16, 16, 16), dtype=np.int64)
    assert dice_score(target.copy(), target, (3,)) == 1.0


def test_empty_target_nonempty_prediction_scores_zero():
    target = np.zeros((16, 16, 16), dtype=np.int64)
    pred = target.copy()
    pred[0:4, 0:4, 0:4] = 3
    assert dice_score(pred, target, (3,)) == 0.0


def test_identical_maps_score_one():
    target = _volume_with([((2, 10, 2, 10, 2, 10), 3), ((1, 3, 1, 3, 1, 3), 1)])
    assert dice_score(target.copy(), target, (3,)) == 1.0


def test_half_overlap_scores_half():
    target = _volume_with([((4, 12, 4, 12, 4, 12), 2)])
    pred = target.copy()
    pred[8:16, 8:16, 8:16] = 2  # union is 2*8**3 - 4**3 voxels
    assert np.isclose(dice_score(pred, target, (2,)), 2 * 8**3 / (960 + 512))


def test_regional_dice_maps_regions_correctly():
    # Disjoint slab regions so label volumes do not overwrite each other.
    target = _volume_with(
        [((4, 8, 4, 8, 4, 8), 1), ((8, 12, 8, 12, 8, 12), 3), ((12, 16, 12, 16, 12, 16), 2)]
    )
    pred = np.zeros_like(target)
    pred[8:12, 8:12, 8:12] = 3  # only the enhancing slab
    scores = regional_dice(pred, target)
    assert np.isclose(scores["ET"], 1.0)
    # TC = labels {1, 3}: 4**3 pred fully inside 2 * 4**3 target.
    assert np.isclose(scores["TC"], 2 * 4**3 / (4**3 + 2 * 4**3))
    assert np.isclose(scores["WT"], 2 * 4**3 / (4**3 + 3 * 4**3))
    assert np.isclose(
        mean_regional_dice(pred, target),
        np.mean([scores["ET"], scores["TC"], scores["WT"]]),
    )
