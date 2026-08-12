"""Tests for the BraTS patch dataset (sampling, normalization, label gate)."""

import numpy as np
import pytest
import torch

from scanvidence.data.datasets import BraTSDataset
from scanvidence.training import EXPECTED_LABELS, BraTSPatchDataset

PATCH = 32


def _records(root: str) -> list[dict]:
    return BraTSDataset(root, track="GLI").discover()


def test_patch_shape_and_label_set(synthetic_brats):
    root, ids = synthetic_brats
    dataset = BraTSPatchDataset(_records(root), patch_size=PATCH, seed=1, augment=False)
    assert len(dataset) == 3
    patch, seg = dataset[0]
    assert patch.shape == (4, PATCH, PATCH, PATCH)
    assert seg.shape == (PATCH, PATCH, PATCH)
    assert patch.dtype == torch.float32
    assert set(torch.unique(seg).tolist()) <= set(EXPECTED_LABELS)


def test_zscore_on_nonzero_brain_voxels(tmp_path):
    """A constant nonzero volume must normalize to exactly zero on brain."""
    import nibabel as nib
    from nibabel import Nifti1Image

    from tests.test_training.conftest import write_case

    root = tmp_path / "constant"
    write_case(root, "BraTS-GLI-00030-000", seed=0)
    case = BraTSDataset(str(root), track="GLI").discover()[0]
    case_dir = root / "BraTS-GLI-00030-000"
    for key in ("t1n", "t1c", "t2w", "t2f"):
        const = np.full((64, 64, 64), 5.0, dtype=np.float32)
        nib.save(Nifti1Image(const, np.eye(4)), case_dir / f"BraTS-GLI-00030-000-{key}.nii.gz")
    dataset = BraTSPatchDataset([case], patch_size=PATCH, seed=0, augment=False)
    dataset._load_case(case)  # populate cache
    volumes, _ = dataset._cache[case["patient_id"]]
    for key, vol in volumes.items():
        brain = vol > 0
        assert np.allclose(vol[brain], 0.0, atol=1e-5), key


def test_foreground_sampling_hits_tumor(synthetic_brats):
    root, ids = synthetic_brats
    dataset = BraTSPatchDataset(
        _records(root), patch_size=PATCH, foreground_prob=1.0, seed=3, augment=False
    )
    for index in range(len(dataset)):
        _, seg = dataset[index]
        assert (seg > 0).any(), f"foreground sample missed tumor at index {index}"


def test_volume_smaller_than_patch_raises(synthetic_brats):
    root, _ = synthetic_brats
    dataset = BraTSPatchDataset(_records(root), patch_size=96, seed=0, augment=False)
    with pytest.raises(ValueError, match="smaller than patch"):
        dataset[0]


def test_label4_fails_closed_without_flag(synthetic_brats_label4):
    root = synthetic_brats_label4
    dataset = BraTSPatchDataset(_records(root), patch_size=PATCH, seed=0, augment=False)
    with pytest.raises(ValueError, match="label 4"):
        dataset[1]


def test_label4_remaps_behind_explicit_flag(synthetic_brats_label4):
    root = synthetic_brats_label4
    dataset = BraTSPatchDataset(
        _records(root),
        patch_size=PATCH,
        seed=0,
        augment=False,
        remap_legacy_four=True,
    )
    for index in range(len(dataset)):
        _, seg = dataset[index]
        assert set(torch.unique(seg).tolist()) <= set(EXPECTED_LABELS)


def test_out_of_range_labels_fail_hard(tmp_path):
    from tests.test_training.conftest import write_case

    root = tmp_path / "bogus"
    write_case(root, "BraTS-GLI-00040-000", seed=0, out_of_range_label=9)
    dataset = BraTSPatchDataset(
        BraTSDataset(str(root), track="GLI").discover(), patch_size=PATCH, seed=0
    )
    with pytest.raises(ValueError, match="unexpected labels"):
        dataset[0]


def test_geometry_mismatch_fails_closed(synthetic_brats_bad_geometry):
    root = synthetic_brats_bad_geometry
    dataset = BraTSPatchDataset(_records(root), patch_size=PATCH, seed=0, augment=False)
    with pytest.raises(RuntimeError, match="geometry mismatch"):
        dataset[1]


def test_augmented_patch_and_mask_stay_consistent(synthetic_brats):
    root, _ = synthetic_brats
    dataset = BraTSPatchDataset(_records(root), patch_size=PATCH, seed=7, augment=True)
    patch, seg = dataset[0]
    assert patch.shape == (4, PATCH, PATCH, PATCH)
    assert seg.shape == (PATCH, PATCH, PATCH)
    assert torch.isfinite(patch).all()
    assert set(torch.unique(seg).tolist()) <= set(EXPECTED_LABELS)
    assert torch.unique(patch).numel() > 1  # intensity perturbation applied


def test_same_seed_and_index_is_deterministic(synthetic_brats):
    root, _ = synthetic_brats
    a = BraTSPatchDataset(_records(root), patch_size=PATCH, seed=11, augment=True)
    b = BraTSPatchDataset(_records(root), patch_size=PATCH, seed=11, augment=True)
    for index in range(len(a)):
        pa, sa = a[index]
        pb, sb = b[index]
        assert torch.equal(pa, pb) and torch.equal(sa, sb)


def test_no_augment_flag_disables_perturbation(synthetic_brats):
    root, _ = synthetic_brats
    dataset = BraTSPatchDataset(_records(root), patch_size=PATCH, seed=0, augment=False)
    patch, _ = dataset[0]
    assert torch.isfinite(patch).all()
    # Plain z-scores on brain voxels; no scale/shift or flips/rotations.
    assert patch.min() >= -20.0 and patch.max() <= 20.0


def test_record_missing_seg_raises(synthetic_brats):
    root, _ = synthetic_brats
    record = dict(_records(root)[0])
    record["seg_path"] = None
    dataset = BraTSPatchDataset([record], patch_size=PATCH, seed=0, augment=False)
    with pytest.raises(ValueError, match="missing segmentation"):
        dataset[0]


def test_invalid_constructor_args_raise(synthetic_brats):
    root, _ = synthetic_brats
    records = _records(root)
    with pytest.raises(ValueError, match="patch_size"):
        BraTSPatchDataset(records, patch_size=0)
    with pytest.raises(ValueError, match="foreground_prob"):
        BraTSPatchDataset(records, foreground_prob=1.5)
