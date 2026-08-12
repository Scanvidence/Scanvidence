"""Shared fixtures: tiny synthetic BraTS-style NIfTI volumes on disk.

Shapes and label statistics are engineered so 16**3 / 32**3 patches
contain large tumor regions — enough for the trainer and CLI gating
tests on CPU. No real patient data anywhere.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest
from nibabel import Nifti1Image

MODALITIES = ("t1n", "t1c", "t2w", "t2f")
AFFINE = np.eye(4)


def write_case(
    root: str | pytest.TempPathFactory,
    case_id: str,
    shape: tuple[int, int, int] = (64, 64, 64),
    seed: int = 0,
    label4: bool = False,
    bad_geometry_modality: str | None = None,
    out_of_range_label: int | None = None,
) -> None:
    """Write one synthetic case; volumes contain three tumor-like blocks.

    Intensity layout mimics skull-stripped BraTS: an exact-zero background
    outside a central positive-intensity "brain" cube.
    """
    rng = np.random.default_rng(seed)
    lower = shape[0] // 4
    middle = shape[0] // 2
    upper = 3 * shape[0] // 4
    seg = np.zeros(shape, dtype=np.int64)
    seg[lower:middle, lower:middle, lower:middle] = 1  # necrotic core
    seg[middle:upper, middle:upper, middle:upper] = 2  # edema
    seg[lower + 4 : middle + 4, lower + 4 : middle + 4, lower + 4 : middle + 4] = 3  # ET
    if out_of_range_label is not None:
        seg[0:4, 0:4, 0:4] = out_of_range_label
    elif label4:
        seg[0:4, 0:4, 0:4] = 4

    case_dir = root / case_id if not isinstance(root, str) else type(root)(root) / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    for mod in MODALITIES:
        vol_shape = shape if mod != bad_geometry_modality else (32, 64, 64)
        vol = np.zeros(vol_shape, dtype=np.float32)
        b = tuple(slice(lo := s // 4, s - lo) for s in vol_shape)
        vol[b] = rng.normal(100.0, 10.0, size=vol[b].shape).astype(np.float32)
        img = nib.as_closest_canonical(Nifti1Image(vol, AFFINE))
        nib.save(img, case_dir / f"{case_id}-{mod}.nii.gz")
    seg_img = nib.as_closest_canonical(Nifti1Image(seg.astype(np.int16), AFFINE))
    nib.save(seg_img, case_dir / f"{case_id}-seg.nii.gz")


@pytest.fixture
def synthetic_brats(tmp_path) -> tuple[str, list[str]]:
    """Root with three ordinary GLI-named cases."""
    root = tmp_path / "braTS"
    ids = ["BraTS-GLI-00001-000", "BraTS-GLI-00002-000", "BraTS-GLI-00003-000"]
    for i, case_id in enumerate(ids):
        write_case(root, case_id, seed=i)
    return str(root), ids


@pytest.fixture
def synthetic_brats_label4(tmp_path) -> str:
    """Root with one correctly-labeled case and one legacy-label-4 case."""
    root = tmp_path / "braTS4"
    write_case(root, "BraTS-GLI-00010-000", seed=0)
    write_case(root, "BraTS-GLI-00011-000", seed=1, label4=True)
    return str(root)


@pytest.fixture
def synthetic_brats_bad_geometry(tmp_path) -> str:
    """Root with one case whose t2f volume has a different shape."""
    root = tmp_path / "braTSbg"
    write_case(root, "BraTS-GLI-00020-000", seed=0)
    write_case(root, "BraTS-GLI-00021-000", seed=1, bad_geometry_modality="t2f")
    return str(root)
