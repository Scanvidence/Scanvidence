"""96-cubed patch dataset for BraTS-style multi-sequence volumes.

Implements the Milestone 2 preprocessing contract for B0 training:

- per-modality zscore normalization computed on nonzero brain voxels,
- foreground-aware patch sampling (center chosen from tumor voxels with
  probability ``foreground_prob``),
- augmentations (flips, 90-degree rotations, intensity scale/shift)
  applied identically to all modalities and the mask,
- a fail-closed label gate: BraTS 2023 enhancing tumor is label 3; the
  legacy label 4 only passes behind the explicit ``remap_legacy_four``
  flag, and any other out-of-range value raises.

Volumes are loaded lazily per ``__getitem__`` and cached with a bounded
LRU so a 240**3 multi-sequence case (~50 MB) is never duplicated for the
whole dataset, and the batch-1 patch pipeline fits any 8 GB GPU.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, cast

import nibabel as nib
import numpy as np
import torch
from nibabel import Nifti1Image
from torch import Tensor
from torch.utils.data import Dataset

MODALITY_KEYS: tuple[str, ...] = ("t1n", "t1c", "t2w", "t2f")

# BraTS 2023 (and the pre-release BraTS-GLI label map used across the
# whole project): 1 = necrotic core, 2 = edema, 3 = enhancing tumor.
EXPECTED_LABELS: frozenset[int] = frozenset({0, 1, 2, 3})
LEGACY_LABEL_4 = 4


class BraTSPatchDataset(Dataset):
    """Patched, normalized, augmented BraTS volumes for training.

    Parameters
    ----------
    records : list of dict
        Records from ``BraTSDataset(...).discover()`` pre-filtered to
        cases with all four modalities and a segmentation mask.
    patch_size : int
        Cubed patch edge length. Must divide ``2 ** len(model widths)``
        when consumed by B0 (96 for the default architecture).
    foreground_prob : float
        Probability that a patch center is a tumor voxel rather than a
        uniformly random voxel (foreground-aware sampling).
    seed : int
        Root seed; patch sampling and augmentation are re-seeded per item
        as ``seed + index`` so runs are reproducible independently of
        DataLoader worker count/platform (spawn vs fork).
    augment : bool
        Apply flips, 90-degree rotations, and intensity perturbations.
    remap_legacy_four : bool
        Explicit opt-in to remap BraTS 2020-style label 4 to 3. Without
        the flag, a volume containing 4 raises instead of silently
        training on a mismatched release.
    release : str
        Declared dataset release, only used in error messages so a
        mismatch is attributable.
    max_cached_cases : int
        Bounded in-process LRU cache of loaded volume arrays.

    Raises
    ------
    RuntimeError
        If a case's modalities disagree in geometry (shape or affine).
    ValueError
        If a segmentation volume contains out-of-range labels, or label 4
        without ``remap_legacy_four``.
    """

    def __init__(
        self,
        records: list[dict],
        patch_size: int = 96,
        foreground_prob: float = 0.5,
        seed: int = 0,
        augment: bool = True,
        remap_legacy_four: bool = False,
        release: str = "BraTS-2023-GLI",
        max_cached_cases: int = 4,
    ) -> None:
        super().__init__()
        if patch_size < 1:
            raise ValueError(f"patch_size must be >= 1, got {patch_size}.")
        if not 0.0 <= foreground_prob <= 1.0:
            raise ValueError(f"foreground_prob must be in [0, 1], got {foreground_prob}.")
        self.records = records
        self.patch_size = patch_size
        self.foreground_prob = foreground_prob
        self.seed = seed
        self.augment = augment
        self.remap_legacy_four = remap_legacy_four
        self.release = release
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._max_cached_cases = max_cached_cases

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        record = self.records[index]
        volumes, seg = self._load_case(record)
        rng = np.random.default_rng(self.seed + index)
        patch_np, seg_np = self._crop(volumes, seg, rng)
        patch = torch.from_numpy(patch_np)
        seg_patch = torch.from_numpy(seg_np)
        if self.augment:
            patch, seg_patch = self._augment(patch, seg_patch, rng)
        return patch.float(), seg_patch.long()

    def _load_case(self, record: dict) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Load and sanity-check one case, caching it in a bounded LRU."""
        case_id = record["patient_id"]
        if case_id in self._cache:
            self._cache.move_to_end(case_id)
            return self._cache[case_id]

        volumes: dict[str, np.ndarray] = {}
        affines: list[np.ndarray] = []
        for key in MODALITY_KEYS:
            path = record[f"{key}_path"]
            if path is None:
                raise ValueError(f"({case_id}) missing modality {key}; not trainable.")
            img = cast(Nifti1Image, nib.load(str(path)))
            volumes[key] = np.asarray(img.get_fdata(), dtype=np.float32)
            affines.append(np.asarray(img.affine))
        seg_path = record.get("seg_path")
        if seg_path is None:
            raise ValueError(f"({case_id}) missing segmentation mask; not trainable.")
        seg_img = cast(Nifti1Image, nib.load(str(seg_path)))
        seg = np.asarray(seg_img.get_fdata(), dtype=np.int64)
        affines.append(np.asarray(seg_img.affine))

        self._check_geometry(case_id, list(volumes.values()), seg, affines)
        self._check_labels(case_id, seg)

        volumes = {key: self._zscore_nonzero(vol) for key, vol in volumes.items()}
        if len(self._cache) >= self._max_cached_cases:
            self._cache.popitem(last=False)
        self._cache[case_id] = (volumes, seg)
        return volumes, seg

    @staticmethod
    def _check_geometry(
        case_id: str,
        volumes: list[np.ndarray],
        seg: np.ndarray,
        affines: list[np.ndarray],
    ) -> None:
        shapes = [list(vol.shape) for vol in volumes] + [list(seg.shape)]
        if len({tuple(s) for s in shapes}) != 1:
            raise RuntimeError(
                f"({case_id}) geometry mismatch between modalities or mask: {shapes}."
            )
        reference = affines[0]
        for other in affines[1:]:
            if not np.allclose(reference, other):
                raise RuntimeError(f"({case_id}) affine mismatch across modalities.")

    def _check_labels(self, case_id: str, seg: np.ndarray) -> None:
        found = set(np.unique(seg).tolist())
        if LEGACY_LABEL_4 in found:
            if not self.remap_legacy_four:
                raise ValueError(
                    f"({case_id}) found label 4, the legacy BraTS 2020 enhancing-tumor "
                    f"code. Declared release is {self.release}, whose labels are "
                    f"{sorted(EXPECTED_LABELS)}. Refusing to train on a mismatched "
                    f"release; pass remap_legacy_four=True only if you know the labels."
                )
            seg[seg == LEGACY_LABEL_4] = 3
            found.discard(LEGACY_LABEL_4)
        out_of_range = found - set(EXPECTED_LABELS)
        if out_of_range:
            raise ValueError(
                f"({case_id}) unexpected labels {sorted(out_of_range)}; "
                f"declared release {self.release} expects {sorted(EXPECTED_LABELS)}."
            )

    @staticmethod
    def _zscore_nonzero(volume: np.ndarray) -> np.ndarray:
        """Zscore each modality using statistics of nonzero brain voxels."""
        brain = volume > 0
        if not brain.any():
            return volume
        mean = volume[brain].mean()
        std = volume[brain].std()
        out = volume.copy()
        if std > 0:
            out[brain] = (out[brain] - mean) / std
        else:
            out[brain] = out[brain] - mean
        return out

    def _crop(
        self,
        volumes: dict[str, np.ndarray],
        seg: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample a patch around a foreground-aware center."""
        p = self.patch_size
        shape = np.asarray(seg.shape)
        if np.any(shape < p):
            raise ValueError(f"Volume {shape} smaller than patch {p}; pad volumes before training.")
        lo = p // 2
        hi = shape - p // 2
        if rng.random() < self.foreground_prob and np.any(seg > 0):
            fg = np.argwhere(seg > 0)
            center = fg[rng.integers(0, len(fg))]
            center = np.clip(center, lo, hi[0])
        else:
            center = np.asarray([rng.integers(int(lo), int(v)) for v in hi])
        center = np.clip(center, lo, hi)
        slices = tuple(slice(int(c - p // 2), int(c + p // 2)) for c in center)
        num_modalities = len(volumes)
        patch = np.empty((num_modalities, p, p, p), dtype=np.float32)
        for i, key in enumerate(MODALITY_KEYS):
            patch[i] = volumes[key][slices]
        return patch, np.ascontiguousarray(seg[slices])

    @staticmethod
    def _augment(
        patch: Tensor,
        seg: Tensor,
        rng: np.random.Generator,
    ) -> tuple[Tensor, Tensor]:
        """Augment patch and mask with identical geometric transforms."""
        for dim in range(3):
            if rng.random() < 0.5:
                patch = torch.flip(patch, dims=(dim + 1,))
                seg = torch.flip(seg, dims=(dim,))
        plane = int(rng.integers(0, 3))
        k = int(rng.integers(1, 4))
        spatial = [(0, 1), (0, 2), (1, 2)][plane]
        patch = torch.rot90(patch, k, dims=tuple(d + 1 for d in spatial))
        seg = torch.rot90(seg, k, dims=spatial)
        scale = torch.from_numpy(rng.uniform(0.9, 1.1, size=patch.shape[0])).float()
        shift = torch.from_numpy(rng.uniform(-0.1, 0.1, size=patch.shape[0])).float()
        patch = patch * scale.view(-1, 1, 1, 1) + shift.view(-1, 1, 1, 1)
        return patch, seg
