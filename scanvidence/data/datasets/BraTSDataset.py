"""BraTS (Brain Tumor Segmentation) multi-cohort dataset adapter.

Supports BraTS-GLI, BraTS-MEN, and BraTS-METS datasets with standardized
label maps (BraTS 2023: 1=Core, 2=Edema, 3=Enhancing).
"""

import os
from typing import Any, cast

import nibabel as nib
from nibabel import Nifti1Image

from .base import BaseDataset

# Standard BraTS 2023 modality filename suffixes.
# Across GLI, MEN, and METS, sequences are named consistently.
_MODALITY_SUFFIXES = {
    "t1n": ["t1n", "t1"],
    "t1c": ["t1c", "t1ce"],
    "t2w": ["t2w", "t2"],
    "t2f": ["t2f", "flair", "t2-flair"],
}

_SEG_SUFFIXES = ["seg", "mask"]

# Map BraTS directory prefixes to tumor types
_TRACK_PREFIXES = {
    "BraTS-GLI": "glioma",
    "BraTS-MEN": "meningioma",
    "BraTS-MET": "metastasis",
}


class BraTSDataset(BaseDataset):
    """Multi-cohort adapter for BraTS-GLI, BraTS-MEN, and BraTS-METS.

    Discovers multi-modal MRI cases (T1n, T1c, T2w, T2-FLAIR) with
    segmentation masks following the standard BraTS directory layout.
    Infers ``tumor_type`` from directory naming conventions.

    Parameters
    ----------
    root : str
        Root directory containing BraTS case directories.
    track : str or None, optional
        Restrict discovery to a specific track ('GLI', 'MEN', 'METS').
        If None, discovers all tracks found in the root directory.
    """

    def __init__(self, root: str, track: str | None = None):
        super().__init__(root)
        self.track = track.upper() if track else None

    def _infer_tumor_type(self, case_name: str) -> str:
        """Infer tumor type from the BraTS case directory name.

        Parameters
        ----------
        case_name : str
            Directory name (e.g., 'BraTS-GLI-00001-000').

        Returns
        -------
        str
            One of 'glioma', 'meningioma', 'metastasis', or 'unknown'.
        """
        upper = case_name.upper()
        for prefix, tumor_type in _TRACK_PREFIXES.items():
            if upper.startswith(prefix.upper()):
                return tumor_type
        return "unknown"

    def _matches_track(self, case_name: str) -> bool:
        """Check if a case belongs to the configured track filter."""
        if self.track is None:
            return True
        return case_name.upper().startswith(f"BRATS-{self.track}")

    def _find_modality(self, case_path: str, case_name: str, suffixes: list[str]) -> str | None:
        """Find a NIfTI file matching one of the given suffixes."""
        for f in os.listdir(case_path):
            if not f.endswith((".nii", ".nii.gz")):
                continue
            f_lower = f.lower()
            for suffix in suffixes:
                if f"-{suffix}." in f_lower or f"_{suffix}." in f_lower:
                    return os.path.join(case_path, f)
        return None

    def discover(self) -> list[dict]:
        """Discover all BraTS cases with modality paths and tumor type.

        Returns
        -------
        list of dict
            Each dict contains:
            - ``patient_id``: Case directory name
            - ``path``: Full path to case directory
            - ``tumor_type``: 'glioma', 'meningioma', or 'metastasis'
            - ``t1n_path``, ``t1c_path``, ``t2w_path``, ``t2f_path``: Modality paths
            - ``seg_path``: Segmentation mask path (or None)
            - ``available_sequences``: List of found sequence keys
        """
        records = []
        for case_dir in sorted(os.listdir(self.root)):
            case_path = os.path.join(self.root, case_dir)
            if not os.path.isdir(case_path):
                continue
            if not self._matches_track(case_dir):
                continue

            record: dict[str, Any] = {
                "patient_id": case_dir,
                "path": case_path,
                "tumor_type": self._infer_tumor_type(case_dir),
            }

            available = []
            for key, suffixes in _MODALITY_SUFFIXES.items():
                path = self._find_modality(case_path, case_dir, suffixes)
                record[f"{key}_path"] = path
                if path is not None:
                    available.append(key)

            seg_path = self._find_modality(case_path, case_dir, _SEG_SUFFIXES)
            record["seg_path"] = seg_path
            record["available_sequences"] = available

            records.append(record)

        return records

    def load_case(self, record: dict) -> tuple[dict[str, Any], dict]:
        """Load a BraTS case with all available modalities.

        Parameters
        ----------
        record : dict
            A record from :meth:`discover`.

        Returns
        -------
        modalities : dict[str, numpy.ndarray]
            Maps sequence key ('t1n', 't1c', 't2w', 't2f') to 3D arrays.
            Includes 'seg' if segmentation mask is available.
        metadata : dict
            Case metadata including ``patient_id``, ``tumor_type``,
            ``available_sequences``.
        """
        modalities: dict[str, Any] = {}

        for key in _MODALITY_SUFFIXES:
            path = record.get(f"{key}_path")
            if path is not None:
                img = cast(Nifti1Image, nib.load(path))
                modalities[key] = img.get_fdata()

        if record.get("seg_path") is not None:
            seg_img = cast(Nifti1Image, nib.load(record["seg_path"]))
            modalities["seg"] = seg_img.get_fdata()

        metadata = {
            "patient_id": record["patient_id"],
            "tumor_type": record["tumor_type"],
            "available_sequences": record["available_sequences"],
        }

        return modalities, metadata
