"""BraTS (Brain Tumor Segmentation) dataset adapter."""

from .base import BaseDataset


class BraTSDataset(BaseDataset):
    """Adapter for the BraTS / BraTS-GLI benchmark dataset.

    Discovers multi-modal MRI cases (T1, T1ce, T2, FLAIR) with
    segmentation masks following the standard BraTS directory layout.

    Parameters
    ----------
    root : str
        Root directory of the BraTS dataset.
    """

    def discover(self) -> list[dict]:
        """Discover all BraTS cases."""
        import os

        records = []
        for case_dir in sorted(os.listdir(self.root)):
            case_path = os.path.join(self.root, case_dir)
            if os.path.isdir(case_path):
                records.append(
                    {
                        "patient_id": case_dir,
                        "path": case_path,
                    }
                )
        return records

    def load_case(self, record: dict):
        """Load a BraTS case with all modalities."""
        import os

        import nibabel as nib

        path = record["path"]
        modalities = {}
        for f in sorted(os.listdir(path)):
            if f.endswith((".nii", ".nii.gz")):
                key = f.replace(".nii.gz", "").replace(".nii", "")
                img = nib.load(os.path.join(path, f))
                modalities[key] = img.get_fdata()

        return modalities, {"patient_id": record["patient_id"]}
