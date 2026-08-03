"""OASIS (Open Access Series of Imaging Studies) dataset adapter."""

from .base import BaseDataset


class OASISDataset(BaseDataset):
    """Adapter for the OASIS benchmark datasets (OASIS-1/2/3/4).

    Primary dataset for the Alzheimer's disease track. Discovers
    structural MRI scans organized by subject ID with associated
    clinical diagnosis labels (demented / nondemented).

    Parameters
    ----------
    root : str
        Root directory of the OASIS dataset.
    """

    def discover(self) -> list[dict]:
        """Discover all OASIS cases."""
        import os

        records = []
        for subject_dir in sorted(os.listdir(self.root)):
            subject_path = os.path.join(self.root, subject_dir)
            if os.path.isdir(subject_path):
                records.append(
                    {
                        "patient_id": subject_dir,
                        "path": subject_path,
                    }
                )
        return records

    def load_case(self, record: dict):
        """Load an OASIS case."""
        import os
        from typing import cast

        import nibabel as nib
        from nibabel import Nifti1Image

        path = record["path"]
        # Find the first NIfTI file in the subject directory
        for root_dir, _, files in os.walk(path):
            for f in sorted(files):
                if f.endswith((".nii", ".nii.gz")):
                    img = cast(Nifti1Image, nib.load(os.path.join(root_dir, f)))
                    return img.get_fdata(), {
                        "patient_id": record["patient_id"],
                        "filename": f,
                    }
        raise FileNotFoundError(f"No NIfTI file found in {path}")
