"""ADNI (Alzheimer's Disease Neuroimaging Initiative) dataset adapter."""

from .base import BaseDataset


class ADNIDataset(BaseDataset):
    """Adapter for the ADNI benchmark dataset.

    Discovers structural MRI scans organized by subject ID with
    associated clinical diagnosis labels.

    Parameters
    ----------
    root : str
        Root directory of the ADNI dataset.
    """

    def discover(self) -> list[dict]:
        """Discover all ADNI cases."""
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
        """Load an ADNI case."""
        import os

        import nibabel as nib

        path = record["path"]
        # Find the first NIfTI file in the subject directory
        for root_dir, _, files in os.walk(path):
            for f in sorted(files):
                if f.endswith((".nii", ".nii.gz")):
                    img = nib.load(os.path.join(root_dir, f))
                    return img.get_fdata(), {
                        "patient_id": record["patient_id"],
                        "filename": f,
                    }
        raise FileNotFoundError(f"No NIfTI file found in {path}")
