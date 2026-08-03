"""NIfTI (.nii / .nii.gz) scan loader."""

from .base import BaseLoader


class NIfTILoader(BaseLoader):
    """Load NIfTI-format MRI volumes.

    Uses nibabel under the hood. Returns a numpy array in RAS orientation
    with associated affine and header metadata.

    Parameters
    ----------
    data_root : str
        Root directory containing NIfTI files.
    """

    def load(self, path: str):
        """Load a NIfTI file.

        Parameters
        ----------
        path : str
            Path to the .nii or .nii.gz file.

        Returns
        -------
        array : numpy.ndarray
            Image volume.
        metadata : dict
            Keys: ``affine``, ``header``, ``shape``.
        """
        import os

        import nibabel as nib

        full_path = os.path.join(self.data_root, path)
        img = nib.load(full_path)
        return img.get_fdata(), {
            "affine": img.affine,
            "header": dict(img.header),
            "shape": img.shape,
        }

    def validate(self, path: str) -> bool:
        """Check if the file is a valid NIfTI file."""
        return path.endswith((".nii", ".nii.gz"))
