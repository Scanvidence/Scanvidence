"""DICOM scan loader."""

from .base import BaseLoader


class DICOMLoader(BaseLoader):
    """Load DICOM-format medical images.

    Uses SimpleITK under the hood for robust series handling.

    Parameters
    ----------
    data_root : str
        Root directory containing DICOM series.
    """

    def load(self, path: str):
        """Load a DICOM series from a directory.

        Parameters
        ----------
        path : str
            Path to the directory containing the DICOM series.

        Returns
        -------
        array : numpy.ndarray
            Image volume.
        metadata : dict
            Keys: ``spacing``, ``origin``, ``direction``, ``shape``.
        """
        import SimpleITK as sitk
        import numpy as np
        import os

        full_path = os.path.join(self.data_root, path)
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(full_path)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        return sitk.GetArrayFromImage(image), {
            "spacing": image.GetSpacing(),
            "origin": image.GetOrigin(),
            "direction": image.GetDirection(),
            "shape": image.GetSize(),
        }

    def validate(self, path: str) -> bool:
        """Check if the path contains valid DICOM files."""
        import os

        if not os.path.isdir(os.path.join(self.data_root, path)):
            return False
        return any(
            f.endswith(".dcm")
            for f in os.listdir(os.path.join(self.data_root, path))
        )
