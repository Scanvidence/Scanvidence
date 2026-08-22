"""N4 Bias Field Correction using SimpleITK."""

import numpy as np

from .base import BaseTransform


class BiasFieldCorrection(BaseTransform):
    """N4ITK Bias Field Correction for MRI volumes.

    Corrects low-frequency intensity variations caused by scanner magnetic
    field inhomogeneities. This is a mandatory preprocessing step for
    multi-center BraTS data to ensure consistent intensity distributions
    before Nyul Histogram Standardization.

    Parameters
    ----------
    shrink_factor : int, optional
        Isotropic shrink factor to reduce computation time during correction,
        by default 4.
    convergence : dict, optional
        Dictionary defining number of iterations and convergence threshold
        for each pyramid level.
        Default: {"iters": [50, 50, 50, 50], "tol": 0.0}.
    """

    def __init__(self, shrink_factor: int = 4, convergence: dict | None = None):
        super().__init__()
        self.shrink_factor = shrink_factor
        self.convergence = convergence or {"iters": [50, 50, 50, 50], "tol": 0.0}

    def __call__(self, array: np.ndarray, metadata: dict | None = None) -> tuple[np.ndarray, dict]:
        """Apply N4 bias field correction to a 3D MRI volume.

        Parameters
        ----------
        array : numpy.ndarray
            Input 3D image volume.
        metadata : dict or None
            Associated metadata (spacing, orientation, etc.).

        Returns
        -------
        tuple[numpy.ndarray, dict]
            Corrected image volume and updated metadata.
        """
        if metadata is None:
            metadata = {}

        import SimpleITK as sitk

        # Convert numpy array to SimpleITK image
        image_sitk = sitk.GetImageFromArray(array)

        # Apply spacing if available
        if "spacing" in metadata:
            image_sitk.SetSpacing(metadata["spacing"])

        # Cast to Float32 for N4 filter
        image_f32 = sitk.Cast(image_sitk, sitk.sitkFloat32)

        # Generate mask using Otsu thresholding
        mask_sitk = sitk.OtsuThreshold(image_f32, 0, 1, 200)

        # Shrink image and mask to speed up processing
        shrink_factor_list = [self.shrink_factor] * image_f32.GetDimension()
        shrinked_image = sitk.Shrink(image_f32, shrink_factor_list)
        shrinked_mask = sitk.Shrink(mask_sitk, shrink_factor_list)

        # Initialize N4 filter
        corrector = sitk.N4BiasFieldCorrectionImageFilter()

        iters = self.convergence.get("iters", [50, 50, 50, 50])
        tol = self.convergence.get("tol", 0.0)
        corrector.SetMaximumNumberOfIterations(iters)
        corrector.SetConvergenceThreshold(tol)

        # Run correction on shrunk images
        corrector.Execute(shrinked_image, shrinked_mask)

        # Upsample the bias field to original resolution and apply
        log_bias_field = corrector.GetLogBiasFieldAsImage(image_f32)
        corrected_image = image_f32 / sitk.Exp(log_bias_field)

        # Convert back to numpy array
        corrected_array = sitk.GetArrayFromImage(corrected_image)

        metadata["n4_bias_corrected"] = True
        return corrected_array, metadata
