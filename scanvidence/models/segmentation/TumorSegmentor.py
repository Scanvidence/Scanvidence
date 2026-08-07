"""Brain tumor segmentation model."""

from ..base import BaseModel


class TumorSegmentor(BaseModel):
    """nnU-Net-based multi-cohort brain tumor segmentation.

    Segments full 3D MRI volumes using a Unified Label Map across
    BraTS-GLI, BraTS-MEN, and BraTS-METS datasets:
    - Class 0: Background
    - Class 1: Solid / Active Tumor Core
    - Class 2: Surrounding Edema
    - Class 3: Enhancing Tumor (where applicable)

    Notes
    -----
    - Must process full-brain volumes via sliding-window patches to preserve
      spatial context for accurate boundary detection.
    - The output mask is subsequently used by `utils.extract_roi` to generate
      3D bounding boxes for the downstream T3 ROI Classifier.

    Parameters
    ----------
    config : dict
        nnU-Net configuration for the unified segmentation task.
    """

    def build(self):
        """Build the segmentation network."""
        # Stub — nnUNet integration
        pass

    def forward(self, x):
        """Run forward pass to produce unified 3D segmentation mask."""
        self.check_model()
        return self._model(x)
