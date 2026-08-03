"""Brain tumor segmentation model."""

from ..base import BaseModel


class TumorSegmentor(BaseModel):
    """nnU-Net-based brain tumor segmentation.

    Segments MRI volumes into whole tumor, tumor core, and enhancing
    tumor sub-regions.

    Parameters
    ----------
    config : dict
        nnU-Net configuration for the segmentation task.
    """

    def build(self):
        """Build the segmentation network."""
        # Stub — nnUNet integration
        pass

    def forward(self, x):
        """Run forward pass to produce segmentation mask."""
        self.check_model()
        return self._model(x)
