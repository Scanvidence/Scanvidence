"""Brain tumor type classification model."""

from enum import Enum

from ..base import BaseModel


class ClassificationMode(Enum):
    """Input mode for the Negative Control Bias Test.

    ROI
        Cropped tumor region only (primary result).
    FULL_BRAIN
        Entire standardized brain volume (comparator).
    BACKGROUND_ONLY
        Tumor region masked to zero (negative control).
        Expected accuracy: ~33% (chance level for 3 classes).
    """

    ROI = "roi"
    FULL_BRAIN = "full_brain"
    BACKGROUND_ONLY = "background_only"


# Standard tumor type labels
TUMOR_TYPES = {"glioma": 0, "meningioma": 1, "metastasis": 2}


class TumorClassifier(BaseModel):
    """Multi-class 3D brain tumor ROI classifier.

    Classifies cropped 3D tumor Regions of Interest (ROIs) into:

    - Glioma (BraTS-GLI)
    - Meningioma (BraTS-MEN)
    - Brain Metastasis (BraTS-METS)

    Part of the Two-Stream Pipeline: the T4 segmentor produces a 3D
    mask, ``utils.extract_roi`` crops the original MRI to the tumor
    bounding box, and this classifier predicts tumor type from the
    cropped ROI volume.

    Notes
    -----
    - All patients are tumor-positive. No binary screening.
    - Pituitary tumors are excluded (incompatible MRI protocols).
    - The ``input_mode`` parameter enables the Negative Control Bias
      Test: train three parallel classifiers on ROI-only, full-brain,
      and background-only inputs to detect multi-cohort dataset bias.

    Parameters
    ----------
    config : dict
        Must contain:
        - ``backbone`` (str): e.g. 'resnet18_3d', 'densenet121_3d'
        - ``num_classes`` (int): default 3
        - ``input_mode`` (str): 'roi', 'full_brain', or 'background_only'
        - ``in_channels`` (int): number of MRI sequences (default 4)
    """

    def _validate_config(self) -> None:
        """Validate the classification configuration.

        Raises
        ------
        ValueError
            If required config keys are missing or invalid.
        """
        if "backbone" not in self.config:
            raise ValueError("Config must specify 'backbone'.")

        mode = self.config.get("input_mode", "roi")
        try:
            ClassificationMode(mode)
        except ValueError:
            valid = [m.value for m in ClassificationMode]
            raise ValueError(
                f"Invalid input_mode '{mode}'. Must be one of {valid}."
            )

        num_classes = self.config.get("num_classes", 3)
        if num_classes != len(TUMOR_TYPES):
            raise ValueError(
                f"num_classes must be {len(TUMOR_TYPES)} for tumor typing, "
                f"got {num_classes}."
            )

    def build(self):
        """Build the tumor classification network.

        Raises
        ------
        ValueError
            If configuration is invalid.
        """
        self._validate_config()
        # Stub — architecture selection via config['backbone']
        # Implementation will use the backbone registry from models/backbone/
        pass

    def forward(self, x):
        """Run forward pass on a 3D MRI ROI volume.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, in_channels, D, H, W)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, num_classes)``.
        """
        self.check_model()
        return self._model(x)
