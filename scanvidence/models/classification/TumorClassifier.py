"""Brain tumor classification model."""

from ..base import BaseModel


class TumorClassifier(BaseModel):
    """Multi-class brain tumor classifier.

    Classifies MRI scans into glioma, meningioma, pituitary, or no-tumor
    using a configurable backbone architecture.

    Parameters
    ----------
    config : dict
        Must contain ``backbone`` (str) and ``num_classes`` (int).
    """

    def build(self):
        """Build the tumor classification network."""
        # Stub — architecture selection via config
        pass

    def forward(self, x):
        """Run forward pass on preprocessed MRI input."""
        self.check_model()
        return self._model(x)
