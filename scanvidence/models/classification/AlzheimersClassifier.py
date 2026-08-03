"""Alzheimer's disease classification model."""

from ..base import BaseModel


class AlzheimersClassifier(BaseModel):
    """Alzheimer's disease stage classifier.

    Classifies structural MRI scans into cognitive normal (CN),
    mild cognitive impairment (MCI), or Alzheimer's disease (AD).

    Parameters
    ----------
    config : dict
        Must contain ``backbone`` (str) and ``num_classes`` (int, default 3).
    """

    def build(self):
        """Build the Alzheimer's classification network."""
        # Stub — architecture selection via config
        pass

    def forward(self, x):
        """Run forward pass on preprocessed MRI input."""
        self.check_model()
        return self._model(x)
