"""MC-Dropout uncertainty estimation."""

from .base import BaseCalibrator


class MCDropout(BaseCalibrator):
    """Monte Carlo Dropout for epistemic uncertainty estimation.

    Runs multiple stochastic forward passes with dropout enabled
    to estimate prediction uncertainty.

    Parameters
    ----------
    model : scanvidence.models.base.BaseModel
        Model with dropout layers.
    n_samples : int
        Number of MC forward passes.
    """

    def __init__(self, model=None, n_samples: int = 30):
        super().__init__(model)
        self.n_samples = n_samples

    def fit(self, logits, labels):
        """No fitting needed for MC-Dropout."""
        self._fitted = True

    def calibrate(self, logits):
        """Return mean prediction across MC samples."""
        # Stub — requires running model in eval mode with dropout on
        return logits
