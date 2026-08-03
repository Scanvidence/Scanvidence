"""Temperature scaling calibration."""

from .base import BaseCalibrator


class TemperatureScaling(BaseCalibrator):
    """Temperature scaling for post-hoc calibration.

    Learns a single scalar temperature parameter that divides the
    logits before softmax, minimising NLL on the calibration set.
    """

    def __init__(self, model=None):
        super().__init__(model)
        self.temperature = 1.0

    def fit(self, logits, labels):
        """Fit temperature on calibration data."""
        # Stub — optimise temperature via LBFGS
        self._fitted = True

    def calibrate(self, logits):
        """Apply temperature scaling."""
        return logits / self.temperature
