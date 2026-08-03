"""Post-hoc probability calibration (temperature scaling, Platt scaling,
isotonic regression) fit on a dedicated calibration subset, separate from
both the hyperparameter-tuning subset and the locked test set.
"""

from .base import BaseCalibrator
