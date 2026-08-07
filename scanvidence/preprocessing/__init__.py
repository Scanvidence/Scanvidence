"""MRI preprocessing: skull stripping, registration, bias-field
correction, intensity normalization (z-score / Nyul), sequence fusion.

2D transforms for the classification track and 3D/nnU-Net-native
transforms for the segmentation track both live here.
"""

from .base import BaseTransform
from .BiasFieldCorrection import BiasFieldCorrection
from .LabelUnifier import (
    BRATS_LABELS,
    collapse_to_binary,
    get_tumor_regions,
    validate_brats_labels,
)
from .Normalizer import Normalizer
from .Pipeline import Pipeline
from .Registration import Registration
from .SkullStripper import SkullStripper

__all__ = [
    "BaseTransform",
    "BiasFieldCorrection",
    "BRATS_LABELS",
    "collapse_to_binary",
    "get_tumor_regions",
    "validate_brats_labels",
    "Normalizer",
    "Pipeline",
    "Registration",
    "SkullStripper",
]
