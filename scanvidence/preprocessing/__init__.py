"""MRI preprocessing: skull stripping, registration, bias-field
correction, intensity normalization (z-score / Nyul), sequence fusion.

2D transforms for the classification track and 3D/nnU-Net-native
transforms for the segmentation track both live here.
"""

from .base import BaseTransform
