"""Model wrappers for classification and segmentation tracks.

Keep architecture selection config-driven (see configs/) rather than
hardcoded, so a training run's config file is a complete record of
what produced a given result.
"""

from .backbone import SegResNetB0
from .base import BaseModel
from .classification import AlzheimersClassifier, TumorClassifier
from .segmentation import TumorSegmentor

__all__ = ["BaseModel", "SegResNetB0", "AlzheimersClassifier", "TumorClassifier", "TumorSegmentor"]
