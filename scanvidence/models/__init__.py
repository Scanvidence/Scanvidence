"""Model wrappers for classification and segmentation tracks.

Keep architecture selection config-driven (see configs/) rather than
hardcoded, so a training run's config file is a complete record of
what produced a given result.
"""

from .base import BaseModel
