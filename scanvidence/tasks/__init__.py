"""Task orchestrators that wire the full detection pipeline.

Each task composes data loading, preprocessing, model inference,
calibration, and XAI into a single ``run()`` call.
"""

from .base import BaseTask
from .BrainTumorTask import BrainTumorTask
from .AlzheimersTask import AlzheimersTask
