"""Training subpackage: patch data, losses, metrics, and the B0 trainer.

Owns the Milestone 2/3 training contract (batch-1 patching, patient-level
splits, AdamW + AMP + gradient accumulation with post-unscale clipping)
and exposes the ``python -m scanvidence.training`` CLI.
"""

from .base import BaseTrainer, RunHistory
from .BraTSPatchDataset import (
    EXPECTED_LABELS,
    LEGACY_LABEL_4,
    MODALITY_KEYS,
    BraTSPatchDataset,
)
from .Loss import SegmentationLoss
from .Metrics import REGION_CLASSES, dice_score, mean_regional_dice, regional_dice
from .SegmentationTrainer import SegmentationTrainer

__all__ = [
    "BaseTrainer",
    "RunHistory",
    "SegmentationTrainer",
    "BraTSPatchDataset",
    "MODALITY_KEYS",
    "EXPECTED_LABELS",
    "LEGACY_LABEL_4",
    "SegmentationLoss",
    "REGION_CLASSES",
    "dice_score",
    "mean_regional_dice",
    "regional_dice",
]
