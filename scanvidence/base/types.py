"""Shared Pydantic types for the scanvidence pipeline.

Every task's ``run()`` method returns a :class:`DetectionResult`, which
bundles the prediction, confidence, uncertainty, and XAI artefacts into
a single validated object.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict


class ScanInput(BaseModel):
    """Validated input to a detection task.

    Parameters
    ----------
    scan_path : str
        Path to a NIfTI / DICOM scan file.
    patient_id : str or None
        Optional patient identifier for provenance tracking.
    metadata : dict
        Arbitrary key-value metadata (e.g., scanner model, acquisition date).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    scan_path: str
    patient_id: str | None = None
    metadata: dict[str, Any] = {}


class XAIResult(BaseModel):
    """Container for explainability outputs.

    Parameters
    ----------
    method : str
        Name of the XAI method (e.g., ``"GradCAM"``, ``"SHAP"``).
    heatmap : Any
        The attribution heatmap (typically a numpy array).
    metrics : dict
        Quantitative explanation-quality metrics (IoU, pointing-game, etc.).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    method: str
    heatmap: Any = None
    metrics: dict[str, float] = {}


class DetectionResult(BaseModel):
    """Output of a detection task's ``run()`` method.

    Parameters
    ----------
    task : str
        Task identifier (e.g., ``"brain_tumor"``, ``"alzheimers"``).
    prediction : str
        Predicted class label.
    confidence : float
        Model confidence (post-calibration if available).
    uncertainty : float or None
        Epistemic uncertainty estimate (e.g., MC-Dropout variance).
    explanations : list of XAIResult
        One entry per XAI method applied.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    task: str
    prediction: str
    confidence: float
    uncertainty: float | None = None
    explanations: list[XAIResult] = []
