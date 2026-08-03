"""Request/Response Pydantic models for the REST API."""

from __future__ import annotations

from pydantic import BaseModel


class DetectionRequest(BaseModel):
    """API request for a detection task.

    Parameters
    ----------
    task : str
        Task name: ``"brain_tumor"`` or ``"alzheimers"``.
    scan_path : str
        Path to the uploaded scan file.
    """

    task: str
    scan_path: str


class DetectionResponse(BaseModel):
    """API response for a detection task.

    Parameters
    ----------
    task : str
        Task that was executed.
    prediction : str
        Predicted label.
    confidence : float
        Model confidence.
    uncertainty : float or None
        Epistemic uncertainty estimate.
    """

    task: str
    prediction: str
    confidence: float
    uncertainty: float | None = None
