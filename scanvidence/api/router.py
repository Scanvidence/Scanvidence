"""FastAPI router for detection endpoints."""

from fastapi import APIRouter, HTTPException

from .schemas import DetectionRequest, DetectionResponse
from scanvidence.tasks import BrainTumorTask, AlzheimersTask

router = APIRouter(prefix="/api/v1")

_TASK_REGISTRY = {
    "brain_tumor": BrainTumorTask,
    "alzheimers": AlzheimersTask,
}


@router.post("/detect/{task_name}", response_model=DetectionResponse)
async def detect(task_name: str, request: DetectionRequest):
    """Run a detection task.

    Parameters
    ----------
    task_name : str
        One of ``"brain_tumor"`` or ``"alzheimers"``.
    request : DetectionRequest
        The scan path and task configuration.
    """
    if task_name not in _TASK_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown task: {task_name}. Available: {list(_TASK_REGISTRY.keys())}",
        )

    task_cls = _TASK_REGISTRY[task_name]
    task = task_cls()
    result = task.run(request.scan_path)
    return DetectionResponse(
        task=result.task,
        prediction=result.prediction,
        confidence=result.confidence,
        uncertainty=result.uncertainty,
    )
