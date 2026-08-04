"""REST API for clinical deployment of scanvidence detection tasks.

Provides FastAPI endpoints for brain tumor and Alzheimer's detection.
"""

from .app import create_app

__all__ = ["create_app"]
