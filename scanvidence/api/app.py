"""FastAPI application factory."""

from fastapi import FastAPI

from .router import router


def create_app() -> FastAPI:
    """Create and configure the Scanvidence API application.

    Returns
    -------
    app : FastAPI
        Configured FastAPI application.
    """
    app = FastAPI(
        title="Scanvidence API",
        description=(
            "Unified REST API for explainable, uncertainty-aware "
            "medical imaging detection. Supports brain tumor and "
            "Alzheimer's disease detection."
        ),
        version="0.1.0",
    )
    app.include_router(router)
    return app
