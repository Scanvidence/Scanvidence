"""Tests for the FastAPI detection endpoints."""

import pytest
from fastapi.testclient import TestClient

from scanvidence.api import create_app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app())


def test_app_metadata():
    app = create_app()
    assert app.title == "Scanvidence API"
    assert app.version == "0.1.0"


def test_detect_brain_tumor_returns_detection_response(client):
    response = client.post(
        "/api/v1/detect/brain_tumor",
        json={"task": "brain_tumor", "scan_path": "patient_001.nii.gz"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["task"] == "BrainTumorTask"
    assert body["prediction"] == "unknown"
    assert body["confidence"] == 0.0
    assert body["uncertainty"] is None


def test_detect_alzheimers_returns_detection_response(client):
    response = client.post(
        "/api/v1/detect/alzheimers",
        json={"task": "alzheimers", "scan_path": "subject_042.nii.gz"},
    )
    assert response.status_code == 200
    assert response.json()["task"] == "AlzheimersTask"


def test_detect_unknown_task_returns_404(client):
    response = client.post(
        "/api/v1/detect/not_a_task",
        json={"task": "not_a_task", "scan_path": "x.nii.gz"},
    )
    assert response.status_code == 404
    assert "brain_tumor" in response.json()["detail"]
