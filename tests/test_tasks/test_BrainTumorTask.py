"""Tests for the brain tumor detection task orchestrator."""

from scanvidence.tasks import BrainTumorTask


def test_brain_tumor_task_instantiates():
    task = BrainTumorTask(config={"model": "resnet50"})
    assert task.config == {"model": "resnet50"}
