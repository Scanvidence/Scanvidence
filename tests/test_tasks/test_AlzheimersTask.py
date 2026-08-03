"""Tests for the Alzheimer's detection task orchestrator."""

from scanvidence.tasks import AlzheimersTask


def test_alzheimers_task_instantiates():
    task = AlzheimersTask(config={"model": "resnet50"})
    assert task.config == {"model": "resnet50"}
