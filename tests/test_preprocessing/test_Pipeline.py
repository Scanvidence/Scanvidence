"""Tests for the composable preprocessing pipeline."""

from scanvidence.preprocessing import Normalizer, Pipeline


def test_empty_pipeline_passes_input_through():
    array = [1, 2, 3]
    out, metadata = Pipeline()(array)
    assert out == array
    assert metadata == {}


def test_pipeline_applies_transforms_in_order():
    seen = []

    class RecordingTransform:
        def __call__(self, array, metadata=None):
            seen.append(array)
            return array, metadata or {}

    Pipeline([RecordingTransform(), RecordingTransform()])([1])
    assert seen == [[1], [1]]


def test_metadata_propagates_through_pipeline():
    class AddTag:
        def __call__(self, array, metadata=None):
            metadata = metadata or {}
            metadata["seen"] = metadata.get("seen", 0) + 1
            return array, metadata

    _, metadata = Pipeline([AddTag(), AddTag()])([1])
    assert metadata == {"seen": 2}


def test_append_adds_a_transform():
    pipeline = Pipeline()
    pipeline.append(Normalizer(method="minmax"))
    assert len(pipeline.transforms) == 1
