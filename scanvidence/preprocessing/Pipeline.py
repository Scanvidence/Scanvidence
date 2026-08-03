"""Composable preprocessing pipeline."""

from typing import Any

from .base import BaseTransform


class Pipeline:
    """Chain multiple transforms into a sequential pipeline.

    Parameters
    ----------
    transforms : list of BaseTransform
        Transforms to apply in order.

    Examples
    --------
    >>> from scanvidence.preprocessing import Pipeline, Normalizer, SkullStripper
    >>> pipe = Pipeline([SkullStripper(), Normalizer(method="zscore")])
    >>> processed, meta = pipe(volume)
    """

    def __init__(self, transforms: list[BaseTransform] | None = None):
        self.transforms = transforms or []

    def __call__(self, array: Any, metadata: dict | None = None) -> tuple[Any, dict]:
        """Apply all transforms in sequence."""
        metadata = metadata or {}
        for transform in self.transforms:
            array, metadata = transform(array, metadata)
        return array, metadata

    def append(self, transform: BaseTransform) -> "Pipeline":
        """Add a transform to the end of the pipeline."""
        self.transforms.append(transform)
        return self
