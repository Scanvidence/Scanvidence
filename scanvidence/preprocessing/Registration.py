"""Spatial registration for MRI volumes."""

from .base import BaseTransform


class Registration(BaseTransform):
    """Register MRI volumes to a template space.

    Parameters
    ----------
    template : str
        Path to the template volume (e.g., MNI152).
    method : str
        Registration method. Options: ``"rigid"``, ``"affine"``, ``"deformable"``.
    """

    def __init__(self, template: str | None = None, method: str = "affine"):
        self.template = template
        self.method = method

    def __call__(self, array, metadata=None):
        """Apply spatial registration."""
        # Stub — implementation requires SimpleITK or ANTs
        return array, metadata or {}
