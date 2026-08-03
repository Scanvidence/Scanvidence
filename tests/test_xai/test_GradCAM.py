"""Tests for the Grad-CAM explainer."""

from scanvidence.xai import BaseExplainer
from scanvidence.xai.GradCAM import GradCAM


def test_gradcam_is_base_explainer_subclass():
    assert issubclass(GradCAM, BaseExplainer)
