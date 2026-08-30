"""3D Gradient-weighted Class Activation Mapping (Grad-CAM) for 3D CNNs."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .base import BaseExplainer


class GradCAM(BaseExplainer):
    """3D Grad-CAM explainability for volumetric segmentation models (SegResNetB0).

    Computes class-discriminative 3D spatial attribution heatmaps by capturing
    feature activations and backpropagating gradients through target 3D conv layers.

    Parameters
    ----------
    model : torch.nn.Module
        The 3D CNN model (e.g., SegResNetB0).
    target_layer : torch.nn.Module or str, optional
        Target 3D convolutional module to attach hooks to. If None, targets the
        final decoder block convolution.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: torch.nn.Module | str | None = None,
    ) -> None:
        super().__init__(model)
        self.model = model
        self.target_layer = target_layer or self._find_default_target_layer()

        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._hooks: list[Any] = []
        self._register_hooks()

    def _find_default_target_layer(self) -> torch.nn.Module:
        """Find the last Conv3d layer in the decoder."""
        if hasattr(self.model, "decoder") and len(self.model.decoder) > 0:
            last_dec = self.model.decoder[-1]
            if hasattr(last_dec, "block") and hasattr(last_dec.block, "conv2"):
                return last_dec.block.conv2
        for module in reversed(list(self.model.modules())):
            if isinstance(module, torch.nn.Conv3d):
                return module
        raise ValueError("Could not find a Conv3d layer in the model.")

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on target layer."""
        target = (
            dict(self.model.named_modules())[self.target_layer]
            if isinstance(self.target_layer, str)
            else self.target_layer
        )

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self._hooks.append(target.register_forward_hook(forward_hook))
        self._hooks.append(target.register_full_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        """Clean up PyTorch hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def generate_heatmap(
        self,
        x: torch.Tensor,
        target_class: int = 3,
    ) -> np.ndarray:
        """Generate 3D Grad-CAM heatmap for an input MRI tensor.

        Parameters
        ----------
        x : torch.Tensor of shape (1, 4, D, H, W)
            Input 4-channel MRI scan.
        target_class : int, default=3 (Enhancing Tumor)
            Target segmentation class (1: NCR, 2: ED, 3: ET).

        Returns
        -------
        heatmap : np.ndarray of shape (D, H, W)
            Normalized [0, 1] 3D spatial attribution heatmap.
        """
        self.model.eval()
        self.model.zero_grad()

        if x.dim() == 4:
            x = x.unsqueeze(0)

        # Forward pass
        logits = self.model(x)

        # Compute class softmax probability sum for target
        probs = F.softmax(logits, dim=1)
        score = probs[:, target_class].sum()
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to capture gradients or activations.")

        # 3D Global Average Pooling of gradients: alpha_k = 1/Z sum(dL/dA_k)
        weights = torch.mean(self.gradients, dim=(2, 3, 4), keepdim=True)

        # Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        if float(cam.detach().max()) == 0.0:
            cam = torch.abs(torch.sum(weights * self.activations, dim=1, keepdim=True))

        # Upsample 3D CAM to original input dimensions (D, H, W)
        cam = F.interpolate(cam, size=x.shape[2:], mode="trilinear", align_corners=False)
        cam_np = cam.squeeze().detach().cpu().numpy()

        # Min-max normalization
        c_min, c_max = float(cam_np.min()), float(cam_np.max())
        return (cam_np - c_min) / (c_max - c_min) if c_max > c_min else np.zeros_like(cam_np)

    def explain(self, scan: Any, target_class: int = 3, **kwargs) -> dict[str, Any]:
        """Generate Grad-CAM explanation dictionary."""
        scan_t = torch.from_numpy(scan).float() if isinstance(scan, np.ndarray) else scan
        heatmap = self.generate_heatmap(scan_t, target_class=target_class)
        return {
            "method": "GradCAM_3D",
            "heatmap": heatmap,
            "target_class": target_class,
            "max_activation_coords": list(np.unravel_index(np.argmax(heatmap), heatmap.shape)),
        }
