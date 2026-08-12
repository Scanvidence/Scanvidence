"""Loss functions for 3D brain tumor segmentation."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SegmentationLoss(nn.Module):
    """Soft Dice + cross-entropy combined loss for B0 training.

    Classes with no voxels in the batch's target are excluded from the
    Dice term. A region that is empty in both prediction and target would
    otherwise contribute a spurious constant ~0.5 loss per empty class
    (the BraTS empty-region convention is applied at metric time, in
    ``training.Metrics``, where it belongs).

    Parameters
    ----------
    dice_weight : float
        Weight of the soft Dice term.
    ce_weight : float
        Weight of the voxelwise cross-entropy term.
    smooth : float
        Additive smoothing in both numerator and denominator.
    class_weights : Tensor or None
        Per-class weights for dropout-style class imbalance in both terms.
    ignore_empty_classes : bool
        Drop target-empty classes from the Dice term (default).
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        smooth: float = 1e-5,
        class_weights: Tensor | None = None,
        ignore_empty_classes: bool = True,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth
        self.class_weights = class_weights
        self.ignore_empty_classes = ignore_empty_classes

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """Compute the combined loss.

        Parameters
        ----------
        logits : Tensor
            Shape ``(B, C, D, H, W)``.
        target : Tensor
            Integer labels, shape ``(B, D, H, W)``.

        Returns
        -------
        Tensor
            Scalar loss.
        """
        probabilities = F.softmax(logits, dim=1)
        one_hot = F.one_hot(target, num_classes=probabilities.shape[1]).permute(0, 4, 1, 2, 3)
        ones = torch.ones_like(one_hot, dtype=logits.dtype)
        intersection = (probabilities * one_hot).sum(dim=(0, 2, 3, 4))
        pred_volume = (probabilities * ones).sum(dim=(0, 2, 3, 4))
        target_volume = (one_hot * ones).sum(dim=(0, 2, 3, 4))

        if self.ignore_empty_classes and self.dice_weight > 0.0:
            active = target_volume > 0
            if active.any():
                dice = (2.0 * intersection[active] + self.smooth) / (
                    pred_volume[active] + target_volume[active] + self.smooth
                )
                dice_loss = 1.0 - dice.mean()
            else:
                dice_loss = probabilities.new_zeros(())
        else:
            dice = (2.0 * intersection + self.smooth) / (pred_volume + target_volume + self.smooth)
            dice_loss = 1.0 - dice.mean()

        ce = F.cross_entropy(logits, target, weight=self.class_weights)
        return self.dice_weight * dice_loss + self.ce_weight * ce
