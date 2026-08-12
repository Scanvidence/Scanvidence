"""Tests for the Dice + CE segmentation loss."""

import pytest
import torch

from scanvidence.training import SegmentationLoss


def _make(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return logits.requires_grad_(), target


def test_perfect_prediction_is_near_zero_loss():
    target = torch.zeros(1, 8, 8, 8, dtype=torch.long)
    target[0, 2:6, 2:6, 2:6] = 3
    target[0, 1:3, 1:3, 1:3] = 1
    target[0, 6:8, 0:4, 6:8] = 2
    logits = torch.nn.functional.one_hot(target, num_classes=4).permute(0, 4, 1, 2, 3).float()
    logits = logits * 10.0
    loss = SegmentationLoss()(logits, target)
    assert loss.item() < 0.1


def test_worst_prediction_has_dice_contribution():
    target = torch.zeros(1, 8, 8, 8, dtype=torch.long)
    target[0, 2:6, 2:6, 2:6] = 3
    wrong = torch.zeros(1, 4, 8, 8, 8)
    wrong[:, 0] = 10.0  # everything predicted background
    loss = SegmentationLoss()(wrong, target)
    assert loss.item() > 0.5


def test_empty_targets_are_not_penalized_by_dice():
    target = torch.zeros(1, 8, 8, 8, dtype=torch.long)  # all background
    logits = torch.zeros(1, 4, 8, 8, 8)
    logits[:, 0] = 1.0
    loss = SegmentationLoss()(logits, target)
    assert torch.isfinite(loss)
    loss = SegmentationLoss()(logits.requires_grad_(), target)
    loss.backward()
    assert loss.item() > 0.0  # CE still nonzero, Dice term skipped


def test_gradients_flow_and_are_finite():
    target = torch.zeros(1, 16, 16, 16, dtype=torch.long)
    target[0, 4:12, 4:12, 4:12] = 2
    logits = torch.randn(1, 4, 16, 16, 16, requires_grad=True)
    loss = SegmentationLoss()(logits, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_weights_change_the_loss():
    target = torch.zeros(1, 8, 8, 8, dtype=torch.long)
    target[0, 2:6, 2:6, 2:6] = 3
    logits = torch.randn(1, 4, 8, 8, 8)
    plain = SegmentationLoss(dice_weight=1.0, ce_weight=1.0)(logits, target)
    dice_only = SegmentationLoss(dice_weight=1.0, ce_weight=0.0)(logits, target)
    assert dice_only.item() < plain.item()


def test_shapes_are_validated_by_forward():
    target = torch.zeros(1, 8, 8, 8, dtype=torch.long)
    logits = torch.zeros(1, 4, 8, 8)  # wrong rank
    with pytest.raises(RuntimeError):
        SegmentationLoss()(logits, target)
