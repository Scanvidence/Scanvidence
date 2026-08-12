"""End-to-end tests for the segmentation trainer on CPU with tiny models."""

import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from scanvidence.data.datasets import BraTSDataset
from scanvidence.models.backbone import SegResNetB0
from scanvidence.training import SegmentationTrainer
from scanvidence.training.base import BaseTrainer
from scanvidence.training.BraTSPatchDataset import BraTSPatchDataset


def _loaders(root: str, seed: int = 0, patch: int = 16) -> tuple[DataLoader, DataLoader]:
    records = BraTSDataset(root, track="GLI").discover()
    train_dataset = BraTSPatchDataset(records, patch_size=patch, seed=seed, augment=False)
    val_dataset = BraTSPatchDataset(records, patch_size=patch, seed=seed + 1, augment=False)
    train_gen = torch.Generator().manual_seed(seed)
    val_gen = torch.Generator().manual_seed(seed + 1)
    return (
        DataLoader(train_dataset, batch_size=1, shuffle=True, generator=train_gen),
        DataLoader(val_dataset, batch_size=1, shuffle=False, generator=val_gen),
    )


def _trainer(seed: int = 17, **overrides) -> SegmentationTrainer:
    model = SegResNetB0(in_channels=4, num_classes=4, widths=(2, 4, 8, 16))
    config = {
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "amp": False,
        "grad_accum_steps": 2,
        "max_grad_norm": 1.0,
    }
    config.update(overrides)
    torch.manual_seed(seed)
    return SegmentationTrainer(model, config)


def test_trainer_subclasses_base():
    assert issubclass(SegmentationTrainer, BaseTrainer)


def test_fit_records_measured_history(synthetic_brats, tmp_path):
    root, _ = synthetic_brats
    torch.manual_seed(0)
    trainer = _trainer()
    train_loader, val_loader = _loaders(root)
    history = trainer.fit(train_loader, val_loader, epochs=2, out_dir=tmp_path / "run")
    assert len(history.train_loss) == 2
    assert len(history.val_dice) == 2
    assert history.best_epoch in (1, 2)
    assert history.best_val_dice > 0.0
    assert len(history.step_time_s) == len(train_loader.dataset) * 2
    assert len(history.epoch_duration_s) == 2
    assert (tmp_path / "run" / "best.pt").exists()
    assert (tmp_path / "run" / "run.json").exists()
    run = json.loads((tmp_path / "run" / "run.json").read_text())
    assert "measured" in run and "best_val_dice" in run["measured"]


def test_fit_uses_gradient_accumulation_final_flush(synthetic_brats):
    root, _ = synthetic_brats
    torch.manual_seed(0)
    trainer = _trainer(grad_accum_steps=5)  # 3 patches -> leftover flush path
    history = trainer.fit(*_loaders(root), epochs=1, out_dir=None, log_every=0)
    assert len(history.train_loss) == 1
    assert np.isfinite(history.train_loss).all()


def test_fit_is_deterministic_for_same_seed(synthetic_brats):
    root, _ = synthetic_brats
    torch.manual_seed(0)
    a = _trainer(seed=99)
    history_a = a.fit(*_loaders(root, seed=99), epochs=2, out_dir=None, log_every=0)
    torch.manual_seed(0)
    b = _trainer(seed=99)
    history_b = b.fit(*_loaders(root, seed=99), epochs=2, out_dir=None, log_every=0)
    assert history_a.train_loss == history_b.train_loss
    assert history_a.best_val_dice == pytest.approx(history_b.best_val_dice)


def test_predict_returns_argmax_label_map(synthetic_brats):
    root, _ = synthetic_brats
    torch.manual_seed(0)
    trainer = _trainer()
    trainer.fit(*_loaders(root), epochs=1, out_dir=None, log_every=0)
    x = torch.randn(1, 4, 16, 16, 16)
    pred = trainer.predict(x)
    assert pred.shape == (1, 16, 16, 16)
    assert pred.dtype == torch.int64
    assert set(torch.unique(pred).tolist()) <= {0, 1, 2, 3}
    assert not pred.requires_grad


def test_checkpoint_round_trip(synthetic_brats, tmp_path):
    root, _ = synthetic_brats
    torch.manual_seed(0)
    trainer = _trainer()
    path = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(str(path), epoch=3, best_val_dice=0.5)
    clone = _trainer()
    checkpoint = clone.load_checkpoint(str(path))
    assert checkpoint["best_val_dice"] == 0.5
    loaded = next(clone.model.parameters())
    original = next(trainer.model.parameters())
    assert torch.equal(loaded, original)


def test_trainer_loss_improves_on_overfit_small_data(tmp_path):
    """Two cases, one model: loss must go down and Dice up (the 2-case gate)."""
    from tests.test_training.conftest import write_case

    root = tmp_path / "two"
    write_case(root, "BraTS-GLI-00050-000", seed=0)
    write_case(root, "BraTS-GLI-00051-000", seed=1)
    records = BraTSDataset(str(root), track="GLI").discover()
    dataset = BraTSPatchDataset(records, patch_size=32, seed=17, augment=False)
    gen = torch.Generator().manual_seed(17)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, generator=gen)
    torch.manual_seed(17)
    trainer = SegmentationTrainer(
        SegResNetB0(in_channels=4, num_classes=4, widths=(4, 8, 16, 32)),
        {"lr": 1e-3, "weight_decay": 1e-5, "amp": False, "grad_accum_steps": 1},
    )
    # Memorization gate semantics: validate on the same patch set (same seed).
    history = trainer.fit(loader, loader, epochs=15, out_dir=None, log_every=0)
    assert history.train_loss[-1] < history.train_loss[0]
    assert history.best_val_dice > 0.30
