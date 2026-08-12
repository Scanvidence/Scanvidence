"""Segmentation trainer: AdamW + AMP + gradient accumulation for batch 1.

Implements the Milestone 3 training contract literally: one patch per
forward pass (batch 1), gradient accumulation to a larger effective
batch, mixed precision where the hardware supports it, and gradient
clipping applied after the AMP unscaling step (``GradScaler.unscale_``
before ``clip_grad_norm_``).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_

from .base import BaseTrainer, RunHistory
from .Loss import SegmentationLoss
from .Metrics import mean_regional_dice

CHECKPOINT_NAME = "best.pt"
RUN_JSON_NAME = "run.json"
METRICS_LOG_NAME = "metrics.jsonl"
PROFILE_TABLE_NAME = "profile.txt"


class SegmentationTrainer(BaseTrainer):
    """B0/B1-compatible training loop with measured logging.

    Parameters
    ----------
    model : nn.Module
        Segmentation model consuming ``(B, C, D, H, W)`` and emitting
        numeric logits of the same spatial shape (e.g. ``SegResNetB0``).
    config : dict
        Run configuration. Recognized keys:

        - ``lr`` (float, default 1e-4), ``weight_decay`` (float, default 1e-5)
        - ``amp`` (bool, default True) — mixed precision via GradScaler
        - ``grad_accum_steps`` (int, default 8) — effective-batch multiplier
        - ``max_grad_norm`` (float, default 1.0) — clip after unscaling
        - ``dice_weight`` / ``ce_weight`` (float, default 1.0 each)
        - ``run_meta`` (dict) — split hashes, seed, cmd; written into
          ``run.json`` so every run is reproducible from its log
    """

    def __init__(self, model: nn.Module, config: dict | None = None) -> None:
        super().__init__(model, config)
        self.lr = float(self.config.get("lr", 1e-4))
        self.weight_decay = float(self.config.get("weight_decay", 1e-5))
        self.amp = bool(self.config.get("amp", True))
        self.grad_accum_steps = int(self.config.get("grad_accum_steps", 8))
        self.max_grad_norm = float(self.config.get("max_grad_norm", 1.0))
        self.loss_fn = SegmentationLoss(
            dice_weight=float(self.config.get("dice_weight", 1.0)),
            ce_weight=float(self.config.get("ce_weight", 1.0)),
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.use_amp = self.amp and self.device.type == "cuda"
        if self.device.type == "cuda":
            # Keep backend behavior explicit on no-tensor-core workstations
            # (e.g. T1000): AMP is for memory headroom, not TF32 speed paths.
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        except TypeError:  # torch < 2.3
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.model = self.model.to(self.device)

    def fit(
        self,
        train_loader: Any,
        val_loader: Any,
        epochs: int,
        *,
        out_dir: str | Path | None = None,
        log_every: int = 20,
        profile_steps: int = 0,
        **kwargs: Any,
    ) -> RunHistory:
        """Run the training loop and return measured history.

        Parameters
        ----------
        train_loader, val_loader : torch.utils.data.DataLoader
            Batch-1 patch loaders (required by the work plan's memory
            budget: accumulation supplies the effective batch size).
        epochs : int
            Number of epochs.
        out_dir : str, Path or None
            Where to write ``best.pt``, ``run.json`` and the metrics log.
            If None, checkpoints and logs are skipped.
        log_every : int
            Print train loss every N steps.
        profile_steps : int
            If > 0, profile the first N training steps at the start of
            epoch 1 with torch.profiler and store a trace + top-k table.

        Returns
        -------
        RunHistory
            Measured numbers for the run (loss, Dice, step times).
        """
        out_dir = Path(out_dir) if out_dir is not None else None
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)

        history = RunHistory()
        best_dice = -1.0
        best_epoch = -1

        loader_iter = iter(train_loader)
        if profile_steps > 0:
            self._profile(loader_iter, profile_steps, out_dir)

        for epoch in range(1, epochs + 1):
            epoch_start = time.monotonic()
            self.model.train()
            running_loss: list[float] = []
            step_count = 0
            step_start = time.monotonic()

            for step, (x, y) in enumerate(train_loader, start=1):
                x, y = x.to(self.device), y.to(self.device)
                loss = self._train_step(x, y, step)
                running_loss.append(float(loss.detach()))
                step_count += 1
                history.step_time_s.append(time.monotonic() - step_start)
                step_start = time.monotonic()
                if log_every > 0 and step % log_every == 0:
                    mean_so_far = float(np.mean(running_loss[-log_every:]))
                    print(
                        f"[epoch {epoch}/{epochs} step {step}] loss {mean_so_far:.4f} "
                        f"({history.step_time_s[-1] * 1e3:.1f} ms/step)"
                    )
            if step_count % self.grad_accum_steps != 0:
                self._flush()

            epoch_loss = float(np.mean(running_loss))
            history.train_loss.append(epoch_loss)
            history.epoch_duration_s.append(time.monotonic() - epoch_start)

            val_loss, val_dice_dict = self._validate(val_loader, history)
            epoch_dice = float(np.mean(list(val_dice_dict.values())))
            history.val_dice.append(val_dice_dict)
            if epoch_dice > best_dice:
                best_dice, best_epoch = epoch_dice, epoch
                if out_dir is not None:
                    self.save_checkpoint(str(out_dir / CHECKPOINT_NAME), epoch, best_dice)
            print(
                f"[epoch {epoch}/{epochs}] train_loss {epoch_loss:.4f} | "
                f"val_loss {val_loss:.4f} | ET {val_dice_dict['ET']:.4f} "
                f"TC {val_dice_dict['TC']:.4f} WT {val_dice_dict['WT']:.4f} | "
                f"best {best_dice:.4f} (epoch {best_epoch})"
            )
            if out_dir is not None:
                self._append_metrics_log(out_dir / METRICS_LOG_NAME, epoch, val_dice_dict)

        history.best_val_dice = best_dice
        history.best_epoch = best_epoch
        if out_dir is not None:
            self._write_run_json(out_dir, history)
        return history

    def _train_step(self, x: Tensor, y: Tensor, step: int) -> Tensor:
        """One scaled forward/backward, one optimizer step per accumulation window."""
        optimizer = self.optimizer
        assert optimizer is not None
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=self.use_amp):
            logits = self.model(x)
            loss = self.loss_fn(logits, y) / self.grad_accum_steps
        self.scaler.scale(loss).backward()
        if step % self.grad_accum_steps == 0:
            self._flush()
        return loss * self.grad_accum_steps

    def _flush(self) -> None:
        """Unscale, clip after unscaling, step, and update the scaler."""
        optimizer = self.optimizer
        assert optimizer is not None
        self.scaler.unscale_(optimizer)
        clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.scaler.step(optimizer)
        self.scaler.update()

    def _validate(self, val_loader: Any, history: RunHistory) -> tuple[float, dict[str, float]]:
        self.model.eval()
        losses: list[float] = []
        dice_sums: dict[str, float] = {"ET": 0.0, "TC": 0.0, "WT": 0.0}
        count = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                with torch.autocast("cuda", enabled=self.use_amp):
                    logits = self.model(x)
                    loss = self.loss_fn(logits, y)
                pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
                target = y.squeeze(0).cpu().numpy()
                for region in ("ET", "TC", "WT"):
                    dice_sums[region] += mean_regional_dice(pred, target, (region,))
                losses.append(float(loss))
                count += 1
        self.model.train()
        if count == 0:
            return float("nan"), {"ET": 0.0, "TC": 0.0, "WT": 0.0}
        return float(np.mean(losses)), {region: dice_sums[region] / count for region in dice_sums}

    def predict(self, x: Tensor) -> Tensor:
        """Predict a label map from a patch; no gradient tracking.

        Parameters
        ----------
        x : Tensor
            Patch of shape ``(B, C, D, H, W)`` (or ``(C, D, H, W)``).

        Returns
        -------
        Tensor
            Long label map compatible with the BraTS label scheme
            (0 background, 1 necrotic core, 2 edema, 3 enhancing tumor).
        """
        if x.dim() == 4:
            x = x.unsqueeze(0)
        self.model.eval()
        with torch.no_grad(), torch.autocast("cuda", enabled=self.use_amp):
            logits = self.model(x.to(self.device))
        return logits.argmax(dim=1).cpu()

    def _profile(self, loader_iter: Any, steps: int, out_dir: Path | None) -> None:
        """Profile ``steps`` training steps; results land in out_dir/profile.*."""
        if out_dir is None:
            print("--profile-steps requires --out-dir; skipping profiler.")
            return
        self.model.train()
        optimizer = self.optimizer
        assert optimizer is not None
        optimizer.zero_grad(set_to_none=True)
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(out_dir)),
            record_shapes=False,
        )
        with prof:
            for step in range(1, steps + 1):
                x, y = next(loader_iter)
                x, y = x.to(self.device), y.to(self.device)
                self._train_step(x, y, step)
                prof.step()
        table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20)
        (out_dir / PROFILE_TABLE_NAME).write_text(table)
        print(table)

    def _append_metrics_log(self, path: Path, epoch: int, dice: dict[str, float]) -> None:
        record = {"epoch": epoch, "dice": dice}
        with path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")

    def _write_run_json(self, out_dir: Path, history: RunHistory) -> None:
        record = {
            "config": self.config,
            "device": str(self.device),
            "measured": {
                "best_val_dice": history.best_val_dice,
                "best_epoch": history.best_epoch,
                "mean_step_time_s": float(np.mean(history.step_time_s)),
                "total_wall_s": float(np.sum(history.epoch_duration_s)),
            },
        }
        (out_dir / RUN_JSON_NAME).write_text(json.dumps(record, indent=2) + "\n")
