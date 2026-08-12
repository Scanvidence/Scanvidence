"""Abstract base trainer (pgmpy pattern) and run-history record.

A trainer owns the fit/predict lifecycle for one model and records only
measured numbers (ground rule 1: measure, don't estimate). Anything a
thesis table might cite — train loss, validation Dice, step times, best
checkpoint — lands in :class:`RunHistory` or in the checkpoint file, and
nowhere else.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn


@dataclass
class RunHistory:
    """Measured results of one training run (never hand-written numbers).

    Attributes
    ----------
    train_loss : list of float
        Mean loss per completed epoch.
    val_loss : list of float
        Mean validation loss per epoch.
    val_dice : list of dict
        Per-epoch regional Dice scores with keys ``ET``, ``TC``, ``WT``.
    best_val_dice : float
        Best epoch-mean of ET/TC/WT Dice seen during training.
    best_epoch : int
        Epoch (1-based) when ``best_val_dice`` was reached.
    step_time_s : list of float
        Wall-clock seconds per training step, measured during the run.
    epoch_duration_s : list of float
        Wall-clock seconds per completed epoch.
    """

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_dice: list[dict[str, float]] = field(default_factory=list)
    best_val_dice: float = 0.0
    best_epoch: int = -1
    step_time_s: list[float] = field(default_factory=list)
    epoch_duration_s: list[float] = field(default_factory=list)


class BaseTrainer(ABC):
    """Abstract base class for training loops.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    config : dict or None
        Complete configuration of the run (architecture, optimizer,
        schedule, metrics). Saved verbatim into checkpoints so a
        checkpoint is a full record of how it was produced.
    """

    def __init__(self, model: nn.Module, config: dict | None = None):
        self.model = model
        self.config = config or {}
        self.optimizer: torch.optim.Optimizer | None = None
        self.loss_fn = nn.Module()
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")

    @abstractmethod
    def fit(
        self,
        train_loader: Any,
        val_loader: Any,
        epochs: int,
        **kwargs: Any,
    ) -> RunHistory:
        """Train ``self.model`` and return the measured :class:`RunHistory`."""
        pass

    @abstractmethod
    def predict(self, x: Tensor) -> Tensor:
        """Run inference; returns a tensor without gradient tracking."""
        pass

    def save_checkpoint(self, path: str, epoch: int, best_val_dice: float) -> None:
        """Persist model weights plus the full run config.

        Parameters
        ----------
        path : str
            Destination file path.
        epoch : int
            Epoch the checkpoint was written at.
        best_val_dice : float
            Best measured validation Dice at this point of the run.
        """
        torch.save(
            {
                "config": self.config,
                "state_dict": self.model.state_dict(),
                "epoch": epoch,
                "best_val_dice": best_val_dice,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> dict:
        """Load weights and the recorded config from a checkpoint.

        Parameters
        ----------
        path : str
            Checkpoint file produced by :meth:`save_checkpoint`.

        Returns
        -------
        dict
            The checkpoint contents (config, epoch, best_val_dice).
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.config = checkpoint["config"]
        self.model.load_state_dict(checkpoint["state_dict"])
        return checkpoint
