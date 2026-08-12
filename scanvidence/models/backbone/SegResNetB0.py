"""B0 — compact SegResNet-style 3D CNN, the independent reference baseline.

This is the architecture-controlled reference model from the Scanvidence
work plan: a plain residual encoder–decoder with no Transformer. The same
stem, decoder, and training budget are later shared by B1, H0, and H1 so
the only thing that changes between models is the mechanism under test.

Design (deliberately boring and reproducible):

- Pre-activation residual blocks (GroupNorm, ReLU, Conv3d) x2, the
  SegResNet building block ordering.
- Encoder widths ``(16, 32, 64, 128)`` with a ``MaxPool3d(2)`` between
  stages; a final unsampled pool produces the ``6**3`` bottleneck from a
  ``96**3`` patch, matching the H0/H1 lattice resolution.
- U-Net decoder: ``ConvTranspose3d`` upsampling followed by skip
  concatenation and one residual block per stage.
- GroupNorm everywhere, ``Dropout3d`` optional (off by default).
- ``1x1x1`` head to ``num_classes`` (BraTS-GLI: 0 background, 1 necrosis,
  2 edema, 3 enhancing tumor).

Constraints configured at construction time so a training config file is
a complete record of the run:

- ``in_channels`` : number of MRI sequences (4 for BraTS, 1 for T1w
  self-supervised pretraining).
- ``widths`` : encoder feature widths; the number of entries fixes the
  number of down/upsampling stages. Patch edge lengths must be divisible
  by ``2 ** len(widths)`` (96 -> 6 for the default 4-stage net).
- Per-stage parameters: parameter count is measurable at import time via
  ``parameter_count`` — never estimated from paper claims (ground rule 1).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

DEFAULT_WIDTHS: tuple[int, ...] = (16, 32, 64, 128)
DEFAULT_IN_CHANNELS = 4
DEFAULT_NUM_CLASSES = 4
DEFAULT_NUM_GROUPS = 8


def _groups_for(requested: int, channels: int) -> int:
    """Largest divisor of ``channels`` that is <= ``requested`` groups."""
    groups = min(requested, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


class _ConvBlock(nn.Module):
    """Pre-activation SegResNet residual block.

    Applies ``(GroupNorm, ReLU, Conv3d)`` twice with a skip shortcut.
    When ``in_channels != out_channels`` the shortcut is a ``1x1x1``
    convolution; otherwise it is the identity.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int = DEFAULT_NUM_GROUPS,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        groups_in = _groups_for(num_groups, in_channels)
        groups_out = _groups_for(num_groups, out_channels)
        self.norm1 = nn.GroupNorm(groups_in, in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups_out, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0.0 else nn.Identity()
        self.shortcut: nn.Module
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.shortcut(x)
        out = torch.relu(self.norm1(x))
        out = self.conv1(out)
        out = torch.relu(self.norm2(out))
        out = self.conv2(out)
        return self.dropout(out + residual)


class _EncoderBlock(nn.Module):
    """One encoder stage: ``MaxPool3d(2)`` followed by a residual block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2)
        self.block = _ConvBlock(in_channels, out_channels, num_groups, dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(self.pool(x))


class _DecoderBlock(nn.Module):
    """One U-Net decoder stage: transposed-conv upsample, skip concat, block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        num_groups: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = _ConvBlock(out_channels + skip_channels, out_channels, num_groups, dropout)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        out = self.up(x)
        return self.block(torch.cat([out, skip], dim=1))


class SegResNetB0(nn.Module):
    """B0 reference segmentation CNN (compact SegResNet with U-Net decoder).

    Parameters
    ----------
    in_channels : int
        Number of input MRI sequences (4 for multi-sequence BraTS).
    num_classes : int
        Number of output classes (4 for the BraTS-GLI label map).
    widths : Sequence[int]
        Encoder width per stage. Length defines the number of
        down/upsampling stages; spatial dims must be divisible by
        ``2 ** len(widths)``.
    num_groups : int
        Group count for GroupNorm (capped at the channel count per layer).
    dropout : float
        Dropout3d rate inside residual blocks (``0.0`` = off).

    Notes
    -----
    - Batch size 1 on 96-cubed, 4-channel patches is the design point;
      nothing here requires a particular batch size.
    - ``from_config`` validates the config dict so experiment YAMLs stay
      the single record of what produced a run.
    """

    def __init__(
        self,
        in_channels: int = DEFAULT_IN_CHANNELS,
        num_classes: int = DEFAULT_NUM_CLASSES,
        widths: Sequence[int] = DEFAULT_WIDTHS,
        num_groups: int = DEFAULT_NUM_GROUPS,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self._validate(in_channels, num_classes, widths, num_groups, dropout)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.widths = tuple(widths)
        self.num_groups = num_groups
        self.dropout = dropout

        self.stem = _ConvBlock(in_channels, widths[0], num_groups, dropout)
        self.encoder = nn.ModuleList(
            [
                _EncoderBlock(prev, width, num_groups, dropout)
                for prev, width in zip(widths[:-1], widths[1:], strict=True)
            ]
        )
        decoder_pairs = list(zip(reversed(widths), reversed(widths[:-1]), strict=False)) + [
            (widths[0], widths[0])
        ]
        self.decoder = nn.ModuleList(
            [
                _DecoderBlock(in_channels, out_channels, in_channels, num_groups, dropout)
                for in_channels, out_channels in decoder_pairs
            ]
        )
        self.head = nn.Conv3d(widths[0], num_classes, kernel_size=1)

    @staticmethod
    def _validate(
        in_channels: int,
        num_classes: int,
        widths: Sequence[int],
        num_groups: int,
        dropout: float,
    ) -> None:
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}.")
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}.")
        if not 2 <= len(widths) <= 5:
            raise ValueError(f"widths must have 2..5 stages, got {len(widths)}.")
        if any(w < 1 for w in widths):
            raise ValueError(f"Every width must be >= 1, got {widths}.")
        if num_groups < 1:
            raise ValueError(f"num_groups must be >= 1, got {num_groups}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> SegResNetB0:
        """Build a model from a config dict, rejecting unknown keys.

        Parameters
        ----------
        config : Mapping[str, Any]
            May contain any of ``in_channels``, ``num_classes``,
            ``widths``, ``num_groups``, ``dropout``.

        Returns
        -------
        SegResNetB0
            The configured model.

        Raises
        ------
        ValueError
            If the config contains keys this architecture does not
            understand or values that fail validation.
        """
        known = {"in_channels", "num_classes", "widths", "num_groups", "dropout"}
        unknown = set(config) - known
        if unknown:
            raise ValueError(f"Unknown SegResNetB0 config keys: {sorted(unknown)}.")
        return cls(
            in_channels=int(config.get("in_channels", DEFAULT_IN_CHANNELS)),
            num_classes=int(config.get("num_classes", DEFAULT_NUM_CLASSES)),
            widths=tuple(int(w) for w in config.get("widths", DEFAULT_WIDTHS)),
            num_groups=int(config.get("num_groups", DEFAULT_NUM_GROUPS)),
            dropout=float(config.get("dropout", 0.0)),
        )

    @property
    def parameter_count(self) -> int:
        """Number of trainable parameters (measured, not estimated)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: Tensor) -> Tensor:
        """Segment a 3D patch.

        Parameters
        ----------
        x : Tensor
            Patch of shape ``(B, in_channels, D, H, W)`` with each spatial
            dim divisible by ``2 ** len(widths)``.

        Returns
        -------
        Tensor
            Logits of shape ``(B, num_classes, D, H, W)``.

        Raises
        ------
        ValueError
            If the input rank or spatial divisibility constraint is not met.
        """
        if x.dim() != 5:
            raise ValueError(f"Expected a 5-D tensor (B, C, D, H, W), got dim {x.dim()}.")
        stride = 2 ** len(self.widths)
        spatial = tuple(x.shape[2:])
        if any(s % stride != 0 for s in spatial):
            raise ValueError(
                f"Spatial dims must be divisible by 2 ** len(widths) = {stride} "
                f"(96 -> 6 for the default net); got {spatial}."
            )
        skips = [self.stem(x)]
        for block in self.encoder:
            skips.append(block(skips[-1]))
        out = nn.functional.max_pool3d(skips[-1], kernel_size=2)
        for block, skip in zip(self.decoder, reversed(skips), strict=True):
            out = block(out, skip)
        return self.head(out)
