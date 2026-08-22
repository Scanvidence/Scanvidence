"""Shared CNN building blocks (stem, encoder, decoder) for B1 and future H0.

These modules form the architecture-controlled skeleton shared by the B1
CNN baseline and the H0 hybrid. B1 uses them directly; H0 will import
them and insert a ViT + fusion block between the encoder and decoder,
keeping all channel widths and spatial resolutions identical so the
only difference between B1 and H0 is the mechanism under test.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class ResBlock(nn.Module):
    """Pre-activation residual block with GroupNorm and SiLU.

    Two 3x3x3 convolutions with residual shortcut. When used inside
    the shared stem/encoder/decoder the channel count is constant,
    so the shortcut is an identity.
    """

    def __init__(self, channels: int, num_groups: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(num_groups, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class SharedStem(nn.Module):
    """Stem: 4x96^3 input -> 16x48^3 features.

    Strided 3x3x3 convolution (factor 2) followed by GroupNorm, SiLU,
    and one residual block.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 16,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU(),
            ResBlock(out_channels, num_groups),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class CNNEncoder(nn.Module):
    """Encoder: 48^3 -> 6^3, returning skip connections at each stage.

    Each stage: downsample by 2 (strided 3x3x3 conv with GroupNorm+SiLU)
    followed by a residual block. Spatial resolution halves at each step.
    """

    def __init__(
        self,
        widths: Sequence[int] = (16, 32, 64, 128),
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        if len(widths) < 2:
            raise ValueError("widths must have at least two entries")
        self.widths = tuple(widths)
        self.num_groups = num_groups

        self.downs = nn.ModuleList()
        self.stages = nn.ModuleList()
        for i in range(1, len(widths)):
            self.downs.append(
                nn.Sequential(
                    nn.GroupNorm(num_groups, widths[i - 1]),
                    nn.SiLU(),
                    nn.Conv3d(widths[i - 1], widths[i], kernel_size=3, stride=2, padding=1),
                )
            )
            self.stages.append(ResBlock(widths[i], num_groups))

    def forward(self, x: Tensor) -> list[Tensor]:
        """Return skip list: [stem_out, stage1_out, stage2_out, stage3_out]."""
        skips = [x]
        for down, stage in zip(self.downs, self.stages, strict=True):
            x = stage(down(x))
            skips.append(x)
        return skips


class UNetDecoder(nn.Module):
    """Decoder: skip connections -> 4x96^3 logits.

    Upsamples with ConvTranspose3d (factor 2), concatenates skip,
    applies 3x3x3 conv + residual block. Final head upsamples by 2
    to recover 96^3, then 1x1x1 to num_classes.
    """

    def __init__(
        self,
        widths: Sequence[int] = (16, 32, 64, 128),
        num_classes: int = 4,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        if len(widths) < 2:
            raise ValueError("widths must have at least two entries")
        self.widths = tuple(widths)
        self.num_classes = num_classes

        ws = list(widths)
        self.ups = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for i in range(len(ws) - 1, 0, -1):
            self.ups.append(nn.ConvTranspose3d(ws[i], ws[i - 1], kernel_size=2, stride=2))
            self.blocks.append(
                nn.Sequential(
                    nn.Conv3d(ws[i - 1] * 2, ws[i - 1], kernel_size=3, padding=1),
                    ResBlock(ws[i - 1], num_groups),
                )
            )
        self.head = nn.Sequential(
            nn.GroupNorm(num_groups, ws[0]),
            nn.SiLU(),
            nn.Conv3d(ws[0], num_classes, kernel_size=1),
        )

    def forward(self, skips: list[Tensor]) -> Tensor:
        x = skips[-1]
        for up, blk, skip in zip(self.ups, self.blocks, reversed(skips[:-1]), strict=True):
            x = blk(torch.cat([up(x), skip], dim=1))
        # Final 2x upsample to 96^3
        x = torch.nn.functional.interpolate(
            x, scale_factor=2, mode="trilinear", align_corners=False
        )
        return self.head(x)


class B1Segmentor(nn.Module):
    """B1 architecture-controlled CNN baseline.

    Composed of SharedStem + CNNEncoder + UNetDecoder. No Transformer,
    no fusion — the "skeleton" that H0 will extend by inserting a ViT
    and fusion block between encoder and decoder.

    Parameters
    ----------
    in_channels : int
        Number of input MRI sequences (4 for BraTS).
    num_classes : int
        Number of segmentation classes (4 for BraTS-GLI: 0 bg, 1 necrotic,
        2 edema, 3 enhancing).
    widths : Sequence[int]
        Encoder/decoder channel widths per stage.
    num_groups : int
        GroupNorm group count (capped per layer).
    dropout : float
        Dropout rate (unused in current design, kept for API compatibility).
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        widths: Sequence[int] = (16, 32, 64, 128),
        num_groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}")
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        if len(widths) < 2:
            raise ValueError(f"widths must have at least two entries, got {widths}")
        if any(w < 1 for w in widths):
            raise ValueError(f"every width must be >= 1, got {widths}")
        if num_groups < 1:
            raise ValueError(f"num_groups must be >= 1, got {num_groups}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.widths = tuple(widths)
        self.num_groups = num_groups
        self.dropout = dropout

        self.stem = SharedStem(in_channels, widths[0], num_groups)
        self.encoder = CNNEncoder(widths, num_groups)
        self.decoder = UNetDecoder(widths, num_classes, num_groups)

    @property
    def parameter_count(self) -> int:
        """Number of trainable parameters (measured, not estimated)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: Tensor) -> Tensor:
        """Segment a 3D patch.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(B, C, D, H, W)`` where spatial dims are
            divisible by ``2 ** len(widths)`` (96 -> 6 for default 4 stages).

        Returns
        -------
        Tensor
            Logits of shape ``(B, num_classes, D, H, W)``.
        """
        if x.dim() != 5:
            raise ValueError(f"Expected 5-D tensor (B, C, D, H, W), got dim {x.dim()}.")
        stride = 2 ** len(self.widths)
        spatial = tuple(x.shape[2:])
        if any(s % stride != 0 for s in spatial):
            raise ValueError(
                f"Spatial dims must be divisible by 2 ** len(widths) = {stride}; got {spatial}."
            )
        return self.decoder(self.encoder(self.stem(x)))
