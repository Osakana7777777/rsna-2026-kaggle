"""Compact multi-view convolutional model for knee MRI studies."""

from __future__ import annotations

import torch
from torch import nn


def convolution_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Create one downsampling feature block."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
        nn.MaxPool2d(2),
    )


class KneeAbnormalityModel(nn.Module):
    """Encode individual slices and pool them into study-level predictions."""

    def __init__(self, num_targets: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            convolution_block(1, 24),
            convolution_block(24, 48),
            convolution_block(48, 96),
            convolution_block(96, 160),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(160),
            nn.Dropout(0.25),
            nn.Linear(160, num_targets),
        )

    def forward(self, images: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        batch_size, num_views, channels, height, width = images.shape
        features = self.encoder(
            images.reshape(batch_size * num_views, channels, height, width)
        ).reshape(batch_size, num_views, -1)
        weights = present.to(features.dtype).unsqueeze(-1)
        pooled = (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.head(pooled)
