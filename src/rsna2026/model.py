"""Study-level DINOv2 model for knee MRI slices."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def find_dinov2_source(input_root: str | Path, variant: str) -> Path:
    """Find exactly one mounted local DINOv2 model of the requested variant."""
    root = Path(input_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Model input root does not exist: {root}")
    candidates = []
    for directory, names, files in os.walk(root):
        names[:] = [
            name for name in names if name not in {"train_series", "test_series"}
        ]
        path = Path(directory)
        if (
            "config.json" in files
            and "dinov2" in str(path).lower()
            and variant.lower() in str(path).lower()
        ):
            candidates.append(path)
    candidates.sort()
    if len(candidates) != 1:
        raise FileNotFoundError(
            "Expected exactly one mounted DINOv2 "
            f"{variant} model, found {candidates}"
        )
    return candidates[0]


class KneeAbnormalityModel(nn.Module):
    """Encode individual grayscale slices with DINOv2 and mean-pool views."""

    def __init__(
        self,
        backbone: nn.Module,
        num_targets: int,
        unfreeze_last: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if unfreeze_last <= 0:
            raise ValueError("At least one DINOv2 encoder block must be trainable")
        self.backbone = backbone
        layers = self._encoder_layers()
        if unfreeze_last > len(layers):
            raise ValueError(
                f"Cannot unfreeze {unfreeze_last} of {len(layers)} encoder blocks"
            )
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for layer in layers[-unfreeze_last:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        layernorm = getattr(self.backbone, "layernorm", None)
        if layernorm is None:
            raise TypeError("DINOv2 backbone does not expose its final layernorm")
        for parameter in layernorm.parameters():
            parameter.requires_grad = True

        hidden_size = int(self.backbone.config.hidden_size)
        feature_size = hidden_size * 2
        self.head = nn.Sequential(
            nn.LayerNorm(feature_size),
            nn.Dropout(dropout),
            nn.Linear(feature_size, num_targets),
        )
        self.unfreeze_last = unfreeze_last
        self.feature_size = feature_size
        self.register_buffer(
            "image_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        )

    def _encoder_layers(self) -> nn.ModuleList:
        encoder = getattr(self.backbone, "encoder", None)
        layers = getattr(encoder, "layer", None)
        if not isinstance(layers, nn.ModuleList):
            raise TypeError("DINOv2 backbone does not expose encoder.layer")
        return layers

    def forward(self, images: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        batch_size, num_views, channels, height, width = images.shape
        if channels != 1:
            raise ValueError(f"Expected one grayscale channel, received {channels}")
        pixels = images.reshape(batch_size * num_views, 1, height, width)
        pixels = pixels.expand(-1, 3, -1, -1)
        pixels = (pixels - self.image_mean) / self.image_std
        tokens = self.backbone(pixel_values=pixels).last_hidden_state
        if tokens.shape[1] < 2:
            raise ValueError("DINOv2 output did not contain patch tokens")
        features = torch.cat((tokens[:, 0], tokens[:, 1:].mean(dim=1)), dim=1)
        features = features.reshape(batch_size, num_views, self.feature_size)
        weights = present.to(features.dtype).unsqueeze(-1)
        pooled = (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.head(pooled)


def build_dinov2_model(
    input_root: str | Path,
    variant: str,
    num_targets: int,
    unfreeze_last: int,
    dropout: float,
) -> tuple[KneeAbnormalityModel, Path]:
    """Load a mounted DINOv2 model without any network access."""
    from transformers import AutoModel

    source = find_dinov2_source(input_root, variant)
    backbone = AutoModel.from_pretrained(str(source), local_files_only=True)
    model = KneeAbnormalityModel(
        backbone=backbone,
        num_targets=num_targets,
        unfreeze_last=unfreeze_last,
        dropout=dropout,
    )
    return model, source
