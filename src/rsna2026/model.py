"""Study-level DINOv2 model for knee MRI slices."""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch
from torch import nn


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TargetAttentionHead(nn.Module):
    """Use a separate masked slot distribution for every target."""

    def __init__(
        self,
        feature_size: int,
        num_slots: int,
        num_targets: int,
        hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if num_slots <= 0 or num_targets <= 0 or hidden_size <= 0:
            raise ValueError("Slot, target, and attention sizes must be positive")
        self.projection = nn.Sequential(
            nn.LayerNorm(feature_size),
            nn.Linear(feature_size, hidden_size),
            nn.GELU(),
        )
        self.slot_embedding = nn.Parameter(
            torch.randn(num_slots, hidden_size) * 0.02
        )
        self.target_query = nn.Parameter(
            torch.randn(num_targets, hidden_size) * 0.02
        )
        self.output_weight = nn.Parameter(
            torch.randn(num_targets, hidden_size) * 0.02
        )
        self.output_bias = nn.Parameter(torch.zeros(num_targets))
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.num_slots = num_slots
        self.num_targets = num_targets

    def attention_weights(
        self, hidden: torch.Tensor, slot_present: torch.Tensor
    ) -> torch.Tensor:
        """Return target-by-slot weights, with absent slots exactly zero."""
        scores = torch.einsum(
            "bsh,th->bts", hidden, self.target_query
        ) / math.sqrt(self.hidden_size)
        mask = slot_present.to(dtype=torch.bool).unsqueeze(1)
        weights = torch.softmax(scores.masked_fill(~mask, -1e4), dim=-1)
        weights = weights * mask.to(weights.dtype)
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def forward(
        self, slot_features: torch.Tensor, slot_present: torch.Tensor
    ) -> torch.Tensor:
        if slot_features.shape[1] != self.num_slots:
            raise ValueError(
                f"Expected {self.num_slots} slots, received "
                f"{slot_features.shape[1]}"
            )
        hidden = self.projection(slot_features) + self.slot_embedding.unsqueeze(0)
        attention = self.attention_weights(hidden, slot_present)
        context = self.dropout(torch.einsum("bts,bsh->bth", attention, hidden))
        return (
            (context * self.output_weight.unsqueeze(0)).sum(dim=-1)
            + self.output_bias
        )


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
    """Encode three-slice 2.5D slots and attend to them by target."""

    def __init__(
        self,
        backbone: nn.Module,
        num_targets: int,
        unfreeze_last: int,
        dropout: float,
        num_slots: int,
        slices_per_slot: int,
        attention_hidden_size: int,
    ) -> None:
        super().__init__()
        if unfreeze_last <= 0:
            raise ValueError("At least one DINOv2 encoder block must be trainable")
        if slices_per_slot <= 0:
            raise ValueError("Slices per slot must be positive")
        if slices_per_slot != 3:
            raise ValueError("2.5D input requires exactly three slices per slot")
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
        self.head = TargetAttentionHead(
            feature_size=feature_size,
            num_slots=num_slots,
            num_targets=num_targets,
            hidden_size=attention_hidden_size,
            dropout=dropout,
        )
        self.unfreeze_last = unfreeze_last
        self.feature_size = feature_size
        self.num_slots = num_slots
        self.slices_per_slot = slices_per_slot
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
        expected_views = self.num_slots * self.slices_per_slot
        if num_views != expected_views:
            raise ValueError(
                f"Expected {expected_views} views, received {num_views}"
            )
        if present.shape != (batch_size, num_views):
            raise ValueError(
                f"Expected presence shape {(batch_size, num_views)}, "
                f"received {tuple(present.shape)}"
            )
        slice_present = present.reshape(
            batch_size, self.num_slots, self.slices_per_slot
        ).to(dtype=torch.bool)
        slot_present = slice_present.any(dim=2)
        slot_images = images.reshape(
            batch_size, self.num_slots, self.slices_per_slot, height, width
        )
        pixels = slot_images[slot_present]
        pixels = (pixels - self.image_mean) / self.image_std
        channel_present = slice_present[slot_present].to(pixels.dtype)
        pixels = pixels * channel_present.view(-1, 3, 1, 1)
        if pixels.shape[0]:
            tokens = self.backbone(pixel_values=pixels).last_hidden_state
            if tokens.shape[1] < 2:
                raise ValueError("DINOv2 output did not contain patch tokens")
            encoded = torch.cat(
                (tokens[:, 0], tokens[:, 1:].mean(dim=1)), dim=1
            )
            features = encoded.new_zeros(
                (batch_size * self.num_slots, self.feature_size)
            )
            features[slot_present.reshape(-1)] = encoded
        else:
            features = images.new_zeros(
                (batch_size * self.num_slots, self.feature_size)
            )
        slot_features = features.reshape(
            batch_size, self.num_slots, self.feature_size
        )
        return self.head(slot_features, slot_present)


def build_dinov2_model(
    input_root: str | Path,
    variant: str,
    num_targets: int,
    unfreeze_last: int,
    dropout: float,
    num_slots: int,
    slices_per_slot: int,
    attention_hidden_size: int,
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
        num_slots=num_slots,
        slices_per_slot=slices_per_slot,
        attention_hidden_size=attention_hidden_size,
    )
    return model, source
