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


class HierarchicalTargetAttentionHead(nn.Module):
    """Attend to windows, series, and acquisition slots for every target."""

    def __init__(
        self,
        feature_size: int,
        hidden_size: int,
        num_slots: int,
        num_targets: int,
        position_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.window_projection = nn.Sequential(
            nn.LayerNorm(feature_size),
            nn.Linear(feature_size, hidden_size),
            nn.GELU(),
        )
        self.position_projection = nn.Sequential(
            nn.Linear(position_size, hidden_size), nn.GELU()
        )
        self.window_query = nn.Parameter(
            torch.randn(num_targets, hidden_size) * 0.02
        )
        self.series_projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.series_query = nn.Parameter(
            torch.randn(num_targets, hidden_size) * 0.02
        )
        self.slot_projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.slot_embedding = nn.Parameter(
            torch.randn(num_slots, hidden_size) * 0.02
        )
        self.slot_query = nn.Parameter(
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
        self.position_size = position_size

    @staticmethod
    def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(scores.masked_fill(~mask, -1e4), dim=-1)
        weights = weights * mask.to(weights.dtype)
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def forward(
        self,
        window_features: torch.Tensor,
        window_present: torch.Tensor,
        position_features: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        _, num_slots, _, _, _ = window_features.shape
        if num_slots != self.num_slots:
            raise ValueError(f"Expected {self.num_slots} slots, received {num_slots}")
        if window_present.shape != window_features.shape[:4]:
            raise ValueError("Window mask shape does not match window features")
        if position_features.shape != (
            *window_features.shape[:4],
            self.position_size,
        ):
            raise ValueError("Position feature shape does not match windows")
        hidden = self.window_projection(window_features)
        hidden = hidden + self.position_projection(position_features)
        window_scores = torch.einsum(
            "bsrwh,th->btsrw", hidden, self.window_query
        ) / math.sqrt(self.hidden_size)
        window_mask = window_present.to(torch.bool).unsqueeze(1)
        window_attention = self._masked_softmax(window_scores, window_mask)
        series_context = torch.einsum(
            "btsrw,bsrwh->btsrh", window_attention, hidden
        )
        series_hidden = self.series_projection(series_context)
        series_present = window_present.any(dim=-1).unsqueeze(1)
        series_scores = (
            series_hidden
            * self.series_query.view(1, self.num_targets, 1, 1, -1)
        ).sum(dim=-1) / math.sqrt(self.hidden_size)
        series_attention = self._masked_softmax(series_scores, series_present)
        slot_context = torch.einsum(
            "btsr,btsrh->btsh", series_attention, series_hidden
        )
        slot_hidden = self.slot_projection(slot_context)
        slot_hidden = slot_hidden + self.slot_embedding.view(
            1, 1, self.num_slots, self.hidden_size
        )
        slot_present = series_present.any(dim=-1)
        slot_scores = (
            slot_hidden * self.slot_query.view(1, self.num_targets, 1, -1)
        ).sum(dim=-1) / math.sqrt(self.hidden_size)
        slot_attention = self._masked_softmax(slot_scores, slot_present)
        study_context = self.dropout(
            torch.einsum("bts,btsh->bth", slot_attention, slot_hidden)
        )
        logits = (
            study_context * self.output_weight.unsqueeze(0)
        ).sum(dim=-1) + self.output_bias
        if return_attention:
            return logits, window_attention, series_attention, slot_attention
        return logits


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
        max_series_per_slot: int,
        window_count: int,
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
        if window_count <= 0:
            raise ValueError("Window count must be positive")
        if max_series_per_slot <= 0:
            raise ValueError("Maximum series per slot must be positive")
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
        self.head = HierarchicalTargetAttentionHead(
            feature_size=feature_size,
            hidden_size=attention_hidden_size,
            num_slots=num_slots,
            num_targets=num_targets,
            position_size=3,
            dropout=dropout,
        )
        self.unfreeze_last = unfreeze_last
        self.feature_size = feature_size
        self.num_slots = num_slots
        self.max_series_per_slot = max_series_per_slot
        self.window_count = window_count
        self.num_targets = num_targets
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

    def forward(
        self,
        images: torch.Tensor,
        present: torch.Tensor,
        positions: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        if images.ndim != 7:
            raise ValueError(
                "Expected [batch, slots, series, windows, 3, height, width]"
            )
        (
            batch_size,
            num_slots,
            max_series,
            windows,
            channels,
            height,
            width,
        ) = images.shape
        expected = (
            self.num_slots,
            self.max_series_per_slot,
            self.window_count,
            self.slices_per_slot,
        )
        if (num_slots, max_series, windows, channels) != expected:
            raise ValueError(f"Expected slot/series/window/channel shape {expected}")
        if present.shape != (
            batch_size,
            self.num_slots,
            self.max_series_per_slot,
            self.window_count,
        ):
            raise ValueError("Window presence shape does not match images")
        if positions.shape != (*present.shape, 3):
            raise ValueError("Window position shape does not match images")
        pixels = images[present.to(torch.bool)]
        pixels = (pixels - self.image_mean) / self.image_std
        if pixels.shape[0]:
            tokens = self.backbone(pixel_values=pixels).last_hidden_state
            if tokens.shape[1] < 2:
                raise ValueError("DINOv2 output did not contain patch tokens")
            encoded = torch.cat(
                (tokens[:, 0], tokens[:, 1:].mean(dim=1)), dim=1
            )
            features = encoded.new_zeros(
                (
                    batch_size
                    * self.num_slots
                    * self.max_series_per_slot
                    * self.window_count,
                    self.feature_size,
                )
            )
            features[present.reshape(-1).to(torch.bool)] = encoded
        else:
            dummy_tokens = self.backbone(
                pixel_values=images.new_zeros((1, 3, height, width))
            ).last_hidden_state
            features = images.new_zeros(
                (
                    batch_size
                    * self.num_slots
                    * self.max_series_per_slot
                    * self.window_count,
                    self.feature_size,
                )
            ) + dummy_tokens.sum() * 0.0
        window_features = features.reshape(
            batch_size,
            self.num_slots,
            self.max_series_per_slot,
            self.window_count,
            self.feature_size,
        )
        return self.head(
            window_features, present, positions, return_attention=return_attention
        )


def build_dinov2_model(
    input_root: str | Path,
    variant: str,
    num_targets: int,
    unfreeze_last: int,
    dropout: float,
    num_slots: int,
    max_series_per_slot: int,
    window_count: int,
    slices_per_slot: int,
    attention_hidden_size: int,
    gradient_checkpointing: bool,
) -> tuple[KneeAbnormalityModel, Path]:
    """Load a mounted DINOv2 model without any network access."""
    from transformers import AutoModel
    from transformers.utils import logging as transformers_logging

    source = find_dinov2_source(input_root, variant)
    transformers_logging.disable_progress_bar()
    backbone = AutoModel.from_pretrained(str(source), local_files_only=True)
    checkpointing = getattr(backbone, "gradient_checkpointing_enable", None)
    if gradient_checkpointing and callable(checkpointing):
        try:
            checkpointing({"use_reentrant": False})
        except TypeError:
            checkpointing()
    model = KneeAbnormalityModel(
        backbone=backbone,
        num_targets=num_targets,
        unfreeze_last=unfreeze_last,
        dropout=dropout,
        num_slots=num_slots,
        max_series_per_slot=max_series_per_slot,
        window_count=window_count,
        slices_per_slot=slices_per_slot,
        attention_hidden_size=attention_hidden_size,
    )
    return model, source
