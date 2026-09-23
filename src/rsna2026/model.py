"""Study-level DINOv2 model for knee MRI slices."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch
from torch import nn


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RADIMAGENET_MEAN = (0.5, 0.5, 0.5)
RADIMAGENET_STD = (0.5, 0.5, 0.5)


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


class TransformerTargetAttentionHead(nn.Module):
    """Contextualize all windows before target-specific global pooling."""

    def __init__(
        self,
        feature_size: int,
        hidden_size: int,
        num_slots: int,
        num_targets: int,
        position_size: int,
        dropout: float,
        num_layers: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        if num_layers <= 0 or num_heads <= 0:
            raise ValueError("Transformer layer and head counts must be positive")
        if hidden_size % num_heads:
            raise ValueError("Attention hidden size must be divisible by head count")
        self.window_projection = nn.Sequential(
            nn.LayerNorm(feature_size),
            nn.Linear(feature_size, hidden_size),
            nn.GELU(),
        )
        self.position_projection = nn.Sequential(
            nn.Linear(position_size, hidden_size), nn.GELU()
        )
        self.slot_embedding = nn.Parameter(
            torch.randn(num_slots, hidden_size) * 0.02
        )
        self.transformer_norms = nn.ModuleList(
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        )
        self.transformer_layers = nn.ModuleList(
            nn.MultiheadAttention(
                hidden_size,
                num_heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(hidden_size)
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
        batch_size, num_slots, max_series, windows, _ = window_features.shape
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
        hidden = hidden + self.slot_embedding.view(
            1, self.num_slots, 1, 1, self.hidden_size
        )
        token_count = num_slots * max_series * windows
        hidden = hidden.reshape(batch_size, token_count, self.hidden_size)
        token_present = window_present.reshape(batch_size, token_count).to(torch.bool)
        hidden = hidden.masked_fill(~token_present.unsqueeze(-1), 0.0)

        # MultiheadAttention returns NaNs when every key in a row is masked.
        # Empty studies bypass it and are handled by the global masked pooling.
        nonempty = token_present.any(dim=-1)
        if nonempty.any():
            contextualized = hidden[nonempty]
            key_padding_mask = ~token_present[nonempty]
            for norm, attention in zip(
                self.transformer_norms, self.transformer_layers, strict=True
            ):
                normalized = norm(contextualized)
                attended, _ = attention(
                    normalized,
                    normalized,
                    normalized,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )
                contextualized = contextualized + attended
            hidden = hidden.index_copy(0, nonempty.nonzero().flatten(), contextualized)
        hidden = self.final_norm(hidden)

        scores = torch.einsum(
            "bnh,th->btn", hidden, self.target_query
        ) / math.sqrt(self.hidden_size)
        global_attention = self._masked_softmax(
            scores, token_present.unsqueeze(1)
        )
        context = self.dropout(
            torch.einsum("btn,bnh->bth", global_attention, hidden)
        )
        logits = (
            context * self.output_weight.unsqueeze(0)
        ).sum(dim=-1) + self.output_bias
        if return_attention:
            window_attention = global_attention.reshape(
                batch_size,
                self.num_targets,
                num_slots,
                max_series,
                windows,
            )
            series_attention = window_attention.sum(dim=-1)
            slot_attention = series_attention.sum(dim=-1)
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


def find_coatnet_source(
    input_root: str | Path,
    architecture: str,
    weights_file: str = "model.safetensors",
) -> Path:
    """Find one mounted timm CoAtNet with the requested architecture and weights."""
    root = Path(input_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Model input root does not exist: {root}")
    candidates = []
    for directory, names, files in os.walk(root):
        names[:] = [
            name for name in names if name not in {"train_series", "test_series"}
        ]
        if "config.json" not in files or weights_file not in files:
            continue
        path = Path(directory)
        try:
            config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        configured_architecture = str(config.get("architecture", ""))
        pretrained_config = config.get("pretrained_cfg", {})
        tag = (
            str(pretrained_config.get("tag", ""))
            if isinstance(pretrained_config, dict)
            else ""
        )
        qualified_architecture = (
            f"{configured_architecture}.{tag}" if tag else configured_architecture
        )
        if architecture in {configured_architecture, qualified_architecture}:
            candidates.append(path)
    candidates.sort()
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one mounted CoAtNet {architecture} containing "
            f"{weights_file}, found {candidates}"
        )
    return candidates[0]


def find_radimagenet_source(
    input_root: str | Path, weights_file: str = "ResNet50.pt"
) -> Path:
    """Find exactly one mounted official RadImageNet weight file."""
    root = Path(input_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Model input root does not exist: {root}")
    candidates = []
    for directory, names, files in os.walk(root):
        names[:] = [
            name for name in names if name not in {"train_series", "test_series"}
        ]
        if weights_file in files:
            candidates.append(Path(directory))
    candidates.sort()
    if len(candidates) != 1:
        raise FileNotFoundError(
            "Expected exactly one mounted RadImageNet weight file "
            f"{weights_file}, found {candidates}"
        )
    return candidates[0]


class RadImageNetResNet50(nn.Sequential):
    """Official RadImageNet ResNet-50 feature extractor without its classifier."""

    num_features = 2048

    def __init__(self) -> None:
        from torchvision.models import resnet50

        model = resnet50(weights=None)
        super().__init__(*list(model.children())[:9])

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return super().forward(pixels).flatten(1)


def _strip_radimagenet_backbone_prefix(
    checkpoint: dict[object, object],
) -> dict[str, object]:
    """Validate the official checkpoint namespace and strip its wrapper prefix."""
    if not checkpoint:
        raise ValueError("RadImageNet checkpoint state dictionary is empty")

    prefix = "backbone."
    normalized: dict[str, object] = {}
    for key, value in checkpoint.items():
        if not isinstance(key, str):
            raise TypeError("RadImageNet checkpoint keys must be strings")
        if not key.startswith(prefix):
            raise ValueError(
                "Official RadImageNet checkpoint keys must all start with "
                f"{prefix!r}; found {key!r}"
            )
        stripped_key = key[len(prefix) :]
        if not stripped_key:
            raise ValueError(
                "RadImageNet checkpoint contains an empty key after stripping "
                f"{prefix!r}"
            )
        if stripped_key in normalized:
            raise ValueError(
                "RadImageNet checkpoint key collision after stripping "
                f"{prefix!r}: {stripped_key!r}"
            )
        normalized[stripped_key] = value
    return normalized


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
        backbone_family: str | None = None,
        image_mean: tuple[float, float, float] = IMAGENET_MEAN,
        image_std: tuple[float, float, float] = IMAGENET_STD,
        aggregation: str = "hierarchical-target-series-attention",
        transformer_layers: int = 2,
        transformer_heads: int = 8,
    ) -> None:
        super().__init__()
        if unfreeze_last < 0:
            raise ValueError("Backbone blocks to unfreeze cannot be negative")
        if slices_per_slot <= 0:
            raise ValueError("Slices per slot must be positive")
        if slices_per_slot != 3:
            raise ValueError("2.5D input requires exactly three slices per slot")
        if window_count <= 0:
            raise ValueError("Window count must be positive")
        if max_series_per_slot <= 0:
            raise ValueError("Maximum series per slot must be positive")
        self.backbone = backbone
        self.backbone_family = backbone_family or self._detect_backbone_family()
        if unfreeze_last == 0 and self.backbone_family != "radimagenet-resnet50":
            raise ValueError("At least one backbone block must be trainable")
        if len(image_mean) != 3 or len(image_std) != 3:
            raise ValueError("Image normalization must have three channels")
        layers = self._backbone_blocks()
        if unfreeze_last > len(layers):
            raise ValueError(
                f"Cannot unfreeze {unfreeze_last} of {len(layers)} encoder blocks"
            )
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if unfreeze_last:
            for layer in layers[-unfreeze_last:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
        final_norm = self._final_norm()
        if final_norm is not None:
            for parameter in final_norm.parameters():
                parameter.requires_grad = True
        self._frozen_batch_norms = [
            module
            for module in self.backbone.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
            and not any(parameter.requires_grad for parameter in module.parameters())
        ]

        if self.backbone_family == "dinov2":
            feature_size = int(self.backbone.config.hidden_size) * 2
        else:
            feature_size = int(getattr(self.backbone, "num_features"))
        head_arguments = {
            "feature_size": feature_size,
            "hidden_size": attention_hidden_size,
            "num_slots": num_slots,
            "num_targets": num_targets,
            "position_size": 3,
            "dropout": dropout,
        }
        if aggregation == "hierarchical-target-series-attention":
            self.head = HierarchicalTargetAttentionHead(**head_arguments)
        elif aggregation == "transformer-target-attention":
            self.head = TransformerTargetAttentionHead(
                **head_arguments,
                num_layers=transformer_layers,
                num_heads=transformer_heads,
            )
        else:
            raise ValueError(f"Unsupported aggregation: {aggregation}")
        self.unfreeze_last = unfreeze_last
        self.feature_size = feature_size
        self.num_slots = num_slots
        self.max_series_per_slot = max_series_per_slot
        self.window_count = window_count
        self.num_targets = num_targets
        self.slices_per_slot = slices_per_slot
        self.register_buffer(
            "image_mean", torch.tensor(image_mean).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor(image_std).view(1, 3, 1, 1)
        )

    def _detect_backbone_family(self) -> str:
        encoder = getattr(self.backbone, "encoder", None)
        if isinstance(getattr(encoder, "layer", None), nn.ModuleList):
            return "dinov2"
        if hasattr(self.backbone, "stages") and hasattr(
            self.backbone, "num_features"
        ):
            return "coatnet"
        raise TypeError("Cannot determine the backbone family")

    def _backbone_blocks(self) -> list[nn.Module]:
        if self.backbone_family == "coatnet":
            stages = getattr(self.backbone, "stages", None)
            if not isinstance(stages, nn.Sequential):
                raise TypeError("CoAtNet backbone does not expose sequential stages")
            blocks = []
            for stage in stages:
                stage_blocks = getattr(stage, "blocks", None)
                if not isinstance(stage_blocks, nn.Sequential):
                    raise TypeError("CoAtNet stage does not expose sequential blocks")
                blocks.extend(stage_blocks)
            return blocks
        if self.backbone_family == "radimagenet-resnet50":
            stages = list(self.backbone.children())[4:8]
            if len(stages) != 4 or not all(
                isinstance(stage, nn.Sequential) for stage in stages
            ):
                raise TypeError(
                    "RadImageNet ResNet-50 does not expose layer1 to layer4"
                )
            return [block for stage in stages for block in stage]
        if self.backbone_family != "dinov2":
            raise ValueError(f"Unsupported backbone family: {self.backbone_family}")
        encoder = getattr(self.backbone, "encoder", None)
        layers = getattr(encoder, "layer", None)
        if not isinstance(layers, nn.ModuleList):
            raise TypeError("DINOv2 backbone does not expose encoder.layer")
        return list(layers)

    def _final_norm(self) -> nn.Module | None:
        if self.backbone_family == "radimagenet-resnet50":
            return None
        name = "layernorm" if self.backbone_family == "dinov2" else "norm"
        final_norm = getattr(self.backbone, name, None)
        if not isinstance(final_norm, nn.Module):
            raise TypeError(
                f"{self.backbone_family} backbone does not expose its final norm"
            )
        return final_norm

    def _encode(self, pixels: torch.Tensor) -> torch.Tensor:
        if self.backbone_family in {"coatnet", "radimagenet-resnet50"}:
            encoded = self.backbone(pixels)
            if encoded.ndim != 2 or encoded.shape[1] != self.feature_size:
                raise ValueError(
                    f"{self.backbone_family} backbone returned an invalid "
                    "embedding shape"
                )
            return encoded
        tokens = self.backbone(pixel_values=pixels).last_hidden_state
        if tokens.shape[1] < 2:
            raise ValueError("DINOv2 output did not contain patch tokens")
        return torch.cat((tokens[:, 0], tokens[:, 1:].mean(dim=1)), dim=1)

    def train(self, mode: bool = True) -> KneeAbnormalityModel:
        """Keep running statistics fixed in frozen backbone batch norms."""
        super().train(mode)
        if mode:
            for module in self._frozen_batch_norms:
                module.eval()
        return self

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
            encoded = self._encode(pixels)
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
            dummy_features = self._encode(
                images.new_zeros((1, 3, height, width))
            )
            features = images.new_zeros(
                (
                    batch_size
                    * self.num_slots
                    * self.max_series_per_slot
                    * self.window_count,
                    self.feature_size,
                )
            ) + dummy_features.sum() * 0.0
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
    aggregation: str = "hierarchical-target-series-attention",
    transformer_layers: int = 2,
    transformer_heads: int = 8,
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
        aggregation=aggregation,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
    )
    return model, source


def build_coatnet_model(
    input_root: str | Path,
    architecture: str,
    weights_file: str,
    num_targets: int,
    unfreeze_last: int,
    dropout: float,
    num_slots: int,
    max_series_per_slot: int,
    window_count: int,
    slices_per_slot: int,
    attention_hidden_size: int,
    gradient_checkpointing: bool,
    aggregation: str = "hierarchical-target-series-attention",
    transformer_layers: int = 2,
    transformer_heads: int = 8,
) -> tuple[KneeAbnormalityModel, Path]:
    """Load a mounted timm CoAtNet checkpoint without network access."""
    import timm
    from timm.models import load_checkpoint

    source = find_coatnet_source(input_root, architecture, weights_file)
    backbone = timm.create_model(
        architecture,
        pretrained=False,
        num_classes=1000,
        in_chans=3,
    )
    load_checkpoint(backbone, str(source / weights_file), strict=True)
    backbone.reset_classifier(0, global_pool="avg")
    if int(backbone.num_features) != 768:
        raise ValueError(
            f"Expected a 768-dimensional CoAtNet embedding, got "
            f"{backbone.num_features}"
        )
    if gradient_checkpointing:
        checkpointing = getattr(backbone, "set_grad_checkpointing", None)
        if not callable(checkpointing):
            raise TypeError("CoAtNet backbone does not support gradient checkpointing")
        checkpointing(True)
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
        backbone_family="coatnet",
        aggregation=aggregation,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
    )
    return model, source


def build_radimagenet_resnet50_model(
    input_root: str | Path,
    weights_file: str,
    num_targets: int,
    unfreeze_last: int,
    dropout: float,
    num_slots: int,
    max_series_per_slot: int,
    window_count: int,
    slices_per_slot: int,
    attention_hidden_size: int,
    gradient_checkpointing: bool,
    aggregation: str = "hierarchical-target-series-attention",
    transformer_layers: int = 2,
    transformer_heads: int = 8,
) -> tuple[KneeAbnormalityModel, Path]:
    """Load official RadImageNet ResNet-50 weights without network access."""
    if gradient_checkpointing:
        raise ValueError(
            "RadImageNet ResNet-50 does not support gradient checkpointing"
        )
    source = find_radimagenet_source(input_root, weights_file)
    checkpoint = torch.load(
        source / weights_file, map_location="cpu", weights_only=True
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("RadImageNet checkpoint must be a state dictionary")
    try:
        checkpoint = _strip_radimagenet_backbone_prefix(checkpoint)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "Invalid official RadImageNet ResNet-50 checkpoint namespace: "
            f"{source / weights_file}"
        ) from error
    backbone = RadImageNetResNet50()
    try:
        backbone.load_state_dict(checkpoint, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "Failed to strictly load the official RadImageNet ResNet-50 "
            f"checkpoint: {source / weights_file}"
        ) from error
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
        backbone_family="radimagenet-resnet50",
        image_mean=RADIMAGENET_MEAN,
        image_std=RADIMAGENET_STD,
        aggregation=aggregation,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
    )
    return model, source
