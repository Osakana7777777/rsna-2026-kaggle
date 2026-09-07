from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from rsna2026.model import (  # noqa: E402
    HierarchicalTargetAttentionHead,
    KneeAbnormalityModel,
    TargetAttentionHead,
)


class DummyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("Config", (), {"hidden_size": 2})()
        self.encoder = type("Encoder", (nn.Module,), {})()
        self.encoder.layer = nn.ModuleList([nn.Linear(1, 1)])
        self.layernorm = nn.LayerNorm(2)
        self.received = None

    def forward(self, pixel_values):
        self.received = pixel_values.detach().clone()
        tokens = pixel_values.new_zeros((len(pixel_values), 2, 2))
        return type("Output", (), {"last_hidden_state": tokens})()


def test_target_attention_masks_missing_slots_and_renormalizes() -> None:
    head = TargetAttentionHead(
        feature_size=8,
        num_slots=6,
        num_targets=12,
        hidden_size=4,
        dropout=0.0,
    )
    hidden = torch.randn(2, 6, 4)
    present = torch.tensor(
        [[True, False, True, False, False, True], [False] * 6]
    )

    weights = head.attention_weights(hidden, present)

    assert torch.all(weights[0, :, ~present[0]] == 0)
    assert torch.allclose(weights[0].sum(dim=-1), torch.ones(12))
    assert torch.all(weights[1] == 0)


def test_model_stacks_three_ordered_slices_as_channels() -> None:
    backbone = DummyBackbone()
    model = KneeAbnormalityModel(
        backbone=backbone,
        num_targets=12,
        unfreeze_last=1,
        dropout=0.0,
        num_slots=6,
        max_series_per_slot=2,
        window_count=10,
        slices_per_slot=3,
        attention_hidden_size=4,
    )
    images = torch.zeros((1, 6, 2, 10, 3, 14, 14))
    images[0, 0, 0, 0, 0] = 0.1
    images[0, 0, 0, 0, 1] = 0.2
    images[0, 0, 0, 0, 2] = 0.3
    present = torch.zeros((1, 6, 2, 10), dtype=torch.bool)
    present[0, 0, 0, 0] = True
    positions = torch.zeros((1, 6, 2, 10, 3))

    logits = model(images, present, positions)

    expected = (torch.tensor([0.1, 0.2, 0.3]) - model.image_mean.flatten()) / (
        model.image_std.flatten()
    )
    assert logits.shape == (1, 12)
    assert backbone.received.shape == (1, 3, 14, 14)
    assert torch.allclose(backbone.received[0, :, 0, 0], expected)


def test_model_requires_three_slices_for_2_5d_input() -> None:
    with pytest.raises(ValueError, match="exactly three"):
        KneeAbnormalityModel(
            backbone=DummyBackbone(),
            num_targets=12,
            unfreeze_last=1,
            dropout=0.0,
            num_slots=6,
            max_series_per_slot=2,
            window_count=10,
            slices_per_slot=1,
            attention_hidden_size=4,
        )


def test_hierarchical_attention_masks_windows_and_slots() -> None:
    head = HierarchicalTargetAttentionHead(8, 4, 2, 3, 3, 0.0)
    features = torch.randn(1, 2, 2, 4, 8)
    present = torch.tensor(
        [[
            [[True, False, True, False], [True, False, False, False]],
            [[False] * 4, [False] * 4],
        ]]
    )
    positions = torch.zeros(1, 2, 2, 4, 3)

    logits, window_attention, series_attention, slot_attention = head(
        features, present, positions, return_attention=True
    )

    assert logits.shape == (1, 3)
    assert window_attention.shape == (1, 3, 2, 2, 4)
    assert series_attention.shape == (1, 3, 2, 2)
    assert slot_attention.shape == (1, 3, 2)
    assert torch.all(window_attention[:, :, 0, 0, [1, 3]] == 0)
    assert torch.all(window_attention[:, :, 0, 1, 1:] == 0)
    assert torch.all(window_attention[:, :, 1] == 0)
    assert torch.all(series_attention[:, :, 1] == 0)
    assert torch.all(slot_attention[:, :, 1] == 0)
    assert torch.allclose(
        window_attention[:, :, 0].sum(-1), torch.ones(1, 3, 2)
    )
    assert torch.allclose(series_attention[:, :, 0].sum(-1), torch.ones(1, 3))
    assert torch.allclose(slot_attention.sum(-1), torch.ones(1, 3))
    assert torch.isfinite(logits).all()


def test_series_attention_is_permutation_invariant() -> None:
    head = HierarchicalTargetAttentionHead(8, 4, 1, 3, 3, 0.0)
    head.eval()
    features = torch.randn(1, 1, 2, 3, 8)
    present = torch.ones(1, 1, 2, 3, dtype=torch.bool)
    positions = torch.randn(1, 1, 2, 3, 3)

    first = head(features, present, positions)
    second = head(
        features.flip(2), present.flip(2), positions.flip(2)
    )

    assert torch.allclose(first, second, atol=1e-6)
