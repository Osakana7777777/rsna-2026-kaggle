from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from rsna2026.model import KneeAbnormalityModel, TargetAttentionHead  # noqa: E402


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
        slices_per_slot=3,
        attention_hidden_size=4,
    )
    images = torch.zeros((1, 18, 1, 14, 14))
    images[0, 0] = 0.1
    images[0, 1] = 0.2
    images[0, 2] = 0.3
    present = torch.zeros((1, 18), dtype=torch.bool)
    present[0, :3] = True

    logits = model(images, present)

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
            slices_per_slot=1,
            attention_hidden_size=4,
        )
