from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from rsna2026.model import TargetAttentionHead  # noqa: E402


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
