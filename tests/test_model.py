from __future__ import annotations

import json
import sys
import types

import pytest


torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from rsna2026.model import (  # noqa: E402
    HierarchicalTargetAttentionHead,
    KneeAbnormalityModel,
    RADIMAGENET_MEAN,
    RADIMAGENET_STD,
    RadImageNetResNet50,
    TargetAttentionHead,
    TransformerTargetAttentionHead,
    build_coatnet_model,
    build_radimagenet_resnet50_model,
    find_coatnet_source,
    find_radimagenet_source,
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


class DummyCoAtNet(nn.Module):
    def __init__(self, feature_size: int = 4) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, feature_size, 1), nn.BatchNorm2d(feature_size)
        )
        self.stages = nn.Sequential(
            *(self._stage(count, feature_size) for count in (2, 3, 7, 2))
        )
        self.norm = nn.BatchNorm2d(feature_size)
        self.num_features = feature_size
        self.checkpointing = False

    @staticmethod
    def _stage(count: int, feature_size: int) -> nn.Module:
        stage = nn.Module()
        stage.blocks = nn.Sequential(
            *(
                nn.Sequential(
                    nn.BatchNorm2d(feature_size),
                    nn.Conv2d(feature_size, feature_size, 1),
                )
                for _ in range(count)
            )
        )
        return stage

    def reset_classifier(self, num_classes: int, global_pool: str) -> None:
        assert num_classes == 0
        assert global_pool == "avg"

    def set_grad_checkpointing(self, enabled: bool) -> None:
        self.checkpointing = enabled

    def forward(self, pixels):
        values = self.stem(pixels)
        for stage in self.stages:
            values = stage.blocks(values)
        values = self.norm(values)
        return values.mean(dim=(2, 3))


class DummyRadImageNetResNet50(nn.Sequential):
    num_features = 4

    def __init__(self) -> None:
        stages = [self._stage(count) for count in (3, 4, 6, 3)]
        super().__init__(
            nn.Conv2d(3, 4, 1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Identity(),
            *stages,
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.loaded_strict = None

    @staticmethod
    def _stage(count: int) -> nn.Sequential:
        return nn.Sequential(
            *(
                nn.Sequential(nn.BatchNorm2d(4), nn.Conv2d(4, 4, 1))
                for _ in range(count)
            )
        )

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.loaded_strict = strict
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(self, pixels):
        return super().forward(pixels).flatten(1)


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


@pytest.mark.parametrize(
    ("backbone", "backbone_family"),
    [
        (DummyBackbone, None),
        (DummyCoAtNet, "coatnet"),
    ],
)
def test_non_radimagenet_backbones_require_a_trainable_block(
    backbone, backbone_family
) -> None:
    with pytest.raises(ValueError, match="At least one backbone block"):
        KneeAbnormalityModel(
            backbone=backbone(),
            backbone_family=backbone_family,
            num_targets=2,
            unfreeze_last=0,
            dropout=0.0,
            num_slots=1,
            max_series_per_slot=1,
            window_count=1,
            slices_per_slot=3,
            attention_hidden_size=4,
        )


def test_coatnet_unfreezes_last_six_blocks_and_keeps_frozen_bn_fixed() -> None:
    backbone = DummyCoAtNet()
    model = KneeAbnormalityModel(
        backbone=backbone,
        backbone_family="coatnet",
        num_targets=12,
        unfreeze_last=6,
        dropout=0.0,
        num_slots=1,
        max_series_per_slot=1,
        window_count=1,
        slices_per_slot=3,
        attention_hidden_size=4,
    )
    blocks = [block for stage in backbone.stages for block in stage.blocks]

    assert all(not any(p.requires_grad for p in block.parameters()) for block in blocks[:8])
    assert all(any(p.requires_grad for p in block.parameters()) for block in blocks[8:])
    assert all(parameter.requires_grad for parameter in backbone.norm.parameters())

    model.train()
    assert backbone.stem[1].training is False
    assert blocks[0][0].training is False
    assert blocks[-1][0].training is True
    assert backbone.norm.training is True


def test_coatnet_connects_embeddings_to_hierarchical_attention() -> None:
    backbone = DummyCoAtNet()
    model = KneeAbnormalityModel(
        backbone=backbone,
        backbone_family="coatnet",
        num_targets=12,
        unfreeze_last=6,
        dropout=0.0,
        num_slots=2,
        max_series_per_slot=1,
        window_count=2,
        slices_per_slot=3,
        attention_hidden_size=4,
    )
    images = torch.rand(1, 2, 1, 2, 3, 8, 8)
    present = torch.tensor([[[[True, False]], [[True, True]]]])
    positions = torch.zeros(1, 2, 1, 2, 3)

    logits, window_attention, series_attention, slot_attention = model(
        images, present, positions, return_attention=True
    )

    assert logits.shape == (1, 12)
    assert window_attention.shape == (1, 12, 2, 1, 2)
    assert series_attention.shape == (1, 12, 2, 1)
    assert slot_attention.shape == (1, 12, 2)


def test_coatnet_empty_windows_preserve_backbone_gradient_path() -> None:
    backbone = DummyCoAtNet()
    model = KneeAbnormalityModel(
        backbone=backbone,
        backbone_family="coatnet",
        num_targets=2,
        unfreeze_last=6,
        dropout=0.0,
        num_slots=1,
        max_series_per_slot=1,
        window_count=1,
        slices_per_slot=3,
        attention_hidden_size=4,
    )
    images = torch.zeros(1, 1, 1, 1, 3, 8, 8)
    present = torch.zeros(1, 1, 1, 1, dtype=torch.bool)
    positions = torch.zeros(1, 1, 1, 1, 3)

    model(images, present, positions).sum().backward()

    trainable = [
        parameter
        for parameter in backbone.parameters()
        if parameter.requires_grad
    ]
    assert trainable
    assert all(parameter.grad is not None for parameter in trainable)


def test_find_coatnet_source_requires_matching_config_and_weights(tmp_path) -> None:
    source = tmp_path / "mounted-model"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "architecture": "coatnet_0_rw_224",
                "pretrained_cfg": {"tag": "sw_in1k"},
            }
        ),
        encoding="utf-8",
    )
    (source / "model.safetensors").write_bytes(b"weights")

    assert find_coatnet_source(
        tmp_path, "coatnet_0_rw_224.sw_in1k"
    ) == source

    duplicate = tmp_path / "duplicate"
    duplicate.mkdir()
    (duplicate / "config.json").write_text(
        (source / "config.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (duplicate / "model.safetensors").write_bytes(b"weights")
    with pytest.raises(FileNotFoundError, match="exactly one"):
        find_coatnet_source(tmp_path, "coatnet_0_rw_224.sw_in1k")


def test_build_coatnet_uses_strict_offline_checkpoint_loading(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "mounted-model"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "architecture": "coatnet_0_rw_224",
                "pretrained_cfg": {"tag": "sw_in1k"},
            }
        ),
        encoding="utf-8",
    )
    (source / "model.safetensors").write_bytes(b"weights")
    backbone = DummyCoAtNet(feature_size=768)
    calls = {}

    timm = types.ModuleType("timm")
    timm.create_model = lambda *args, **kwargs: backbone
    timm_models = types.ModuleType("timm.models")

    def fake_load_checkpoint(model, path, strict):
        calls.update(model=model, path=path, strict=strict)

    timm_models.load_checkpoint = fake_load_checkpoint
    monkeypatch.setitem(sys.modules, "timm", timm)
    monkeypatch.setitem(sys.modules, "timm.models", timm_models)

    model, found_source = build_coatnet_model(
        input_root=tmp_path,
        architecture="coatnet_0_rw_224.sw_in1k",
        weights_file="model.safetensors",
        num_targets=12,
        unfreeze_last=6,
        dropout=0.0,
        num_slots=1,
        max_series_per_slot=1,
        window_count=1,
        slices_per_slot=3,
        attention_hidden_size=4,
        gradient_checkpointing=True,
    )

    assert found_source == source
    assert model.backbone is backbone
    assert calls == {
        "model": backbone,
        "path": str(source / "model.safetensors"),
        "strict": True,
    }
    assert backbone.checkpointing is True


def test_find_radimagenet_source_requires_exactly_one_weight_file(
    tmp_path,
) -> None:
    with pytest.raises(FileNotFoundError, match=r"found \[\]"):
        find_radimagenet_source(tmp_path)

    source = tmp_path / "radimagenet"
    source.mkdir()
    (source / "ResNet50.pt").write_bytes(b"weights")
    assert find_radimagenet_source(tmp_path) == source

    duplicate = tmp_path / "duplicate"
    duplicate.mkdir()
    (duplicate / "ResNet50.pt").write_bytes(b"weights")
    with pytest.raises(FileNotFoundError, match="exactly one"):
        find_radimagenet_source(tmp_path)


def test_radimagenet_feature_extractor_returns_2048_global_average_features(
    monkeypatch,
) -> None:
    fake_resnet = nn.Sequential(
        nn.Conv2d(3, 2048, 1),
        nn.Identity(),
        nn.Identity(),
        nn.Identity(),
        nn.Identity(),
        nn.Identity(),
        nn.Identity(),
        nn.Identity(),
        nn.AdaptiveAvgPool2d((1, 1)),
    )
    torchvision = types.ModuleType("torchvision")
    torchvision_models = types.ModuleType("torchvision.models")
    torchvision_models.resnet50 = lambda weights=None: fake_resnet
    monkeypatch.setitem(sys.modules, "torchvision", torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.models", torchvision_models)

    backbone = RadImageNetResNet50()
    features = backbone(torch.randn(2, 3, 8, 8))

    assert features.shape == (2, 2048)


def test_radimagenet_freezes_all_bottlenecks_normalizes_and_freezes_bn(
) -> None:
    backbone = DummyRadImageNetResNet50()
    received = []
    backbone[0].register_forward_pre_hook(
        lambda _module, inputs: received.append(inputs[0].detach().clone())
    )
    model = KneeAbnormalityModel(
        backbone=backbone,
        backbone_family="radimagenet-resnet50",
        image_mean=RADIMAGENET_MEAN,
        image_std=RADIMAGENET_STD,
        num_targets=2,
        unfreeze_last=0,
        dropout=0.0,
        num_slots=1,
        max_series_per_slot=1,
        window_count=1,
        slices_per_slot=3,
        attention_hidden_size=4,
    )
    blocks = [block for stage in list(backbone.children())[4:8] for block in stage]

    assert len(blocks) == 16
    assert not any(parameter.requires_grad for parameter in backbone.parameters())

    model.train()
    assert backbone[1].training is False
    assert blocks[0][0].training is False
    assert blocks[-1][0].training is False

    images = torch.full((1, 1, 1, 1, 3, 4, 4), 0.25)
    present = torch.ones((1, 1, 1, 1), dtype=torch.bool)
    positions = torch.zeros((1, 1, 1, 1, 3))
    assert model(images, present, positions).shape == (1, 2)
    assert torch.allclose(received[0], torch.full((1, 3, 4, 4), -0.5))


def test_build_radimagenet_uses_official_sequential_state_dict_strictly(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "radimagenet"
    source.mkdir()
    checkpoint = source / "ResNet50.pt"
    expected_backbone = DummyRadImageNetResNet50()
    torch.save(
        {
            f"backbone.{key}": value
            for key, value in expected_backbone.state_dict().items()
        },
        checkpoint,
    )
    loaded_backbone = DummyRadImageNetResNet50()
    monkeypatch.setattr(
        "rsna2026.model.RadImageNetResNet50", lambda: loaded_backbone
    )

    model, found_source = build_radimagenet_resnet50_model(
        input_root=tmp_path,
        weights_file=checkpoint.name,
        num_targets=2,
        unfreeze_last=0,
        dropout=0.0,
        num_slots=1,
        max_series_per_slot=1,
        window_count=1,
        slices_per_slot=3,
        attention_hidden_size=4,
        gradient_checkpointing=False,
    )

    assert found_source == source
    assert model.backbone is loaded_backbone
    assert not any(
        parameter.requires_grad for parameter in model.backbone.parameters()
    )
    assert loaded_backbone.loaded_strict is True
    assert model.feature_size == 4
    assert torch.equal(model.image_mean.flatten(), torch.tensor(RADIMAGENET_MEAN))
    assert torch.equal(model.image_std.flatten(), torch.tensor(RADIMAGENET_STD))


@pytest.mark.parametrize(
    "state_dict",
    [
        {},
        {"0.weight": torch.ones(1)},
        {
            "backbone.0.weight": torch.ones(1),
            "1.weight": torch.ones(1),
        },
        {"module.backbone.0.weight": torch.ones(1)},
        {"backbone.": torch.ones(1)},
    ],
    ids=["empty", "missing-prefix", "mixed-prefix", "unexpected-prefix", "empty-key"],
)
def test_build_radimagenet_rejects_invalid_checkpoint_namespace(
    tmp_path, monkeypatch, state_dict
) -> None:
    source = tmp_path / "radimagenet"
    source.mkdir()
    checkpoint = source / "ResNet50.pt"
    torch.save(state_dict, checkpoint)
    monkeypatch.setattr(
        "rsna2026.model.RadImageNetResNet50", DummyRadImageNetResNet50
    )

    with pytest.raises(RuntimeError, match="Invalid official.*namespace"):
        build_radimagenet_resnet50_model(
            input_root=tmp_path,
            weights_file=checkpoint.name,
            num_targets=2,
            unfreeze_last=6,
            dropout=0.0,
            num_slots=1,
            max_series_per_slot=1,
            window_count=1,
            slices_per_slot=3,
            attention_hidden_size=4,
            gradient_checkpointing=False,
        )


def test_build_radimagenet_rejects_key_collision_after_prefix_stripping(
    tmp_path, monkeypatch
) -> None:
    class CollidingCheckpoint(dict):
        def items(self):
            tensor = torch.ones(1)
            return [
                ("backbone.0.weight", tensor),
                ("backbone.0.weight", tensor),
            ]

    source = tmp_path / "radimagenet"
    source.mkdir()
    checkpoint = source / "ResNet50.pt"
    checkpoint.touch()
    monkeypatch.setattr(
        "rsna2026.model.torch.load",
        lambda *_args, **_kwargs: CollidingCheckpoint(x=1),
    )

    with pytest.raises(RuntimeError, match="Invalid official.*namespace"):
        build_radimagenet_resnet50_model(
            input_root=tmp_path,
            weights_file=checkpoint.name,
            num_targets=2,
            unfreeze_last=6,
            dropout=0.0,
            num_slots=1,
            max_series_per_slot=1,
            window_count=1,
            slices_per_slot=3,
            attention_hidden_size=4,
            gradient_checkpointing=False,
        )


def test_build_radimagenet_reports_strict_checkpoint_mismatch(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "radimagenet"
    source.mkdir()
    checkpoint = source / "ResNet50.pt"
    torch.save({"backbone.unexpected.weight": torch.ones(1)}, checkpoint)
    monkeypatch.setattr(
        "rsna2026.model.RadImageNetResNet50", DummyRadImageNetResNet50
    )

    with pytest.raises(RuntimeError, match="Failed to strictly load"):
        build_radimagenet_resnet50_model(
            input_root=tmp_path,
            weights_file=checkpoint.name,
            num_targets=2,
            unfreeze_last=6,
            dropout=0.0,
            num_slots=1,
            max_series_per_slot=1,
            window_count=1,
            slices_per_slot=3,
            attention_hidden_size=4,
            gradient_checkpointing=False,
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


def test_transformer_attention_masks_windows_and_returns_marginals() -> None:
    head = TransformerTargetAttentionHead(8, 8, 2, 3, 3, 0.0, 2, 2)
    features = torch.randn(2, 2, 2, 4, 8)
    present = torch.tensor(
        [
            [
                [[True, False, True, False], [True, False, False, False]],
                [[False] * 4, [False, True, False, False]],
            ],
            [[[False] * 4, [False] * 4], [[False] * 4, [False] * 4]],
        ]
    )
    positions = torch.randn(2, 2, 2, 4, 3)

    logits, window_attention, series_attention, slot_attention = head(
        features, present, positions, return_attention=True
    )

    assert logits.shape == (2, 3)
    assert window_attention.shape == (2, 3, 2, 2, 4)
    assert series_attention.shape == (2, 3, 2, 2)
    assert slot_attention.shape == (2, 3, 2)
    assert torch.isfinite(logits).all()
    assert torch.all(window_attention.masked_select(~present.unsqueeze(1)) == 0)
    assert torch.allclose(window_attention[0].sum(dim=(1, 2, 3)), torch.ones(3))
    assert torch.all(window_attention[1] == 0)
    assert torch.allclose(series_attention, window_attention.sum(dim=-1))
    assert torch.allclose(slot_attention, series_attention.sum(dim=-1))


def test_transformer_attention_is_invariant_to_series_permutation() -> None:
    torch.manual_seed(7)
    head = TransformerTargetAttentionHead(6, 8, 2, 3, 3, 0.0, 2, 2).eval()
    features = torch.randn(1, 2, 3, 2, 6)
    present = torch.tensor(
        [[
            [[True, False], [True, True], [False, True]],
            [[True, True], [False, False], [True, False]],
        ]]
    )
    positions = torch.randn(1, 2, 3, 2, 3)
    permutation = torch.tensor([2, 0, 1])

    original = head(features, present, positions)
    permuted = head(
        features[:, :, permutation],
        present[:, :, permutation],
        positions[:, :, permutation],
    )

    assert torch.allclose(original, permuted, atol=1e-6)


def test_transformer_layers_create_window_interactions() -> None:
    torch.manual_seed(11)
    head = TransformerTargetAttentionHead(4, 8, 1, 2, 3, 0.0, 2, 2).eval()
    features = torch.randn(1, 1, 1, 2, 4)
    positions = torch.zeros(1, 1, 1, 2, 3)
    only_first = torch.tensor([[[[True, False]]]])
    both = torch.tensor([[[[True, True]]]])

    first_alone = head(features, only_first, positions)
    first_with_neighbor = head(features, both, positions)

    assert not torch.allclose(first_alone, first_with_neighbor)


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
