from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rsna2026.dataset import (
    KneeStudyDataset,
    StudyRecord,
    center_crop_physical_fov,
    preprocess_physical_fov,
    read_dicom,
)
from rsna2026.config import load_config
from rsna2026.train import build_configured_model, model_artifact_metadata


def test_center_crop_uses_row_and_column_spacing_independently() -> None:
    pixels = np.arange(81, dtype=np.float32).reshape(9, 9)

    cropped, padding = center_crop_physical_fov(
        pixels, (10.0, 5.0), 30.0
    )

    assert cropped.shape == (3, 6)
    np.testing.assert_array_equal(cropped, pixels[3:6, 1:7])
    assert padding == (0, 0, 0, 0)


def test_center_crop_pads_small_non_square_image_with_extra_below() -> None:
    pixels = np.arange(6, dtype=np.float32).reshape(2, 3)

    cropped, padding = center_crop_physical_fov(
        pixels, (10.0, 10.0), 50.0
    )

    np.testing.assert_array_equal(cropped, pixels)
    assert padding == (1, 2, 1, 1)


@pytest.mark.parametrize(
    "spacing",
    [None, (), (1.0,), (0.0, 1.0), (-1.0, 1.0), (np.nan, 1.0)],
)
def test_center_crop_rejects_invalid_pixel_spacing(spacing) -> None:
    with pytest.raises(ValueError, match="PixelSpacing"):
        center_crop_physical_fov(np.ones((4, 4)), spacing, 130.0)


def test_preprocessing_windows_real_crop_before_zero_padding() -> None:
    pixels = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)

    processed = preprocess_physical_fov(
        pixels, (10.0, 10.0), 40.0, invert=False
    )

    assert processed.shape == (4, 4)
    assert np.isfinite(processed).all()
    assert np.all(processed[[0, -1], :] == 0.0)
    assert np.all(processed[:, [0, -1]] == 0.0)
    assert processed[1:3, 1:3].max() == pytest.approx(1.0)


def test_read_dicom_returns_finite_requested_shape(monkeypatch) -> None:
    dataset = SimpleNamespace(
        pixel_array=np.arange(35, dtype=np.float32).reshape(5, 7),
        PixelSpacing=(10.0, 10.0),
        RescaleSlope=1.0,
        RescaleIntercept=0.0,
        PhotometricInterpretation="MONOCHROME2",
    )
    monkeypatch.setattr(
        "rsna2026.dataset.pydicom.dcmread", lambda *args, **kwargs: dataset
    )

    image = read_dicom(Path("example.dcm"), 224, 30.0)

    assert tuple(image.shape) == (1, 224, 224)
    assert image.isfinite().all()


def test_cache_identity_includes_physical_fov(tmp_path) -> None:
    record = StudyRecord(
        "study",
        ((Path("a.dcm"), Path("slice.dcm"), Path("b.dcm")),),
        ((0.5, 1.0, 1.0),),
        1,
        1,
    )
    crop_130 = KneeStudyDataset(
        [record], 224, 130.0, cache_dir=tmp_path
    )._cache_path(record)
    crop_140 = KneeStudyDataset(
        [record], 224, 140.0, cache_dir=tmp_path
    )._cache_path(record)

    assert crop_130 != crop_140


def test_artifact_metadata_binds_physical_preprocessing() -> None:
    config = {
        "data": {
            "image_size": 224,
            "physical_fov_mm": 130.0,
            "window_count": 10,
            "window_size": 3,
            "max_series_per_slot": 2,
            "series_classification": "competition-series-table-v1",
            "slots": ["slot"],
        },
        "model": {
            "name": "dinov2-small",
            "variant": "small",
            "unfreeze_last": 6,
            "gradient_checkpointing": True,
            "dropout": 0.25,
            "pooling": "cls-patch-mean",
            "aggregation": "hierarchical-target-series-attention",
            "attention_hidden_size": 256,
            "input_representation": "2.5d-slice-channels",
        },
    }

    baseline = model_artifact_metadata(config)
    config["data"]["physical_fov_mm"] = 140.0

    assert model_artifact_metadata(config) != baseline


def test_transformer_artifact_metadata_includes_architecture_settings() -> None:
    config = load_config("configs/baseline.toml")

    metadata = model_artifact_metadata(config)

    assert metadata["aggregation"] == "transformer-target-attention"
    assert metadata["transformer_layers"] == 2
    assert metadata["transformer_heads"] == 8


def test_configured_model_rejects_legacy_window_settings() -> None:
    config = deepcopy(load_config("configs/baseline.toml"))
    config["data"]["sampled_slices_per_slot"] = 12

    with pytest.raises(ValueError, match="Legacy window configuration"):
        build_configured_model(config, torch.device("cpu"))


def test_coatnet_artifact_metadata_includes_weights_file() -> None:
    config = load_config("configs/coatnet-0-224.toml")

    metadata = model_artifact_metadata(config)

    assert metadata["name"] == "coatnet-0"
    assert metadata["variant"] == "coatnet_0_rw_224.sw_in1k"
    assert metadata["weights_file"] == "model.safetensors"
    assert metadata["pooling"] == "global-average"
    assert metadata["unfreeze_last"] == 6


def test_configured_coatnet_routes_all_model_settings(monkeypatch) -> None:
    config = load_config("configs/coatnet-0-224.toml")
    received = {}

    def fake_build_coatnet_model(**kwargs):
        received.update(kwargs)
        return torch.nn.Linear(1, 1), Path("/mounted/coatnet")

    monkeypatch.setattr(
        "rsna2026.train.build_coatnet_model", fake_build_coatnet_model
    )

    _, source, _, _ = build_configured_model(config, torch.device("cpu"))

    assert source == Path("/mounted/coatnet")
    assert received["architecture"] == "coatnet_0_rw_224.sw_in1k"
    assert received["weights_file"] == "model.safetensors"
    assert received["unfreeze_last"] == 6
    assert received["gradient_checkpointing"] is True
    assert received["num_targets"] == 12
    assert received["aggregation"] == "hierarchical-target-series-attention"


def test_configured_dinov2_routes_transformer_settings(monkeypatch) -> None:
    config = load_config("configs/baseline.toml")
    received = {}

    def fake_build_dinov2_model(**kwargs):
        received.update(kwargs)
        return torch.nn.Linear(1, 1), Path("/mounted/dinov2")

    monkeypatch.setattr(
        "rsna2026.train.build_dinov2_model", fake_build_dinov2_model
    )

    _, source, _, _ = build_configured_model(config, torch.device("cpu"))

    assert source == Path("/mounted/dinov2")
    assert received["aggregation"] == "transformer-target-attention"
    assert received["transformer_layers"] == 2
    assert received["transformer_heads"] == 8


def test_configured_coatnet_rejects_non_224_input() -> None:
    config = deepcopy(load_config("configs/coatnet-0-224.toml"))
    config["data"]["image_size"] = 384

    with pytest.raises(ValueError, match="224-pixel"):
        build_configured_model(config, torch.device("cpu"))


def test_radimagenet_metadata_and_model_settings_are_routed(monkeypatch) -> None:
    config = load_config("configs/radimagenet-resnet50-224.toml")
    metadata = model_artifact_metadata(config)
    received = {}

    def fake_build_radimagenet_model(**kwargs):
        received.update(kwargs)
        return torch.nn.Linear(1, 1), Path("/mounted/radimagenet")

    monkeypatch.setattr(
        "rsna2026.train.build_radimagenet_resnet50_model",
        fake_build_radimagenet_model,
    )

    _, source, _, _ = build_configured_model(config, torch.device("cpu"))

    assert metadata["weights_file"] == "ResNet50.pt"
    assert metadata["normalization"] == "minus-one-to-one"
    assert source == Path("/mounted/radimagenet")
    assert received["weights_file"] == "ResNet50.pt"
    assert received["unfreeze_last"] == 0
    assert received["gradient_checkpointing"] is False
    assert received["aggregation"] == "transformer-target-attention"
