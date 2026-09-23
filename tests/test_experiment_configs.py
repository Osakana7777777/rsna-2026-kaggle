from __future__ import annotations

from copy import deepcopy

from rsna2026.config import load_config


def test_baseline_uses_requested_single_gpu_training_settings() -> None:
    baseline = load_config("configs/baseline.toml")

    assert baseline["training"]["batch_size_per_device"] == 1
    assert baseline["training"]["gradient_accumulation_steps"] == 8
    assert baseline["training"]["expected_gpu_count"] == 1
    assert baseline["training"]["distributed"] is False
    assert baseline["model"]["gradient_checkpointing"] is False
    assert baseline["model"]["name"] == "dinov2-small"
    assert baseline["pseudo_labels"]["file_name"] == (
        "weak_labels_gpt56_luna_calibrated_synovitis_imputed.csv"
    )
    assert baseline["pseudo_labels"]["max_studies"] == 3303


def test_radimagenet_experiment_only_changes_declared_baseline_settings() -> None:
    baseline = load_config("configs/baseline.toml")
    experiment = load_config("configs/radimagenet-resnet50-224.toml")

    assert experiment["training"]["epochs"] == 20
    assert experiment["training"]["batch_size_per_device"] == 4
    assert experiment["training"]["gradient_accumulation_steps"] == 2
    assert experiment["training"]["ema_decay"] == 0.999
    assert experiment["training"]["expected_gpu_count"] == 1
    assert experiment["training"]["max_runtime_hours"] == 12.0
    assert experiment["model"] == {
        "name": "radimagenet-resnet50",
        "variant": "resnet50",
        "weights_file": "ResNet50.pt",
        "normalization": "minus-one-to-one",
        "input_representation": "2.5d-slice-channels",
        "unfreeze_last": 0,
        "gradient_checkpointing": False,
        "pooling": "global-average",
        "aggregation": "transformer-target-attention",
        "attention_hidden_size": 256,
        "transformer_layers": 2,
        "transformer_heads": 8,
        "dropout": 0.25,
    }

    normalized_experiment = deepcopy(experiment)
    normalized_experiment["training"].update(
        epochs=baseline["training"]["epochs"],
        batch_size_per_device=baseline["training"]["batch_size_per_device"],
        gradient_accumulation_steps=baseline["training"][
            "gradient_accumulation_steps"
        ],
    )
    normalized_experiment["training"].pop("ema_decay")
    normalized_experiment["model"] = baseline["model"]

    assert normalized_experiment == baseline


def test_radimagenet_smoke_only_changes_epoch_count() -> None:
    experiment = load_config("configs/radimagenet-resnet50-224.toml")
    smoke = load_config("configs/radimagenet-resnet50-224-smoke.toml")

    assert smoke["training"]["epochs"] == 1
    normalized_smoke = deepcopy(smoke)
    normalized_smoke["training"]["epochs"] = experiment["training"]["epochs"]

    assert normalized_smoke == experiment
