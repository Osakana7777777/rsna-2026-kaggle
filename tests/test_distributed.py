from __future__ import annotations

import argparse

import pytest


np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from rsna2026.train import (  # noqa: E402
    DistributedEvaluationSampler,
    apply_runtime_overrides,
    checkpoint_state_dict,
    folds_for_worker,
    mean_loss_from_totals,
    merge_prediction_shards,
)


def test_fold_workers_train_separate_folds_in_two_parallel_rounds() -> None:
    assert folds_for_worker(4, worker_index=0, worker_count=2) == (0, 2)
    assert folds_for_worker(4, worker_index=1, worker_count=2) == (1, 3)


def test_single_fold_worker_trains_every_fold() -> None:
    assert folds_for_worker(4, worker_index=0, worker_count=1) == (0, 1, 2, 3)


def test_fold_worker_assignment_rejects_invalid_worker() -> None:
    with pytest.raises(ValueError, match="Invalid fold worker"):
        folds_for_worker(4, worker_index=2, worker_count=2)


def test_runtime_overrides_preserve_portable_base_config() -> None:
    config = {
        "paths": {"input_root": "/kaggle/input", "output_dir": "/old"},
        "training": {"expected_gpu_count": 2},
        "data": {
            "study_records_cache_path": "/kaggle/input/cache.json.xz",
            "study_records_cache_sha256": "abc",
        },
    }
    args = argparse.Namespace(
        input_root="/workspace/input",
        output_dir="/workspace/outputs",
        expected_gpu_count=4,
        ema_decay=0.995,
        build_selection_cache=True,
    )

    apply_runtime_overrides(config, args)

    assert config["paths"] == {
        "input_root": "/workspace/input",
        "output_dir": "/workspace/outputs",
    }
    assert config["training"]["expected_gpu_count"] == 4
    assert config["training"]["ema_decay"] == 0.995
    assert config["data"]["study_records_cache_path"] == ""
    assert config["data"]["study_records_cache_sha256"] == ""


def test_evaluation_sampler_partitions_without_padding() -> None:
    first = DistributedEvaluationSampler(7, rank=0, world_size=2)
    second = DistributedEvaluationSampler(7, rank=1, world_size=2)

    assert list(first) == [0, 2, 4, 6]
    assert list(second) == [1, 3, 5]
    assert len(first) == 4
    assert len(second) == 3
    assert sorted([*first, *second]) == list(range(7))


def test_prediction_shards_are_restored_to_dataset_order() -> None:
    first_predictions = np.asarray([[0.0, 0.1], [2.0, 2.1]])
    second_predictions = np.asarray([[1.0, 1.1], [3.0, 3.1]])
    first_labels = first_predictions + 10.0
    second_labels = second_predictions + 10.0

    predictions, labels = merge_prediction_shards(
        [
            ([0, 2], first_predictions, first_labels),
            ([1, 3], second_predictions, second_labels),
        ],
        expected_size=4,
    )

    assert predictions[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert labels is not None
    assert labels[:, 0].tolist() == [10.0, 11.0, 12.0, 13.0]


def test_prediction_merge_rejects_duplicate_indices() -> None:
    values = np.asarray([[0.0]])

    with pytest.raises(ValueError, match="Duplicate"):
        merge_prediction_shards(
            [([0], values, None), ([0], values, None)], expected_size=1
        )


class WrappedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.module = nn.Linear(2, 1)


def test_checkpoint_state_omits_distributed_module_prefix() -> None:
    state = checkpoint_state_dict(WrappedModel())

    assert set(state) == {"weight", "bias"}
    assert all(not name.startswith("module.") for name in state)
    assert all(value.device.type == "cpu" for value in state.values())


def test_global_loss_uses_all_reduced_batch_totals() -> None:
    assert mean_loss_from_totals(6.0 + 10.0, 2.0 + 2.0) == 4.0
