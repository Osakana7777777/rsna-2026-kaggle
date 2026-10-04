"""Train, validate, and run inference for the RSNA knee baseline."""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import json
import lzma
import math
import os
import pickle
import random
import shutil
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from rsna2026.acquisition import CLASSIFICATION_METHOD, SLOT_NAMES
from rsna2026.augmentation import AugmentationConfig, augment_dense_windows
from rsna2026.checkpoint import (
    file_sha256,
    resolve_input_file,
    write_checkpoint_manifest,
)
from rsna2026.config import load_config
from rsna2026.losses import pairwise_logistic_ranking_loss
from rsna2026.constants import TARGETS
from rsna2026.cross_validation import (
    bootstrap_macro_auc_interval,
    create_inner_roles,
    create_outer_folds,
    dataframe_sha256,
    selected_training_folds,
    weighted_binary_cross_entropy,
)
from rsna2026.data import competition_input
from rsna2026.early_stopping import EarlyStopping
from rsna2026.dataset import (
    PREPROCESSING_VERSION,
    SERIES_SELECTION_METHOD,
    SLICE_SELECTION_METHOD,
    KneeStudyDataset,
    StudyRecord,
    build_cached_study_records,
    build_study_records,
    load_study_records_cache,
)
from rsna2026.ensemble import (
    ENSEMBLE_FORMAT_VERSION,
    load_ensemble_manifest,
    percentile_rank_mean,
)
from rsna2026.image_cache import (
    DEFAULT_CACHE_SHARD_COUNT,
    ImageCacheReader,
    build_image_cache_shard,
)
from rsna2026.model import (
    build_coatnet_model,
    build_dinov2_model,
    build_radimagenet_resnet50_model,
)
from rsna2026.swa import configured_swa
from rsna2026.orientation import ORIENTATION_NORMALIZATION_METHOD
from rsna2026.public_labels import (
    confidence_weight,
    load_public_label_table,
    report_hash,
    source_validation,
    stable_collection_hash,
    stable_sample,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an RSNA 2026 experiment")
    parser.add_argument("--config", required=True, help="Path to a TOML config")
    parser.add_argument(
        "--mode",
        choices=(
            "train",
            "inference",
            "prepare-selection-cache",
            "prepare-image-cache",
        ),
        required=True,
    )
    parser.add_argument(
        "--cache-shard-index",
        type=int,
        default=0,
        help="Zero-based persistent image-cache shard index",
    )
    parser.add_argument(
        "--cache-shard-count",
        type=int,
        default=DEFAULT_CACHE_SHARD_COUNT,
        help="Total number of persistent image-cache shards",
    )
    parser.add_argument(
        "--input-root",
        help="Override paths.input_root for the current runtime",
    )
    parser.add_argument(
        "--output-dir",
        help="Override paths.output_dir for the current runtime",
    )
    parser.add_argument(
        "--expected-gpu-count",
        type=int,
        help="Override training.expected_gpu_count for the current runtime",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        help="Enable EMA model selection with this decay",
    )
    parser.add_argument(
        "--build-selection-cache",
        action="store_true",
        help="Build the study-record cache instead of using the configured mount",
    )
    return parser.parse_args(argv)


def apply_runtime_overrides(
    config: dict[str, object], args: argparse.Namespace
) -> None:
    """Apply explicit host-specific paths and GPU topology."""
    paths = config.get("paths")
    training = config.get("training")
    if not isinstance(paths, dict) or not isinstance(training, dict):
        raise TypeError("paths and training configurations must be tables")
    if args.input_root is not None:
        paths["input_root"] = args.input_root
    if args.output_dir is not None:
        paths["output_dir"] = args.output_dir
    if args.expected_gpu_count is not None:
        if args.expected_gpu_count < 1:
            raise ValueError("Expected GPU count must be positive")
        training["expected_gpu_count"] = args.expected_gpu_count
    ema_decay = getattr(args, "ema_decay", None)
    if ema_decay is not None:
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        training["ema_decay"] = ema_decay
    if args.build_selection_cache:
        data = config.get("data")
        if not isinstance(data, dict):
            raise TypeError("data configuration must be a table")
        data["study_records_cache_path"] = ""
        data["study_records_cache_sha256"] = ""


def apply_study_records_cache_environment(config: dict[str, object]) -> None:
    """Apply a cache prepared immediately before distributed training."""
    cache_path = os.environ.get("RSNA2026_STUDY_RECORDS_CACHE_PATH", "")
    cache_sha256 = os.environ.get("RSNA2026_STUDY_RECORDS_CACHE_SHA256", "")
    if bool(cache_path) != bool(cache_sha256):
        raise ValueError(
            "Study-record cache path and SHA-256 environment variables must "
            "be set together"
        )
    if not cache_path:
        return
    data_config = config.get("data")
    if not isinstance(data_config, dict):
        raise TypeError("Data configuration must be a table")
    data_config["study_records_cache_path"] = cache_path
    data_config["study_records_cache_sha256"] = cache_sha256


def seed_everything(seed: int) -> None:
    """Set reproducible random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class DistributedContext:
    """Describe one process in a fixed-size distributed training job."""

    enabled: bool
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str | None = None

    @property
    def primary(self) -> bool:
        return self.rank == 0


def folds_for_worker(
    fold_count: int, worker_index: int, worker_count: int
) -> tuple[int, ...]:
    """Assign folds round-robin so each GPU trains at most one fold at a time."""
    if fold_count <= 0:
        raise ValueError("Fold count must be positive")
    if worker_count <= 0 or worker_index < 0 or worker_index >= worker_count:
        raise ValueError("Invalid fold worker index or count")
    return tuple(range(worker_index, fold_count, worker_count))


def cv_fold_execution(
    run_folds: Sequence[int],
    training_config: dict[str, object],
    distributed: DistributedContext,
) -> tuple[tuple[int, ...], DistributedContext, bool]:
    """Choose shared-fold data parallelism or independent fold workers."""
    data_parallel = (
        distributed.enabled
        and training_config.get("distributed_strategy", "one-fold-per-gpu")
        == "data-parallel"
    )
    if data_parallel:
        worker_folds = tuple(run_folds)
        fold_context = distributed
    else:
        worker_fold_indices = folds_for_worker(
            len(run_folds),
            distributed.rank if distributed.enabled else 0,
            distributed.world_size if distributed.enabled else 1,
        )
        worker_folds = tuple(run_folds[index] for index in worker_fold_indices)
        fold_context = DistributedContext(
            enabled=False, local_rank=distributed.local_rank,
        )
    return worker_folds, fold_context, data_parallel


class DistributedEvaluationSampler(Sampler[int]):
    """Shard evaluation indices without padding or duplicate studies."""

    def __init__(self, size: int, rank: int, world_size: int) -> None:
        if size < 0:
            raise ValueError("Evaluation size must be non-negative")
        if world_size <= 0 or rank < 0 or rank >= world_size:
            raise ValueError("Invalid distributed rank or world size")
        self.size = size
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, self.size, self.world_size))

    def __len__(self) -> int:
        if self.rank >= self.size:
            return 0
        return (self.size - 1 - self.rank) // self.world_size + 1


def initialize_distributed(
    config: dict[str, object], mode: str
) -> DistributedContext:
    """Initialize the requested NCCL process group."""
    training_config = config.get("training", {})
    enabled = (
        mode == "train"
        and isinstance(training_config, dict)
        and bool(training_config.get("distributed", False))
    )
    if not enabled:
        return DistributedContext(enabled=False)

    strategy = training_config.get("distributed_strategy", "one-fold-per-gpu")
    if strategy not in {"one-fold-per-gpu", "data-parallel"}:
        raise ValueError(f"Unsupported distributed strategy: {strategy}")
    expected = int(training_config.get("expected_gpu_count", 2))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if expected < 1:
        raise ValueError("Expected GPU count must be positive")
    if world_size != expected:
        raise RuntimeError(
            f"Expected torchrun world size {expected}, received {world_size}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != expected:
        raise RuntimeError(
            f"Expected exactly {expected} visible CUDA devices, found "
            f"{torch.cuda.device_count()}"
        )
    torch.cuda.set_device(local_rank)
    max_runtime_hours = float(training_config.get("max_runtime_hours", 12.0))
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(hours=max_runtime_hours + 1.0),
    )
    print(
        f"Initialized {strategy} NCCL rank {rank}/{world_size} on "
        f"GPU {local_rank}: {torch.cuda.get_device_name(local_rank)}; "
        f"batch_per_gpu={training_config['batch_size_per_device']}, "
        f"accumulation={training_config['gradient_accumulation_steps']}",
        flush=True,
    )
    return DistributedContext(
        enabled=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        backend="nccl",
    )


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model without a distributed wrapper."""
    return getattr(model, "module", model)


class ExponentialMovingAverage:
    """Maintain an evaluation model using parameter exponential averages."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        if not math.isfinite(decay) or not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be finite and in [0, 1)")
        self.decay = float(decay)
        self.model = copy.deepcopy(unwrap_model(model)).eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update floating tensors and copy non-floating buffers exactly."""
        source_state = unwrap_model(model).state_dict()
        averaged_state = self.model.state_dict()
        if source_state.keys() != averaged_state.keys():
            raise RuntimeError("EMA model state does not match the training model")
        for name, averaged in averaged_state.items():
            source = source_state[name].detach()
            if torch.is_floating_point(averaged) or torch.is_complex(averaged):
                averaged.lerp_(source, 1.0 - self.decay)
            else:
                averaged.copy_(source)


def configured_ema(
    model: nn.Module, training_config: dict[str, object]
) -> ExponentialMovingAverage | None:
    """Build EMA only when an experiment explicitly configures its decay."""
    decay = training_config.get("ema_decay")
    if decay is None:
        return None
    return ExponentialMovingAverage(model, float(decay))


def checkpoint_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Copy a wrapper-free model state to CPU for a reusable checkpoint."""
    return {
        name: value.detach().cpu().clone()
        for name, value in unwrap_model(model).state_dict().items()
    }


def mean_loss_from_totals(total_loss: float, total_batches: float) -> float:
    """Calculate the global mean after distributed loss totals are reduced."""
    return total_loss / max(total_batches, 1.0)


def merge_prediction_shards(
    shards: Sequence[tuple[Sequence[int], np.ndarray, np.ndarray | None]],
    expected_size: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Restore distributed evaluation predictions to dataset order."""
    indexed_predictions: dict[int, np.ndarray] = {}
    indexed_labels: dict[int, np.ndarray] = {}
    labels_expected = any(labels is not None for _, _, labels in shards)
    for indices, predictions, labels in shards:
        if len(indices) != len(predictions):
            raise ValueError("Prediction shard length does not match its indices")
        if labels is not None and len(labels) != len(indices):
            raise ValueError("Label shard length does not match its indices")
        for offset, index in enumerate(indices):
            if index in indexed_predictions:
                raise ValueError(f"Duplicate distributed prediction index: {index}")
            indexed_predictions[int(index)] = predictions[offset]
            if labels is not None:
                indexed_labels[int(index)] = labels[offset]
    expected_indices = set(range(expected_size))
    if set(indexed_predictions) != expected_indices:
        missing = sorted(expected_indices.difference(indexed_predictions))
        extra = sorted(set(indexed_predictions).difference(expected_indices))
        raise ValueError(
            f"Distributed predictions are incomplete; missing={missing[:3]}, "
            f"extra={extra[:3]}"
        )
    predictions = np.stack(
        [indexed_predictions[index] for index in range(expected_size)]
    )
    if not labels_expected:
        return predictions, None
    if set(indexed_labels) != expected_indices:
        raise ValueError("Distributed labels are incomplete")
    labels = np.stack([indexed_labels[index] for index in range(expected_size)])
    return predictions, labels


def target_auc_scores(
    labels: np.ndarray, predictions: np.ndarray
) -> dict[str, float]:
    """Compute per-target ROC AUC values without an extra dependency."""
    scores: dict[str, float] = {}
    for target_index in range(labels.shape[1]):
        target = labels[:, target_index]
        prediction = predictions[:, target_index]
        valid = np.isfinite(target) & np.isfinite(prediction)
        target = (target[valid] > 0.5).astype(int)
        prediction = prediction[valid]
        positives = target == 1
        negatives = target == 0
        if not positives.any() or not negatives.any():
            scores[TARGETS[target_index]] = float("nan")
            continue
        order = np.argsort(prediction, kind="mergesort")
        ranks = np.empty(len(order), dtype=np.float64)
        ranks[order] = np.arange(1, len(order) + 1)
        _, inverse, counts = np.unique(
            prediction, return_inverse=True, return_counts=True
        )
        if np.any(counts > 1):
            sums = np.bincount(inverse, weights=ranks)
            ranks = sums[inverse] / counts[inverse]
        positive_ranks = ranks[positives].sum()
        num_positive = int(positives.sum())
        num_negative = int(negatives.sum())
        auc = (
            positive_ranks - num_positive * (num_positive + 1) / 2
        ) / (num_positive * num_negative)
        scores[TARGETS[target_index]] = float(auc)
    return scores


def macro_auc(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Compute the unweighted mean across targets having both classes."""
    scores = target_auc_scores(labels, predictions).values()
    finite = [score for score in scores if math.isfinite(score)]
    return float(np.mean(finite)) if finite else float("nan")


def locate_pseudo_labels(config: dict[str, object]) -> Path:
    """Resolve the pinned public table, with an exact-name fallback."""
    pseudo_config = config["pseudo_labels"]
    if not isinstance(pseudo_config, dict):
        raise TypeError("pseudo_labels configuration must be a table")
    configured = Path(str(pseudo_config["source_path"]))
    if configured.is_file():
        return configured
    input_root = Path(str(config["paths"]["input_root"]))  # type: ignore[index]
    file_name = str(pseudo_config["file_name"])
    candidates = sorted(input_root.glob(f"**/{file_name}"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one {file_name} beneath {input_root}, found {candidates}"
        )
    return candidates[0]


def load_pseudo_supervision(
    train_table: pd.DataFrame,
    config: dict[str, object],
    seed: int,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    dict[str, object],
    dict[str, object],
]:
    """Build public-soft-label training and all-gold validation data."""
    pseudo_config = config["pseudo_labels"]
    if not isinstance(pseudo_config, dict):
        raise TypeError("pseudo_labels configuration must be a table")
    pseudo_path = locate_pseudo_labels(config)
    public = load_public_label_table(
        pseudo_path,
        expected_sha256=str(pseudo_config["sha256"]),
        expected_rows=int(pseudo_config["expected_rows"]),
    )

    table = train_table.copy()
    table["StudyInstanceUID"] = table["StudyInstanceUID"].astype(str)
    competition_uids = set(table["StudyInstanceUID"])
    if competition_uids != set(public.rows):
        missing = sorted(competition_uids.difference(public.rows))
        extra = sorted(set(public.rows).difference(competition_uids))
        raise ValueError(
            f"Public and competition UID sets differ; missing={missing[:3]}, "
            f"extra={extra[:3]}"
        )
    gold_mask = table[TARGETS].notna().all(axis=1)
    validation_table = table.loc[gold_mask].reset_index(drop=True)
    if validation_table.empty:
        raise ValueError("No fully labeled studies are available for validation")
    expected_validation = int(
        pseudo_config.get("expected_validation_studies", 0)
    )
    if expected_validation and len(validation_table) != expected_validation:
        raise ValueError(
            f"Expected {expected_validation} gold validation studies, "
            f"found {len(validation_table)}"
        )

    gold_records = validation_table[["StudyInstanceUID", *TARGETS]].to_dict(
        orient="records"
    )
    label_validation = source_validation(gold_records, public.rows)
    minimum_auc = float(pseudo_config["minimum_source_macro_auc"])
    gate = {
        "all_gold_studies_matched": label_validation["matched_gold_studies"]
        == expected_validation,
        "all_targets_scored": label_validation["scored_targets"] == len(TARGETS),
        "macro_auc_at_least_minimum": math.isfinite(label_validation["macro_auc"])
        and label_validation["macro_auc"] >= minimum_auc,
    }
    gate["passed"] = all(gate.values())
    label_validation["minimum_macro_auc"] = minimum_auc
    label_validation["acceptance_gate"] = gate
    table["report_hash"] = table["Report"].fillna("").astype(str).map(report_hash)
    gold_hashes = set(table.loc[gold_mask, "report_hash"])
    non_gold = table.loc[~gold_mask].copy()
    duplicate_report = non_gold["report_hash"].isin(gold_hashes)
    candidates = non_gold.loc[~duplicate_report].copy()
    candidate_labels = np.asarray(
        [public.rows[uid] for uid in candidates["StudyInstanceUID"]],
        dtype=np.float32,
    )
    if public.confidence_weights is not None and public.masks is not None:
        candidate_weights = np.asarray(
            [
                [
                    weight * public.masks[uid]
                    for weight in public.confidence_weights[uid]
                ]
                for uid in candidates["StudyInstanceUID"]
            ],
            dtype=np.float32,
        )
        supervision_format = "explicit-confidence-weight-and-mask"
    else:
        candidate_weights = np.asarray(
            [
                [
                    confidence_weight(
                        value,
                        str(pseudo_config.get("confidence_weight_method", "legacy-distance")),
                    )
                    for value in row
                ]
                for row in candidate_labels
            ],
            dtype=np.float32,
        )
        supervision_format = "probability-derived-confidence-weight"
    eligible = candidate_weights.sum(axis=1) > 0
    candidates = candidates.loc[eligible].reset_index(drop=True)
    candidate_labels = candidate_labels[eligible]
    candidate_weights = candidate_weights[eligible]

    maximum = int(pseudo_config["max_studies"])
    chosen_uids = stable_sample(candidates["StudyInstanceUID"], maximum, seed)
    candidate_positions = {
        uid: index for index, uid in enumerate(candidates["StudyInstanceUID"])
    }
    chosen_positions = np.asarray(
        [candidate_positions[uid] for uid in chosen_uids], dtype=np.int64
    )
    training_table = candidates.iloc[chosen_positions].reset_index(drop=True)
    training_labels = candidate_labels[chosen_positions]
    training_weights = candidate_weights[chosen_positions]

    validation_labels = validation_table[TARGETS].to_numpy(dtype=np.float32)
    validation_weights = np.ones_like(validation_labels, dtype=np.float32)
    teacher_statistics = {}
    for index, target in enumerate(TARGETS):
        labels = training_labels[:, index]
        weights = training_weights[:, index]
        teacher_statistics[target] = {
            "nonzero_weight_cells": int(np.sum(weights > 0)),
            "weight_sum": float(weights.sum()),
            "positive_expected_mass": float(np.sum(weights * labels)),
            "negative_expected_mass": float(np.sum(weights * (1.0 - labels))),
            "above_half": int(np.sum(labels > 0.5)),
            "below_half": int(np.sum(labels < 0.5)),
            "exactly_half": int(np.sum(labels == 0.5)),
        }
    provenance: dict[str, object] = {
        "pseudo_label_dataset": str(pseudo_config["dataset"]),
        "pseudo_label_source": str(pseudo_path),
        "pseudo_label_file": str(pseudo_config["file_name"]),
        "pseudo_labels_sha256": public.sha256,
        "pseudo_supervision_format": supervision_format,
        "masked_pseudo_label_studies": (
            sum(mask == 0 for mask in public.masks.values())
            if public.masks is not None
            else 0
        ),
        "training_study_uids_sha256": stable_collection_hash(chosen_uids),
        "gold_report_duplicate_candidates_excluded": int(duplicate_report.sum()),
        "eligible_pseudo_label_studies": len(candidates),
        "teacher_statistics": teacher_statistics,
    }
    return (
        training_table,
        training_labels,
        training_weights,
        validation_table,
        validation_labels,
        validation_weights,
        provenance,
        label_validation,
    )


def split_indices(
    count: int, validation_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic study-level train/validation split."""
    if count < 2:
        raise ValueError("At least two labeled studies are required")
    generator = np.random.default_rng(seed)
    indices = generator.permutation(count)
    validation_count = min(max(1, round(count * validation_fraction)), count - 1)
    return indices[validation_count:], indices[:validation_count]


def subset_records(
    records: Sequence[StudyRecord], indices: np.ndarray
) -> list[StudyRecord]:
    return [records[int(index)] for index in indices]


def make_loader(
    records: Sequence[StudyRecord],
    image_size: int,
    physical_fov_mm: float,
    labels: np.ndarray | None,
    label_weights: np.ndarray | None,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    cache_dir: str | Path | None,
    sampler: Sampler[int] | None = None,
    image_cache: ImageCacheReader | None = None,
) -> DataLoader:
    dataset = KneeStudyDataset(
        records,
        image_size,
        physical_fov_mm,
        labels,
        label_weights,
        cache_dir,
        image_cache,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def positive_weights(
    labels: np.ndarray, label_weights: np.ndarray
) -> torch.Tensor:
    """Estimate bounded class weights from weighted soft-label mass."""
    weights = []
    for column, confidence in zip(labels.T, label_weights.T):
        valid = np.isfinite(column) & np.isfinite(confidence) & (confidence > 0)
        positives = float(np.sum(confidence[valid] * column[valid]))
        negatives = float(np.sum(confidence[valid] * (1.0 - column[valid])))
        ratio = negatives / max(positives, 1.0)
        weights.append(min(max(ratio, 0.25), 8.0))
    return torch.tensor(weights, dtype=torch.float32)


def configured_scheduler(
    optimizer: torch.optim.Optimizer,
    training: dict[str, object],
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Use fixed learning rates when requested, retaining existing defaults."""
    kind = str(training.get("scheduler", "one-cycle"))
    if kind == "constant":
        if training.get("swa_enabled", False):
            raise ValueError("Constant learning rates cannot be combined with SWA")
        return None
    if kind != "one-cycle":
        raise ValueError(f"Unsupported learning-rate scheduler: {kind}")
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[float(training["backbone_learning_rate"]),
                float(training["head_learning_rate"])],
        epochs=int(training["epochs"]),
        steps_per_epoch=steps_per_epoch,
        pct_start=float(training["one_cycle_warmup_fraction"]),
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    target_weights: torch.Tensor,
    augmentation: AugmentationConfig,
    gradient_accumulation_steps: int,
    distributed: DistributedContext | None = None,
    ema: ExponentialMovingAverage | None = None,
    pairwise_ranking_loss_weight: float = 0.0,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0
    if gradient_accumulation_steps <= 0:
        raise ValueError("Gradient accumulation steps must be positive")
    if not math.isfinite(pairwise_ranking_loss_weight) or pairwise_ranking_loss_weight < 0:
        raise ValueError("Ranking loss weight must be finite and nonnegative")
    optimizer.zero_grad(set_to_none=True)
    for batch_index, (images, present, positions, labels, label_weights) in enumerate(
        loader
    ):
        images = images.to(device, non_blocking=True)
        present = present.to(device, non_blocking=True)
        positions = positions.to(device, non_blocking=True)
        images = augment_dense_windows(images, present, augmentation)
        labels = labels.to(device, non_blocking=True)
        label_weights = label_weights.to(device, non_blocking=True)
        valid = torch.isfinite(labels) & torch.isfinite(label_weights)
        targets = torch.nan_to_num(labels, nan=0.0)
        weights = torch.where(valid, label_weights, 0.0)
        should_step = (
            (batch_index + 1) % gradient_accumulation_steps == 0
            or batch_index + 1 == len(loader)
        )
        sync_context = contextlib.nullcontext()
        if (
            distributed is not None
            and distributed.enabled
            and not should_step
            and hasattr(model, "no_sync")
        ):
            sync_context = model.no_sync()
        with sync_context, torch.autocast(
            device_type=device.type, enabled=device.type == "cuda"
        ):
            logits = model(images, present, positions)
            losses = functional.binary_cross_entropy_with_logits(
                logits, targets, reduction="none", pos_weight=target_weights
            )
            loss = (losses * weights).sum() / weights.sum().clamp_min(1.0)
            if pairwise_ranking_loss_weight > 0:
                loss = loss + pairwise_ranking_loss_weight * pairwise_logistic_ranking_loss(
                    logits, labels, weights,
                    distributed=distributed is not None and distributed.enabled,
                )
            group_start = (
                batch_index // gradient_accumulation_steps
            ) * gradient_accumulation_steps
            group_size = min(
                gradient_accumulation_steps, len(loader) - group_start
            )
            backward_loss = loss / group_size
        scaler.scale(backward_loss).backward()
        if should_step:
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_ran = scaler.get_scale() >= previous_scale
            if ema is not None and optimizer_ran:
                ema.update(model)
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.detach().cpu())
        total_batches += 1
    totals = torch.tensor(
        [total_loss, float(total_batches)], dtype=torch.float64, device=device
    )
    if distributed is not None and distributed.enabled:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return mean_loss_from_totals(float(totals[0].cpu()), float(totals[1].cpu()))


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[
    np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray
]:
    """Predict once and return window, series, and slot attention weights."""
    model.eval()
    predictions = []
    labels = []
    window_attentions = []
    series_attentions = []
    slot_attentions = []
    for images, present, positions, batch_labels, _ in loader:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits, window_attention, series_attention, slot_attention = model(
                images.to(device, non_blocking=True),
                present.to(device, non_blocking=True),
                positions.to(device, non_blocking=True),
                return_attention=True,
            )
        predictions.append(torch.sigmoid(logits.float()).cpu().numpy())
        window_attentions.append(window_attention.float().cpu().numpy())
        series_attentions.append(series_attention.float().cpu().numpy())
        slot_attentions.append(slot_attention.float().cpu().numpy())
        if batch_labels.numel():
            labels.append(batch_labels.numpy())
    prediction_array = (
        np.concatenate(predictions)
        if predictions
        else np.empty((0, len(TARGETS)))
    )
    label_array = np.concatenate(labels) if labels else None
    return (
        prediction_array,
        label_array,
        np.concatenate(window_attentions),
        np.concatenate(series_attentions),
        np.concatenate(slot_attentions),
    )


def predict_distributed(
    model: nn.Module,
    loader: DataLoader,
    sampler: DistributedEvaluationSampler,
    expected_size: int,
    device: torch.device,
    distributed: DistributedContext,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Predict one non-overlapping shard and restore order on rank zero."""
    predictions, labels, window_attention, series_attention, slot_attention = predict(
        unwrap_model(model), loader, device
    )
    local = (
        list(sampler),
        predictions,
        labels,
        window_attention,
        series_attention,
        slot_attention,
    )
    gathered: list[
        tuple[
            Sequence[int],
            np.ndarray,
            np.ndarray | None,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]
        | None
    ] = [None] * distributed.world_size
    dist.all_gather_object(gathered, local)
    if not distributed.primary:
        return None, None, None, None, None
    complete = [shard for shard in gathered if shard is not None]
    merged_predictions, merged_labels = merge_prediction_shards(
        [
            (indices, values, labels)
            for indices, values, labels, _, _, _ in complete
        ],
        expected_size,
    )
    indexed_window = {}
    indexed_series = {}
    indexed_slot = {}
    for indices, _, _, windows, series, slots in complete:
        for offset, index in enumerate(indices):
            indexed_window[int(index)] = windows[offset]
            indexed_series[int(index)] = series[offset]
            indexed_slot[int(index)] = slots[offset]
    return (
        merged_predictions,
        merged_labels,
        np.stack([indexed_window[index] for index in range(expected_size)]),
        np.stack([indexed_series[index] for index in range(expected_size)]),
        np.stack([indexed_slot[index] for index in range(expected_size)]),
    )


def write_submission(
    sample_submission: pd.DataFrame,
    study_uids: Sequence[str],
    predictions: np.ndarray,
    path: Path,
) -> None:
    """Write predictions in the exact sample-submission order."""
    expected = ["StudyInstanceUID", *TARGETS]
    if list(sample_submission.columns) != expected:
        raise ValueError(
            f"Unexpected submission columns: {list(sample_submission.columns)}"
        )
    prediction_table = pd.DataFrame(predictions, columns=TARGETS)
    prediction_table.insert(0, "StudyInstanceUID", list(study_uids))
    submission = sample_submission[["StudyInstanceUID"]].merge(
        prediction_table, on="StudyInstanceUID", how="left"
    )
    submission[TARGETS] = submission[TARGETS].fillna(0.5).clip(0.0, 1.0)
    submission.to_csv(path, index=False)


def slice_selection_report(
    records_by_split: dict[str, Sequence[StudyRecord]],
    window_count: int,
    max_series_per_slot: int,
) -> dict[str, object]:
    """Create an auditable record of every selected series and slice."""
    report: dict[str, object] = {
        "series_classification": CLASSIFICATION_METHOD,
        "series_selection_method": SERIES_SELECTION_METHOD,
        "slice_selection_method": SLICE_SELECTION_METHOD,
        "slots": list(SLOT_NAMES),
        "max_series_per_slot": max_series_per_slot,
        "window_count": window_count,
        "window_size": 3,
        "orientation_normalization": ORIENTATION_NORMALIZATION_METHOD,
        "medial_side": "image-right",
        "splits": {},
    }

    def distributions(
        records: Sequence[StudyRecord], attribute: str
    ) -> dict[str, dict[str, float | int | None]]:
        result = {}
        for slot in SLOT_NAMES:
            values = [
                float(getattr(selection, attribute))
                for record in records
                for selection in record.selections
                if selection.slot == slot
                and selection.selected_windows
                and float(getattr(selection, attribute)) > 0.0
            ]
            result[slot] = {
                "series": len(values),
                "minimum": min(values) if values else None,
                "median": median(values) if values else None,
                "maximum": max(values) if values else None,
            }
        return result
    split_reports: dict[str, object] = {}
    for split, records in records_by_split.items():
        if any(
            record.num_slots != len(SLOT_NAMES)
            or record.max_series_per_slot != max_series_per_slot
            or len(record.selections)
            != len(SLOT_NAMES) * max_series_per_slot
            for record in records
        ):
            raise ValueError(f"Invalid multi-series record layout in split {split}")
        methods = Counter(
            selection.ordering_method
            for record in records
            for selection in record.selections
        )
        selected_by_rank = Counter(
            (selection.slot, selection.series_rank)
            for record in records
            for selection in record.selections
            if selection.series_uid is not None
        )
        weighting_sources = Counter(
            classification.weighting_source
            for record in records
            for classification in record.classifications
        )
        fat_suppression_sources = Counter(
            classification.fat_suppression_source
            for record in records
            for classification in record.classifications
        )
        unassigned_series = sum(
            not classification.matching_slots
            for record in records
            for classification in record.classifications
        )
        classification_sources = Counter(
            classification.classification_source
            for record in records
            for classification in record.classifications
        )
        unassigned_reasons = Counter(
            classification.unassigned_reason
            for record in records
            for classification in record.classifications
            if not classification.matching_slots
        )
        candidate_series_by_slot = Counter(
            slot
            for record in records
            for classification in record.classifications
            for slot in classification.matching_slots
        )
        selected_relative_paths = {
            path
            for record in records
            for selection in record.selections
            for window in selection.selected_windows
            for path in window
        }
        valid_windows_by_rank = Counter(
            (selection.slot, selection.series_rank)
            for record in records
            for selection in record.selections
            for _ in selection.selected_windows
        )
        rejected_geometry = Counter(
            selection.ordering_method
            for record in records
            for selection in record.selections
            if selection.ordering_method.startswith("invalid-geometry:")
        )
        entries = [
            {
                "study_uid": record.study_uid,
                "series": [asdict(selection) for selection in record.selections],
                "series_classifications": [
                    asdict(classification)
                    for classification in record.classifications
                ],
                "orientation_audits": [
                    asdict(audit) for audit in record.orientation_audits
                ],
            }
            for record in records
        ]
        disagreement_studies = sorted(
            record.study_uid
            for record in records
            if any(
                audit.tag_geometry_disagreement
                for audit in record.orientation_audits
            )
        )
        orientation_audit_reasons = Counter(
            audit.audit_reason
            for record in records
            for audit in record.orientation_audits
            if audit.audit_reason is not None
        )
        summary = {
            "studies": len(records),
            "series_by_ordering_method": dict(sorted(methods.items())),
            "selected_series": sum(selected_by_rank.values()),
            "selected_series_by_slot": {
                slot: sum(
                    selected_by_rank.get((slot, rank), 0)
                    for rank in range(1, max_series_per_slot + 1)
                )
                for slot in SLOT_NAMES
            },
            "selected_series_by_slot_and_rank": {
                slot: {
                    str(rank): selected_by_rank.get((slot, rank), 0)
                    for rank in range(1, max_series_per_slot + 1)
                }
                for slot in SLOT_NAMES
            },
            "missing_slots_by_slot": {
                slot: len(records) - selected_by_rank.get((slot, 1), 0)
                for slot in SLOT_NAMES
            },
            "missing_series_by_slot_and_rank": {
                slot: {
                    str(rank): len(records)
                    - selected_by_rank.get((slot, rank), 0)
                    for rank in range(1, max_series_per_slot + 1)
                }
                for slot in SLOT_NAMES
            },
            "valid_windows_by_slot": {
                slot: sum(
                    valid_windows_by_rank.get((slot, rank), 0)
                    for rank in range(1, max_series_per_slot + 1)
                )
                for slot in SLOT_NAMES
            },
            "valid_windows_by_slot_and_rank": {
                slot: {
                    str(rank): valid_windows_by_rank.get((slot, rank), 0)
                    for rank in range(1, max_series_per_slot + 1)
                }
                for slot in SLOT_NAMES
            },
            "window_capacity_by_slot": {
                slot: len(records) * max_series_per_slot * window_count
                for slot in SLOT_NAMES
            },
            "geometry_rejections": dict(sorted(rejected_geometry.items())),
            "large_gaps": sum(
                selection.large_gaps
                for record in records
                for selection in record.selections
            ),
            "duplicate_positions": sum(
                selection.duplicate_positions
                for record in records
                for selection in record.selections
            ),
            "physical_coverage_mm_by_slot": distributions(
                records, "physical_coverage_mm"
            ),
            "median_spacing_mm_by_slot": distributions(
                records, "median_spacing_mm"
            ),
            "classified_series": sum(
                len(record.classifications) for record in records
            ),
            "unassigned_series": unassigned_series,
            "unassigned_reasons": dict(sorted(unassigned_reasons.items())),
            "classification_sources": dict(sorted(classification_sources.items())),
            "candidate_series_by_slot": {
                slot: candidate_series_by_slot.get(slot, 0) for slot in SLOT_NAMES
            },
            "selected_unique_cache_paths": len(selected_relative_paths),
            "weighting_sources": dict(sorted(weighting_sources.items())),
            "fat_suppression_sources": dict(
                sorted(fat_suppression_sources.items())
            ),
            "orientation_audited_series": sum(
                len(record.orientation_audits) for record in records
            ),
            "orientation_audit_reasons": dict(
                sorted(orientation_audit_reasons.items())
            ),
            "laterality_disagreement_studies": disagreement_studies,
            "laterality_disagreement_study_count": len(disagreement_studies),
        }
        split_reports[split] = {"summary": summary, "studies": entries}
        print(f"{split} slice selection: {summary}", flush=True)
    report["splits"] = split_reports
    return report


def model_artifact_metadata(config: dict[str, object]) -> dict[str, object]:
    """Return architecture fields that must match across both notebook stages."""
    model_config = config["model"]
    data_config = config["data"]
    metadata = {
        "name": str(model_config["name"]),  # type: ignore[index]
        "variant": str(model_config["variant"]),  # type: ignore[index]
        "num_targets": len(TARGETS),
        "targets": list(TARGETS),
        "unfreeze_last": int(model_config["unfreeze_last"]),  # type: ignore[index]
        "gradient_checkpointing": bool(  # type: ignore[index]
            model_config["gradient_checkpointing"]
        ),
        "dropout": float(model_config["dropout"]),  # type: ignore[index]
        "pooling": str(model_config["pooling"]),  # type: ignore[index]
        "aggregation": str(model_config["aggregation"]),  # type: ignore[index]
        "attention_hidden_size": int(  # type: ignore[index]
            model_config["attention_hidden_size"]
        ),
        "image_size": int(data_config["image_size"]),  # type: ignore[index]
        "preprocessing_version": PREPROCESSING_VERSION,
        "physical_fov_mm": float(  # type: ignore[index]
            data_config["physical_fov_mm"]
        ),
        "physical_crop_center": "image-center",
        "physical_crop_padding": "constant-zero",
        "orientation_normalization": ORIENTATION_NORMALIZATION_METHOD,
        "medial_side": "image-right",
        "slice_selection_method": SLICE_SELECTION_METHOD,
        "window_count": int(data_config["window_count"]),  # type: ignore[index]
        "window_size": int(data_config["window_size"]),  # type: ignore[index]
        "max_series_per_slot": int(  # type: ignore[index]
            data_config["max_series_per_slot"]
        ),
        "input_layout": "study-slot-series-window-channel-height-width",
        "position_features": [
            "physical-relative-position",
            "preceding-gap-over-median",
            "following-gap-over-median",
        ],
        "input_representation": str(  # type: ignore[index]
            model_config["input_representation"]
        ),
        "series_classification": str(  # type: ignore[index]
            data_config["series_classification"]
        ),
        "slots": list(data_config["slots"]),  # type: ignore[index]
    }
    if "weights_file" in model_config:  # type: ignore[operator]
        metadata["weights_file"] = str(  # type: ignore[index]
            model_config["weights_file"]  # type: ignore[index]
        )
    if "normalization" in model_config:  # type: ignore[operator]
        metadata["normalization"] = str(  # type: ignore[index]
            model_config["normalization"]  # type: ignore[index]
        )
    if metadata["aggregation"] == "transformer-target-attention":
        metadata["transformer_layers"] = int(  # type: ignore[index]
            model_config["transformer_layers"]
        )
        metadata["transformer_heads"] = int(  # type: ignore[index]
            model_config["transformer_heads"]
        )
    return metadata


def build_configured_model(
    config: dict[str, object], device: torch.device
) -> tuple[nn.Module, Path, int, int]:
    """Build the configured architecture from the mounted base model."""
    model_config = config["model"]
    data_config = config["data"]
    image_size = int(data_config["image_size"])  # type: ignore[index]
    physical_fov_mm = float(data_config["physical_fov_mm"])  # type: ignore[index]
    if not math.isfinite(physical_fov_mm) or physical_fov_mm <= 0:
        raise ValueError("Physical FOV must be finite and positive")
    model_name = str(model_config["name"])  # type: ignore[index]
    if model_name not in {"dinov2-small", "coatnet-0", "radimagenet-resnet50"}:
        model_name = model_config["name"]  # type: ignore[index]
        raise ValueError(f"Unsupported model: {model_name}")
    if model_name == "dinov2-small" and image_size % 14:
        raise ValueError("DINOv2 image size must be divisible by its patch size 14")
    if model_name in {"coatnet-0", "radimagenet-resnet50"} and image_size != 224:
        raise ValueError(f"{model_name} requires a 224-pixel input")
    aggregation = str(model_config["aggregation"])  # type: ignore[index]
    supported_aggregations = {
        "hierarchical-target-series-attention",
        "transformer-target-attention",
    }
    if aggregation not in supported_aggregations:
        raise ValueError(f"Unsupported aggregation: {aggregation}")
    transformer_layers = int(model_config.get("transformer_layers", 2))  # type: ignore[union-attr]
    transformer_heads = int(model_config.get("transformer_heads", 8))  # type: ignore[union-attr]
    if aggregation == "transformer-target-attention":
        if transformer_layers <= 0 or transformer_heads <= 0:
            raise ValueError("Transformer layer and head counts must be positive")
        attention_hidden_size = int(  # type: ignore[index]
            model_config["attention_hidden_size"]
        )
        if attention_hidden_size % transformer_heads:
            raise ValueError(
                "Attention hidden size must be divisible by transformer head count"
            )
    input_representation = str(
        model_config["input_representation"]  # type: ignore[index]
    )
    if input_representation != "2.5d-slice-channels":
        raise ValueError("Only 2.5D slice-channel input is supported")
    legacy = {
        "sampled_slices_per_slot",
        "training_window_stride",
        "inference_window_stride",
        "inference_window_pooling",
        "slice_band_start",
        "slice_band_end",
    }.intersection(data_config)
    if legacy:
        raise ValueError(
            f"Legacy window configuration is not supported: {sorted(legacy)}"
        )
    window_count = int(data_config["window_count"])  # type: ignore[index]
    if window_count <= 0:
        raise ValueError("Window count must be positive")
    window_size = int(data_config["window_size"])  # type: ignore[index]
    if window_size != 3:
        raise ValueError("Physical 2.5D windows require exactly three slices")
    max_series_per_slot = int(  # type: ignore[index]
        data_config["max_series_per_slot"]
    )
    if max_series_per_slot <= 0:
        raise ValueError("Maximum series per slot must be positive")
    configured_slots = data_config["slots"]  # type: ignore[index]
    slot_names = tuple(str(name) for name in configured_slots)
    if slot_names != SLOT_NAMES:
        raise ValueError(f"Configured slots must be {list(SLOT_NAMES)}")
    shared_model_arguments = {
        "input_root": config["paths"]["input_root"],  # type: ignore[index]
        "num_targets": len(TARGETS),
        "unfreeze_last": int(model_config["unfreeze_last"]),  # type: ignore[index]
        "dropout": float(model_config["dropout"]),  # type: ignore[index]
        "num_slots": len(slot_names),
        "max_series_per_slot": max_series_per_slot,
        "window_count": window_count,
        "slices_per_slot": window_size,
        "attention_hidden_size": int(  # type: ignore[index]
            model_config["attention_hidden_size"]
        ),
        "aggregation": aggregation,
        "transformer_layers": transformer_layers,
        "transformer_heads": transformer_heads,
        "gradient_checkpointing": bool(  # type: ignore[index]
            model_config["gradient_checkpointing"]
        ),
    }
    pooling = str(model_config["pooling"])  # type: ignore[index]
    if model_name == "dinov2-small":
        if pooling != "cls-patch-mean":
            raise ValueError("DINOv2 requires cls-patch-mean pooling")
        model, model_source = build_dinov2_model(
            variant=str(model_config["variant"]),  # type: ignore[index]
            **shared_model_arguments,
        )
    elif model_name == "coatnet-0":
        if pooling != "global-average":
            raise ValueError("CoAtNet-0 requires global-average pooling")
        weights_file = str(model_config.get("weights_file", ""))  # type: ignore[union-attr]
        if not weights_file:
            raise ValueError("CoAtNet-0 requires model.weights_file")
        model, model_source = build_coatnet_model(
            architecture=str(model_config["variant"]),  # type: ignore[index]
            normalization=str(model_config.get("normalization", "imagenet")),
            weights_file=weights_file,
            **shared_model_arguments,
        )
    else:
        if pooling != "global-average":
            raise ValueError("RadImageNet ResNet-50 requires global-average pooling")
        if bool(model_config["gradient_checkpointing"]):  # type: ignore[index]
            raise ValueError(
                "RadImageNet ResNet-50 does not support gradient checkpointing"
            )
        normalization = str(
            model_config.get("normalization", "")  # type: ignore[union-attr]
        )
        if normalization != "minus-one-to-one":
            raise ValueError(
                "RadImageNet ResNet-50 requires minus-one-to-one normalization"
            )
        weights_file = str(
            model_config.get("weights_file", "")  # type: ignore[union-attr]
        )
        if not weights_file:
            raise ValueError("RadImageNet ResNet-50 requires model.weights_file")
        model, model_source = build_radimagenet_resnet50_model(
            weights_file=weights_file,
            **shared_model_arguments,
        )
    model = model.to(device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if int(os.environ.get("RANK", "0")) == 0:
        model_name = model_config["name"]  # type: ignore[index]
        print(
            f"Loaded {model_name} from {model_source}; "
            f"parameters={total_parameters}, trainable={trainable_parameters}",
            flush=True,
        )
    return model, model_source, total_parameters, trainable_parameters


def write_json(path: Path, value: object, allow_nan: bool = False) -> None:
    """Write one consistently formatted JSON artifact."""
    path.write_text(
        json.dumps(value, indent=2, allow_nan=allow_nan) + "\n",
        encoding="utf-8",
    )


def open_verified_image_cache(
    manifest_paths: Sequence[Path],
    expected_study_uids: Sequence[str],
    expected_manifest_sha256: Sequence[str],
    image_cache_config: dict[str, object],
) -> tuple[ImageCacheReader, dict[str, int]]:
    """Open and validate the persistent image cache used by a training run."""
    image_cache = ImageCacheReader(
        manifest_paths,
        expected_study_uids=expected_study_uids,
        expected_manifest_sha256=expected_manifest_sha256,
        verify_files=True,
    )
    cache_summary = {
        key: sum(int(summary[key]) for summary in image_cache.summaries)
        for key in (
            "studies",
            "series",
            "images",
            "decoded_images",
            "failed_images",
            "archive_bytes",
        )
    }
    expected_cache = {
        "studies": int(image_cache_config["expected_studies"]),
        "series": int(image_cache_config["expected_series"]),
        "images": int(image_cache_config["expected_images"]),
        "failed_images": int(image_cache_config["expected_failed_images"]),
    }
    for key, expected in expected_cache.items():
        if cache_summary[key] != expected:
            raise ValueError(
                f"Persistent image-cache {key} changed: "
                f"{cache_summary[key]} != {expected}"
            )
    return image_cache, cache_summary


def supervision_arrays(
    table: pd.DataFrame,
    public_rows: dict[str, tuple[float, ...]],
    gold_loss_weight: float,
    use_gold: bool,
    confidence_weight_method: str = "legacy-distance",
    explicit_weights: dict[str, tuple[float, ...]] | None = None,
    explicit_masks: dict[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build aligned targets and cell weights for one CV role."""
    labels = []
    weights = []
    for row in table.to_dict(orient="records"):
        uid = str(row["StudyInstanceUID"])
        is_gold = all(pd.notna(row[target]) for target in TARGETS)
        if use_gold and is_gold:
            labels.append([float(row[target]) for target in TARGETS])
            weights.append([gold_loss_weight] * len(TARGETS))
        else:
            values = [float(value) for value in public_rows[uid]]
            labels.append(values)
            if explicit_weights is not None and explicit_masks is not None:
                weights.append([
                    value * explicit_masks[uid] for value in explicit_weights[uid]
                ])
            else:
                weights.append([
                    confidence_weight(value, confidence_weight_method) for value in values
                ])
    return np.asarray(labels, dtype=np.float32), np.asarray(weights, dtype=np.float32)


def _records_for(
    table: pd.DataFrame, records_by_uid: dict[str, StudyRecord]
) -> list[StudyRecord]:
    return [records_by_uid[str(uid)] for uid in table["StudyInstanceUID"]]


def train_cv_fold(
    config: dict[str, object],
    fold: int,
    started_at: float,
    output_dir: Path,
    table_by_uid: pd.DataFrame,
    roles: pd.DataFrame,
    public_rows: dict[str, tuple[float, ...]],
    records_by_uid: dict[str, StudyRecord],
    device: torch.device,
    distributed: DistributedContext,
    image_cache: ImageCacheReader | None = None,
    explicit_weights: dict[str, tuple[float, ...]] | None = None,
    explicit_masks: dict[str, int] | None = None,
) -> tuple[dict[str, object] | None, pd.DataFrame | None, bool]:
    """Train one outer fold using only its weak inner validation for selection."""
    fold_started_at = time.monotonic()
    run_seed = int(config["run"]["seed"]) + fold  # type: ignore[index]
    seed_everything(run_seed + distributed.rank)
    cv_config = config["cross_validation"]
    training_config = config["training"]
    data_config = config["data"]
    if not isinstance(cv_config, dict):
        raise TypeError("cross_validation configuration must be a table")
    gold_loss_weight = float(cv_config["gold_loss_weight"])
    confidence_weight_method = str(
        config["pseudo_labels"].get("confidence_weight_method", "legacy-distance")
    )

    indexed_roles = roles.set_index("StudyInstanceUID")
    train_uids = indexed_roles.index[indexed_roles["role"] == "training"]
    inner_uids = indexed_roles.index[indexed_roles["role"] == "inner-validation"]
    outer_uids = indexed_roles.index[
        (indexed_roles["role"] == "outer-holdout") & indexed_roles["is_gold"]
    ]
    train_table = table_by_uid.loc[list(train_uids)].reset_index(drop=True)
    inner_table = table_by_uid.loc[list(inner_uids)].reset_index(drop=True)
    outer_table = table_by_uid.loc[list(outer_uids)].reset_index(drop=True)
    train_labels, train_weights = supervision_arrays(
        train_table, public_rows, gold_loss_weight, True, confidence_weight_method,
        explicit_weights, explicit_masks,
    )
    inner_labels, inner_weights = supervision_arrays(
        inner_table, public_rows, gold_loss_weight, False, confidence_weight_method,
        explicit_weights, explicit_masks,
    )
    if explicit_weights is not None:
        train_keep = (train_weights > 0).any(axis=1)
        inner_keep = (inner_weights > 0).any(axis=1)
        train_table = train_table.loc[train_keep].reset_index(drop=True)
        inner_table = inner_table.loc[inner_keep].reset_index(drop=True)
        train_labels, train_weights = train_labels[train_keep], train_weights[train_keep]
        inner_labels, inner_weights = inner_labels[inner_keep], inner_weights[inner_keep]
        if train_table.empty or inner_table.empty:
            raise ValueError("Explicit supervision removed every training or validation study")
    outer_labels = outer_table[TARGETS].to_numpy(dtype=np.float32)
    outer_weights = np.ones_like(outer_labels, dtype=np.float32)

    image_size = int(data_config["image_size"])
    physical_fov_mm = float(data_config["physical_fov_mm"])
    augmentation = AugmentationConfig.from_mapping(config["augmentation"])
    batch_size = int(training_config["batch_size_per_device"])
    accumulation_steps = int(training_config["gradient_accumulation_steps"])
    num_workers = int(training_config["num_workers"])
    cache_dir = None if image_cache is not None else data_config.get("dicom_cache_dir")
    training_sampler = DistributedSampler(
        range(len(train_table)),
        num_replicas=distributed.world_size,
        rank=distributed.rank,
        shuffle=True,
        seed=run_seed,
        drop_last=False,
    )
    inner_sampler = DistributedEvaluationSampler(
        len(inner_table), distributed.rank, distributed.world_size
    )
    outer_sampler = DistributedEvaluationSampler(
        len(outer_table), distributed.rank, distributed.world_size
    )
    training_loader = make_loader(
        _records_for(train_table, records_by_uid),
        image_size,
        physical_fov_mm,
        train_labels,
        train_weights,
        batch_size,
        num_workers,
        True,
        cache_dir,
        training_sampler,
        image_cache,
    )
    inner_loader = make_loader(
        _records_for(inner_table, records_by_uid),
        image_size,
        physical_fov_mm,
        inner_labels,
        inner_weights,
        batch_size,
        num_workers,
        False,
        cache_dir,
        inner_sampler,
        image_cache,
    )
    outer_loader = make_loader(
        _records_for(outer_table, records_by_uid),
        image_size,
        physical_fov_mm,
        outer_labels,
        outer_weights,
        batch_size,
        num_workers,
        False,
        cache_dir,
        outer_sampler,
        image_cache,
    )
    if distributed.primary:
        print(
            f"Fold {fold}: train={len(train_table)}, inner={len(inner_table)}, "
            f"outer_gold={len(outer_table)}",
            flush=True,
        )
    model, model_source, total_parameters, trainable_parameters = (
        build_configured_model(config, device)
    )
    if distributed.enabled:
        model = DistributedDataParallel(
            model,
            device_ids=[distributed.local_rank],
            output_device=distributed.local_rank,
            broadcast_buffers=True,
            find_unused_parameters=False,
        )
    raw_model = unwrap_model(model)
    backbone_learning_rate = float(training_config["backbone_learning_rate"])
    head_learning_rate = float(training_config["head_learning_rate"])
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    parameter
                    for parameter in raw_model.backbone.parameters()
                    if parameter.requires_grad
                ],
                "lr": backbone_learning_rate,
            },
            {"params": raw_model.head.parameters(), "lr": head_learning_rate},
        ],
        weight_decay=float(training_config["weight_decay"]),
    )
    epochs = int(training_config["epochs"])
    scheduler = configured_scheduler(
        optimizer, training_config,
        math.ceil(len(training_loader) / accumulation_steps),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    swa = configured_swa(raw_model, training_config)
    ema = configured_ema(raw_model, training_config)
    early_stopping = EarlyStopping.from_config(training_config)
    target_weights = positive_weights(train_labels, train_weights).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    max_runtime_seconds = float(training_config["max_runtime_hours"]) * 3600.0
    best_score = -math.inf
    best_epoch = 0
    best_swa_snapshots = 0
    best_state: dict[str, torch.Tensor] | None = None
    history = []
    time_limit_reached = False
    selection_targets = np.where(inner_weights > 0, inner_labels, np.nan)
    best_inner_predictions: np.ndarray | None = None
    for epoch in range(epochs):
        epoch_started_at = time.monotonic()
        epoch_scheduler = (
            swa.begin_epoch(epoch + 1, optimizer, scheduler)
            if swa is not None else scheduler
        )
        training_sampler.set_epoch(epoch)
        loss = train_epoch(
            model,
            training_loader,
            optimizer,
            epoch_scheduler,
            scaler,
            device,
            target_weights,
            augmentation,
            accumulation_steps,
            distributed,
            ema,
            pairwise_ranking_loss_weight=float(
                training_config.get("pairwise_ranking_loss_weight", 0.0)
            ),
        )
        evaluation_model = (
            swa.end_epoch(epoch + 1, model) if swa is not None
            else ema.model if ema is not None else model
        )
        swa_snapshots = int(swa.average.n_averaged.item()) if swa is not None else 0
        if distributed.enabled:
            inner_predictions, _, _, _, _ = predict_distributed(
                evaluation_model,
                inner_loader,
                inner_sampler,
                len(inner_table),
                device,
                distributed,
            )
        else:
            inner_predictions, _, _, _, _ = predict(
                evaluation_model, inner_loader, device
            )
        if distributed.primary:
            if inner_predictions is None:
                raise RuntimeError("Primary rank did not receive inner predictions")
            per_target_auc = target_auc_scores(
                selection_targets, inner_predictions
            )
            if any(not math.isfinite(value) for value in per_target_auc.values()):
                raise RuntimeError(
                    f"Fold {fold} inner validation has a non-finite AUC"
                )
            auc = float(np.mean(list(per_target_auc.values())))
            weighted_bce = weighted_binary_cross_entropy(
                inner_labels, inner_predictions, inner_weights
            )
            history.append(
                {
                    "epoch": epoch + 1,
                    "swa_snapshots": swa_snapshots,
                    "loss": loss,
                    "inner_validation_auc": auc,
                    "inner_validation_auc_by_target": per_target_auc,
                    "inner_validation_weighted_bce": weighted_bce,
                    "elapsed_seconds": time.monotonic() - epoch_started_at,
                }
            )
            print(
                f"Fold {fold} epoch {epoch + 1}: loss={loss:.5f}, "
                f"inner_auc={auc:.5f}, inner_bce={weighted_bce:.5f}",
                flush=True,
            )
            if early_stopping.update(auc):
                best_score = auc
                best_epoch = epoch + 1
                best_swa_snapshots = swa_snapshots
                best_state = checkpoint_state_dict(evaluation_model)
                best_inner_predictions = inner_predictions.copy()
            time_limit_reached = (
                time.monotonic() - started_at >= max_runtime_seconds
            )
            if time_limit_reached:
                print(f"Global runtime limit reached in fold {fold}", flush=True)
            if early_stopping.stopped:
                print(f"Fold {fold}: early stopping after epoch {epoch + 1}; "
                      f"best epoch={best_epoch}", flush=True)
        if distributed.enabled:
            stop = torch.tensor(
                [int(time_limit_reached), int(early_stopping.stopped)],
                dtype=torch.int32, device=device
            )
            dist.broadcast(stop, src=0)
            time_limit_reached = bool(stop[0].item())
            early_stopping.stopped = bool(stop[1].item())
        if time_limit_reached or early_stopping.stopped:
            break
    if distributed.primary:
        if best_state is None:
            raise RuntimeError(f"Fold {fold} did not produce a model state")
        raw_model.load_state_dict(best_state)
    if distributed.enabled:
        for tensor in raw_model.state_dict().values():
            dist.broadcast(tensor, src=0)
        (
            outer_predictions,
            predicted_outer_labels,
            outer_window_attention,
            outer_series_attention,
            outer_slot_attention,
        ) = predict_distributed(
            model,
            outer_loader,
            outer_sampler,
            len(outer_table),
            device,
            distributed,
        )
    else:
        (
            outer_predictions,
            predicted_outer_labels,
            outer_window_attention,
            outer_series_attention,
            outer_slot_attention,
        ) = predict(model, outer_loader, device)
    fold_summary: dict[str, object] | None = None
    oof: pd.DataFrame | None = None
    if distributed.primary:
        if (
            outer_predictions is None
            or predicted_outer_labels is None
            or outer_window_attention is None
            or outer_series_attention is None
            or outer_slot_attention is None
        ):
            raise RuntimeError("Primary rank did not receive outer predictions")
        if not np.array_equal(predicted_outer_labels, outer_labels):
            raise RuntimeError(f"Fold {fold} outer labels changed during prediction")
        fold_dir = output_dir / f"fold-{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        if best_inner_predictions is None:
            raise RuntimeError(f"Fold {fold} did not retain inner predictions")
        inner_output = pd.DataFrame(
            {"StudyInstanceUID": inner_table["StudyInstanceUID"].astype(str)}
        )
        for target_index, target in enumerate(TARGETS):
            inner_output[f"label_{target}"] = inner_labels[:, target_index]
            inner_output[f"weight_{target}"] = inner_weights[:, target_index]
            inner_output[f"prediction_{target}"] = best_inner_predictions[
                :, target_index
            ]
        inner_output.to_csv(fold_dir / "inner-predictions.csv", index=False)
        checkpoint_path = fold_dir / "best-model.pt"
        torch.save(best_state, checkpoint_path)
        np.savez_compressed(
            fold_dir / "attention-weights.npz",
            study_uids=outer_table["StudyInstanceUID"].astype(str).to_numpy(),
            window_attention=outer_window_attention,
            series_attention=outer_series_attention,
            slot_attention=outer_slot_attention,
        )
        artifact_metadata = model_artifact_metadata(config)
        manifest = write_checkpoint_manifest(
            checkpoint_path,
            fold_dir / "checkpoint-manifest.json",
            {
                "model": artifact_metadata,
                "fold": fold,
                "seed": run_seed,
                "best_epoch": best_epoch,
                "best_inner_validation_auc": best_score,
                "training_studies": len(train_table),
                "inner_validation_studies": len(inner_table),
                "outer_gold_studies": len(outer_table),
                "attention_weights": "attention-weights.npz",
                "augmentation": augmentation.metadata(),
                "ema_decay": ema.decay if ema is not None else None,
                "swa": swa.metadata(best_swa_snapshots) if swa is not None else None,
            },
        )
        fold_summary = {
            "fold": fold,
            "seed": run_seed,
            "training_studies": len(train_table),
            "training_gold_studies": int(
                train_table[TARGETS].notna().all(axis=1).sum()
            ),
            "inner_validation_studies": len(inner_table),
            "outer_gold_studies": len(outer_table),
            "best_epoch": best_epoch,
            "best_inner_validation_auc": best_score,
            "completed_epochs": len(history),
            "early_stopping": early_stopping.metadata(),
            "inner_predictions": "inner-predictions.csv",
            "time_limit_reached": time_limit_reached,
            "elapsed_seconds": time.monotonic() - fold_started_at,
            "peak_gpu_memory_bytes": (
                torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None
            ),
            "history": history,
            "checkpoint": manifest,
            "model_source": str(model_source),
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "attention_weights": "attention-weights.npz",
            "augmentation": augmentation.metadata(),
            "ema_decay": ema.decay if ema is not None else None,
            "swa": swa.metadata(best_swa_snapshots) if swa is not None else None,
            "distributed": {
                "backend": distributed.backend,
                "world_size": distributed.world_size,
                "batch_size_per_device": batch_size,
                "gradient_accumulation_steps": accumulation_steps,
                "microbatch_global_size": batch_size * distributed.world_size,
                "effective_batch_size": (
                    batch_size * distributed.world_size * accumulation_steps
                ),
            },
        }
        write_json(fold_dir / "run-summary.json", fold_summary, allow_nan=False)
        oof = pd.DataFrame(
            {
                "StudyInstanceUID": outer_table["StudyInstanceUID"].astype(str),
                "outer_fold": fold,
            }
        )
        for target_index, target in enumerate(TARGETS):
            oof[f"label_{target}"] = outer_labels[:, target_index]
            oof[f"prediction_{target}"] = outer_predictions[:, target_index]
    if distributed.enabled:
        dist.barrier()
    del raw_model
    del model, optimizer, scheduler, scaler, swa, ema
    del training_loader, inner_loader, outer_loader
    gc.collect()
    torch.cuda.empty_cache()
    return fold_summary, oof, time_limit_reached


def write_proxy_training_artifacts(
    config: dict[str, object],
    output_dir: Path,
    oof: pd.DataFrame,
    fold_summaries: list[dict[str, object]],
    audit: dict[str, object],
    label_validation: dict[str, object],
    elapsed_seconds: float,
) -> None:
    """Write explicitly partial results without an inference ensemble manifest."""
    labels = oof[[f"label_{target}" for target in TARGETS]].to_numpy()
    predictions = oof[[f"prediction_{target}" for target in TARGETS]].to_numpy()
    completed = all(
        (int(summary["completed_epochs"]) == int(config["training"]["epochs"])
         or summary.get("early_stopping", {}).get("stopped", False))
        and not summary.get("time_limit_reached", False)
        for summary in fold_summaries
    )
    summary = {
        "status": (
            "proxy-training-complete" if completed else "proxy-training-incomplete"
        ),
        "full_oof": False,
        "configured_folds": int(config["cross_validation"]["folds"]),
        "trained_folds": [int(item["fold"]) for item in fold_summaries],
        "gold_studies": len(oof),
        "partial_oof_macro_auc": macro_auc(labels, predictions),
        "partial_oof_auc_by_target": target_auc_scores(labels, predictions),
        "fold_assignment_sha256": audit["assignment_sha256"],
        "inner_assignment_sha256": audit["inner_assignment_sha256"],
        "elapsed_seconds": elapsed_seconds,
        "folds": fold_summaries,
        "config": config,
        "label_source_validation": label_validation,
        "partial_oof_predictions": "partial-oof-predictions.csv",
    }
    oof.to_csv(output_dir / "partial-oof-predictions.csv", index=False)
    write_json(output_dir / "proxy-summary.json", summary, allow_nan=False)
    write_json(output_dir / "run-summary.json", summary, allow_nan=False)


def run_cross_validation_training(
    config: dict[str, object],
    output_dir: Path,
    started_at: float,
    distributed: DistributedContext,
) -> int:
    """Train selected grouped folds, retaining the full split design."""
    run_folds = selected_training_folds(config["cross_validation"])
    augmentation = AugmentationConfig.from_mapping(config["augmentation"])
    input_path = competition_input(
        config["paths"]["input_root"],  # type: ignore[index]
        config["competition"]["input_slug"],  # type: ignore[index]
    )
    train_table = pd.read_csv(input_path / "train.csv")
    train_series = pd.read_csv(input_path / "train_series.csv")
    missing_targets = [target for target in TARGETS if target not in train_table]
    if missing_targets or "Report" not in train_table:
        raise ValueError(
            f"Training table schema is incomplete; targets={missing_targets}, "
            f"report={'Report' in train_table}"
        )
    train_table["StudyInstanceUID"] = train_table["StudyInstanceUID"].astype(str)
    pseudo_config = config["pseudo_labels"]
    cv_config = config["cross_validation"]
    if not isinstance(pseudo_config, dict) or not isinstance(cv_config, dict):
        raise TypeError("Pseudo-label and CV configurations must be tables")
    public = load_public_label_table(
        locate_pseudo_labels(config),
        str(pseudo_config["sha256"]),
        int(pseudo_config["expected_rows"]),
    )
    use_explicit = bool(pseudo_config.get("use_explicit_supervision", False))
    explicit_weights = getattr(public, "confidence_weights", None) if use_explicit else None
    explicit_masks = getattr(public, "masks", None) if use_explicit else None
    if use_explicit and (explicit_weights is None or explicit_masks is None):
        raise ValueError("Explicit supervision requires CSV confidence weights and masks")
    gold = train_table.loc[train_table[TARGETS].notna().all(axis=1)]
    label_validation = source_validation(
        gold[["StudyInstanceUID", *TARGETS]].to_dict(orient="records"), public.rows
    )
    expected_gold = int(pseudo_config["expected_validation_studies"])
    minimum_auc = float(pseudo_config["minimum_source_macro_auc"])
    gate = {
        "all_gold_studies_matched": label_validation["matched_gold_studies"]
        == expected_gold,
        "all_targets_scored": label_validation["scored_targets"] == len(TARGETS),
        "macro_auc_at_least_minimum": label_validation["macro_auc"] >= minimum_auc,
    }
    gate["passed"] = all(gate.values())
    label_validation["minimum_macro_auc"] = minimum_auc
    label_validation["acceptance_gate"] = gate
    if distributed.primary:
        write_json(
            output_dir / "label-source-validation.json", label_validation, True
        )
    if not gate["passed"]:
        raise ValueError("Public label source failed its acceptance gate")

    design = create_outer_folds(
        train_table,
        public.rows,
        int(cv_config["folds"]),
        int(cv_config["outer_seed"]),
        str(cv_config["split_algorithm"]),
    )
    if distributed.primary:
        design.assignments.to_csv(
            output_dir / "fold-assignments.csv", index=False
        )
    table_with_hash = train_table.copy()
    table_with_hash["report_hash"] = table_with_hash["Report"].fillna("").map(
        report_hash
    )
    table_by_uid = table_with_hash.set_index("StudyInstanceUID", drop=False)
    all_roles = []
    for fold in range(int(cv_config["folds"])):
        all_roles.append(
            create_inner_roles(
                design.assignments,
                public.rows,
                fold,
                float(cv_config["inner_validation_fraction"]),
                int(cv_config["inner_seed"]),
            )
        )
    inner_splits = pd.concat(all_roles, ignore_index=True)
    if distributed.primary:
        inner_splits.to_csv(output_dir / "inner-splits.csv", index=False)
    design.audit["inner_seed"] = int(cv_config["inner_seed"])
    design.audit["inner_validation_fraction"] = float(
        cv_config["inner_validation_fraction"]
    )
    design.audit["inner_assignment_sha256"] = dataframe_sha256(inner_splits)
    design.audit["inner_splits"] = {
        str(fold): {
            role: int(count)
            for role, count in fold_rows["role"].value_counts().sort_index().items()
        }
        for fold, fold_rows in inner_splits.groupby("model_fold")
    }
    if distributed.primary:
        write_json(output_dir / "fold-audit.json", design.audit, allow_nan=False)

    data_config = config["data"]
    all_uids = design.assignments["StudyInstanceUID"].astype(str).tolist()
    run_id = os.environ.get("TORCHELASTIC_RUN_ID", "local").replace("/", "-")
    records_cache = Path(
        f"/tmp/rsna2026-distributed-study-records-{run_id}.pkl"
    )
    image_cache: ImageCacheReader | None = None
    cache_summary: dict[str, int] | None = None
    image_cache_config = config.get("image_cache", {})
    persistent_cache_enabled = (
        isinstance(image_cache_config, dict)
        and bool(image_cache_config.get("enabled", False))
    )
    manifest_paths: list[Path] = []
    manifest_sha256: list[str] = []
    if persistent_cache_enabled:
        manifest_paths = [
            resolve_input_file(path, config["paths"]["input_root"])  # type: ignore[index]
            for path in image_cache_config["manifest_paths"]
        ]
        manifest_sha256 = [
            str(value) for value in image_cache_config["manifest_sha256"]
        ]
    if distributed.primary:
        configured_cache = str(data_config.get("study_records_cache_path", ""))
        configured_sha256 = str(data_config.get("study_records_cache_sha256", ""))
        if persistent_cache_enabled:
            image_cache, cache_summary = open_verified_image_cache(
                manifest_paths,
                all_uids,
                manifest_sha256,
                image_cache_config,
            )
            write_json(
                output_dir / "image-cache-validation.json",
                {
                    "manifest_files": [str(path) for path in manifest_paths],
                    "manifest_sha256": manifest_sha256,
                    "summary": cache_summary,
                },
                allow_nan=False,
            )
            print(f"Verified persistent image cache: {cache_summary}", flush=True)
        if configured_cache and configured_sha256:
            selection_cache_path = resolve_input_file(
                configured_cache,
                config["paths"]["input_root"],  # type: ignore[index]
            )
            all_records, selection_summary = load_study_records_cache(
                selection_cache_path,
                input_path / "train_series",
                all_uids,
                int(data_config["window_count"]),
                int(data_config["max_series_per_slot"]),
                list(data_config["slots"]),
                str(data_config["series_classification"]),
                configured_sha256,
                str(data_config["study_records_cache_uids_sha256"]),
            )
            if cache_summary is not None:
                # The selection artifact can retain statistics from an older
                # image cache; report the cache actually validated for this run.
                selection_summary["persistent_image_cache"] = cache_summary
            selection_output = output_dir / "slice-selection.json"
            if selection_cache_path.suffix == ".xz":
                with lzma.open(selection_cache_path, "rb") as source:
                    with selection_output.open("wb") as destination:
                        shutil.copyfileobj(source, destination)
            else:
                shutil.copyfile(selection_cache_path, selection_output)
        elif persistent_cache_enabled:
            assert image_cache is not None
            all_records = build_cached_study_records(
                train_series,
                image_cache,
                all_uids,
                int(data_config["window_count"]),
                int(data_config["max_series_per_slot"]),
                list(data_config["slots"]),
                str(data_config["series_classification"]),
            )
            report = slice_selection_report(
                {"all-training-studies": all_records},
                int(data_config["window_count"]),
                int(data_config["max_series_per_slot"]),
            )
            selection_summary = report["splits"]["all-training-studies"][
                "summary"
            ]
            expected_missing = {
                str(key): int(value)
                for key, value in image_cache_config[
                    "expected_missing_by_slot"
                ].items()
            }
            if selection_summary["missing_slots_by_slot"] != expected_missing:
                raise ValueError(
                    "Reclassified acquisition-slot missing counts changed: "
                    f"{selection_summary['missing_slots_by_slot']}"
                )
            if int(selection_summary["selected_series"]) != int(
                image_cache_config["expected_selected_series"]
            ):
                raise ValueError("Reclassified selected-series count changed")
            if int(selection_summary["unassigned_series"]) != 0:
                raise ValueError("Reclassified series contain unassigned entries")
            selection_summary["persistent_image_cache"] = cache_summary
            selection_output = output_dir / "slice-selection.json"
            write_json(selection_output, report)
        else:
            all_records = build_study_records(
                train_series,
                input_path / "train_series",
                all_uids,
                int(data_config["window_count"]),
                int(data_config["max_series_per_slot"]),
                list(data_config["slots"]),
                str(data_config["series_classification"]),
            )
            report = slice_selection_report(
                {"all-training-studies": all_records},
                int(data_config["window_count"]),
                int(data_config["max_series_per_slot"]),
            )
            selection_output = output_dir / "slice-selection.json"
            write_json(selection_output, report)
            selection_summary = report["splits"]["all-training-studies"]["summary"]
        if distributed.enabled:
            temporary_records_cache = records_cache.with_suffix(".temporary")
            with temporary_records_cache.open("wb") as file:
                pickle.dump(all_records, file, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary_records_cache, records_cache)
        print(
            f"Prepared physical-window study records: {selection_summary}", flush=True
        )
    if not distributed.primary:
        deadline = time.monotonic() + 1800.0
        while not records_cache.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for the study-record cache")
            time.sleep(1.0)
        with records_cache.open("rb") as file:
            all_records = pickle.load(file)
    if persistent_cache_enabled and image_cache is None:
        image_cache = ImageCacheReader(
            manifest_paths,
            expected_study_uids=all_uids,
            expected_manifest_sha256=manifest_sha256,
            verify_files=False,
        )
    if len(all_records) != len(all_uids):
        raise RuntimeError("Distributed study-record cache is incomplete")
    records_by_uid = {record.study_uid: record for record in all_records}
    if distributed.enabled:
        dist.barrier()
    device = torch.device("cuda", distributed.local_rank)
    local_gpu_name = torch.cuda.get_device_name(distributed.local_rank)
    if distributed.enabled:
        gpu_names: list[str | None] = [None] * distributed.world_size
        dist.all_gather_object(gpu_names, local_gpu_name)
        print(
            f"Distributed training on {distributed.world_size} GPUs: "
            f"{gpu_names}",
            flush=True,
        )
    else:
        gpu_names = [local_gpu_name]
        print(f"Single-GPU training on {local_gpu_name}", flush=True)
    fold_summaries = []
    oof_tables = []
    fold_count = int(cv_config["folds"])
    worker_folds, fold_context, data_parallel = cv_fold_execution(
        run_folds, config["training"], distributed
    )
    print(
        f"GPU {distributed.local_rank} training folds {list(worker_folds)}",
        flush=True,
    )
    for fold in worker_folds:
        fold_roles = inner_splits.loc[inner_splits["model_fold"] == fold]
        summary, oof, time_limit = train_cv_fold(
            config,
            fold,
            started_at,
            output_dir,
            table_by_uid,
            fold_roles,
            public.rows,
            records_by_uid,
            device,
            fold_context,
            image_cache,
            explicit_weights,
            explicit_masks,
        )
        if fold_context.primary:
            if summary is None or oof is None:
                raise RuntimeError("Fold worker did not produce fold artifacts")
            fold_summaries.append(summary)
            oof_tables.append(oof)
        if time_limit and fold != worker_folds[-1]:
            raise RuntimeError("Runtime limit prevented all requested folds")

    if data_parallel and not distributed.primary:
        return 0
    if distributed.enabled and not data_parallel:
        gathered_results: list[
            tuple[list[dict[str, object]], list[pd.DataFrame]] | None
        ] = [None] * distributed.world_size
        dist.all_gather_object(gathered_results, (fold_summaries, oof_tables))
        if not distributed.primary:
            return 0
        fold_summaries = [
            summary
            for result in gathered_results
            if result is not None
            for summary in result[0]
        ]
        oof_tables = [
            table
            for result in gathered_results
            if result is not None
            for table in result[1]
        ]
    fold_summaries = sorted(
        fold_summaries, key=lambda summary: int(summary["fold"])
    )
    if [int(summary["fold"]) for summary in fold_summaries] != list(run_folds):
        raise RuntimeError("Fold workers did not produce every requested fold")
    oof = pd.concat(oof_tables, ignore_index=True).sort_values(
        "StudyInstanceUID"
    ).reset_index(drop=True)
    expected_oof_uids = set(
        design.assignments.loc[
            design.assignments["is_gold"]
            & design.assignments["outer_fold"].isin(run_folds),
            "StudyInstanceUID",
        ].astype(str)
    )
    if (
        set(oof["StudyInstanceUID"].astype(str)) != expected_oof_uids
        or not oof["StudyInstanceUID"].is_unique
    ):
        raise RuntimeError("OOF predictions do not cover the requested gold studies")
    if len(run_folds) < fold_count:
        write_proxy_training_artifacts(
            config,
            output_dir,
            oof,
            fold_summaries,
            design.audit,
            label_validation,
            time.monotonic() - started_at,
        )
        print(f"Completed proxy training for folds {list(run_folds)}", flush=True)
        return 0
    if len(oof) != expected_gold or not oof["StudyInstanceUID"].is_unique:
        raise RuntimeError("OOF predictions do not cover every gold study exactly once")
    oof.to_csv(output_dir / "oof-predictions.csv", index=False)
    labels = oof[[f"label_{target}" for target in TARGETS]].to_numpy()
    predictions = oof[[f"prediction_{target}" for target in TARGETS]].to_numpy()
    per_target_auc = target_auc_scores(labels, predictions)
    oof_auc = macro_auc(labels, predictions)
    interval = bootstrap_macro_auc_interval(
        labels,
        predictions,
        macro_auc,
        int(cv_config["bootstrap_resamples"]),
        int(cv_config["outer_seed"]),
    )
    cv_summary = {
        "gold_studies": len(oof),
        "oof_macro_auc": oof_auc,
        "oof_auc_by_target": per_target_auc,
        "bootstrap_macro_auc": interval,
        "fold_assignment_sha256": design.audit["assignment_sha256"],
        "inner_assignment_sha256": design.audit["inner_assignment_sha256"],
    }
    write_json(output_dir / "cv-summary.json", cv_summary, allow_nan=False)
    artifact_metadata = model_artifact_metadata(config)
    members = []
    for summary in fold_summaries:
        fold = int(summary["fold"])
        checkpoint = output_dir / f"fold-{fold}" / "best-model.pt"
        members.append(
            {
                "fold": fold,
                "checkpoint_file": f"fold-{fold}/best-model.pt",
                "checkpoint_sha256": file_sha256(checkpoint),
                "checkpoint_manifest_file": f"fold-{fold}/checkpoint-manifest.json",
                "best_epoch": summary["best_epoch"],
                "best_inner_validation_auc": summary["best_inner_validation_auc"],
            }
        )
    ensemble_manifest = {
        "format_version": ENSEMBLE_FORMAT_VERSION,
        "model": artifact_metadata,
        "members": members,
        "fold_assignment_sha256": design.audit["assignment_sha256"],
        "inner_assignment_sha256": design.audit["inner_assignment_sha256"],
        "test_ensemble": str(cv_config["test_ensemble"]),
        "within_model_pooling": str(config["model"]["aggregation"]),  # type: ignore[index]
        "training_slice_selection_sha256": file_sha256(
            output_dir / "slice-selection.json"
        ),
    }
    write_json(output_dir / "ensemble-manifest.json", ensemble_manifest)
    if persistent_cache_enabled and cache_summary is None:
        raise RuntimeError("Persistent image-cache summary was not initialized")
    elapsed = time.monotonic() - started_at
    run_summary = {
        "status": "cross-validation-training-complete",
        "device": str(device),
        "distributed": {
            "enabled": distributed.enabled,
            "backend": distributed.backend,
            "world_size": distributed.world_size,
            "strategy": (
                "data-parallel" if data_parallel else (
                    "one-fold-per-gpu" if distributed.enabled else "single-gpu-sequential"
                )
            ),
            "simultaneous_folds": (
                distributed.world_size if distributed.enabled and not data_parallel else 1
            ),
            "gpu_names": gpu_names,
            "batch_size_per_device": int(
                config["training"]["batch_size_per_device"]  # type: ignore[index]
            ),
            "microbatch_global_size": int(
                config["training"]["batch_size_per_device"]  # type: ignore[index]
            ) * (distributed.world_size if data_parallel else 1),
            "gradient_accumulation_steps": int(
                config["training"]["gradient_accumulation_steps"]  # type: ignore[index]
            ),
            "effective_batch_size": int(
                config["training"]["batch_size_per_device"]  # type: ignore[index]
            )
            * (distributed.world_size if data_parallel else 1)
            * int(
                config["training"]["gradient_accumulation_steps"]  # type: ignore[index]
            ),
        },
        "optimization": {
            "pairwise_ranking_loss_weight": float(
                config["training"].get("pairwise_ranking_loss_weight", 0.0)
            ),
            "epochs": int(config["training"]["epochs"]),  # type: ignore[index]
            "backbone_learning_rate": float(
                config["training"]["backbone_learning_rate"]  # type: ignore[index]
            ),
            "head_learning_rate": float(
                config["training"]["head_learning_rate"]  # type: ignore[index]
            ),
            "weight_decay": float(
                config["training"]["weight_decay"]  # type: ignore[index]
            ),
            "scheduler": (
                "one-cycle-then-swa" if config["training"].get("swa_enabled", False)
                else str(config["training"].get("scheduler", "one-cycle"))
            ),
            "early_stopping_patience": int(config["training"].get("early_stopping_patience", 0)),
            "early_stopping_min_delta": float(config["training"].get("early_stopping_min_delta", 0.0)),
            "one_cycle_warmup_fraction": float(
                config["training"]["one_cycle_warmup_fraction"]  # type: ignore[index]
            ),
            "ema_decay": (
                float(config["training"]["ema_decay"])  # type: ignore[index]
                if "ema_decay" in config["training"]  # type: ignore[operator]
                else None
            ),
        },
        "augmentation": augmentation.metadata(),
        "elapsed_seconds": elapsed,
        "folds": fold_summaries,
        "swa_configuration": {
            key: value for key, value in config["training"].items()
            if key.startswith("swa_")
        },
        "cross_validation": cv_summary,
        "ensemble_manifest": "ensemble-manifest.json",
        "fold_audit": "fold-audit.json",
        "fold_assignments": "fold-assignments.csv",
        "inner_splits": "inner-splits.csv",
        "oof_predictions": "oof-predictions.csv",
        "label_source_validation": label_validation,
        "study_records_cache": {
            "configured_sha256": str(data_config["study_records_cache_sha256"]),
            "artifact_sha256": file_sha256(output_dir / "slice-selection.json"),
            "study_uids_sha256": str(
                data_config["study_records_cache_uids_sha256"]
            ),
            "artifact": "slice-selection.json",
        },
        "persistent_image_cache": (
            {
                "enabled": True,
                "manifest_files": [str(path) for path in manifest_paths],
                "manifest_sha256": manifest_sha256,
                "summary": cache_summary,
                "selection_artifact_sha256": file_sha256(
                    output_dir / "slice-selection.json"
                ),
            }
            if persistent_cache_enabled
            else {"enabled": False}
        ),
        "pseudo_label_file": str(pseudo_config["file_name"]),
        "pseudo_labels_sha256": public.sha256,
        "model": artifact_metadata,
        "window_count": int(data_config["window_count"]),
        "gradient_accumulation_steps": int(
            config["training"]["gradient_accumulation_steps"]  # type: ignore[index]
        ),
    }
    write_json(output_dir / "run-summary.json", run_summary, allow_nan=False)
    print(
        f"Completed four-fold training: OOF macro AUC={oof_auc:.5f}, "
        f"95% interval=({interval['lower_95']:.5f}, {interval['upper_95']:.5f})",
        flush=True,
    )
    return 0


def run_inference(
    config: dict[str, object], output_dir: Path, started_at: float
) -> int:
    """Load a trained checkpoint and infer competition test studies only."""
    input_root = config["paths"]["input_root"]  # type: ignore[index]
    input_path = competition_input(
        input_root, config["competition"]["input_slug"]  # type: ignore[index]
    )
    test_table = pd.read_csv(input_path / "test.csv")
    test_series = pd.read_csv(input_path / "test_series.csv")
    sample_submission = pd.read_csv(input_path / "sample_submission.csv")
    data_config = config["data"]
    image_size = int(data_config["image_size"])  # type: ignore[index]
    physical_fov_mm = float(  # type: ignore[index]
        data_config["physical_fov_mm"]
    )
    window_count = int(data_config["window_count"])  # type: ignore[index]
    slot_names = list(data_config["slots"])  # type: ignore[index]
    classification_method = str(  # type: ignore[index]
        data_config["series_classification"]
    )
    test_uids = test_table["StudyInstanceUID"].astype(str).tolist()
    test_records = build_study_records(
        test_series,
        input_path / "test_series",
        test_uids,
        window_count,
        int(data_config["max_series_per_slot"]),
        slot_names,
        classification_method,
    )
    selection_report = slice_selection_report(
        {"test": test_records},
        window_count,
        int(data_config["max_series_per_slot"]),
    )
    write_json(output_dir / "slice-selection.json", selection_report)
    test_loader = make_loader(
        test_records,
        image_size,
        physical_fov_mm,
        None,
        None,
        int(config["training"]["inference_batch_size"]),  # type: ignore[index]
        int(config["training"]["num_workers"]),  # type: ignore[index]
        False,
        data_config.get("dicom_cache_dir"),  # type: ignore[union-attr]
    )

    inference_config = config["inference"]
    manifest_path = resolve_input_file(
        inference_config["ensemble_manifest_path"], input_root  # type: ignore[index]
    )
    manifest = load_ensemble_manifest(manifest_path)
    expected_model = model_artifact_metadata(config)
    if manifest.get("model") != expected_model:
        raise ValueError(
            "Ensemble model metadata does not match the inference configuration"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_source, total_parameters, trainable_parameters = (
        build_configured_model(config, device)
    )
    member_predictions = []
    for member in sorted(manifest["members"], key=lambda value: value["fold"]):
        checkpoint_path = manifest_path.parent / member["checkpoint_file"]
        state_dict = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        model.load_state_dict(state_dict)
        predictions, _, window_attention, series_attention, slot_attention = predict(
            model, test_loader, device
        )
        member_predictions.append(predictions)
        np.savez_compressed(
            output_dir / f"attention-weights-fold-{int(member['fold'])}.npz",
            study_uids=np.asarray(test_uids),
            window_attention=window_attention,
            series_attention=series_attention,
            slot_attention=slot_attention,
        )
    if manifest.get("test_ensemble") != "per-target-percentile-rank-mean":
        raise ValueError("Unsupported test ensemble method")
    predictions = percentile_rank_mean(member_predictions)
    submission_path = output_dir / "submission.csv"
    write_submission(sample_submission, test_uids, predictions, submission_path)
    summary = {
        "status": "inference-complete",
        "device": str(device),
        "test_studies": len(test_records),
        "targets": TARGETS,
        "elapsed_seconds": time.monotonic() - started_at,
        "ensemble_manifest_source": str(manifest_path),
        "ensemble": manifest,
        "model": {
            **expected_model,
            "source": str(model_source),
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
        },
        "attention_weights": [
            f"attention-weights-fold-{int(member['fold'])}.npz"
            for member in sorted(
                manifest["members"], key=lambda value: value["fold"]
            )
        ],
        "slice_selection": {
            "selection_method": selection_report["slice_selection_method"],
            "artifact": "slice-selection.json",
            "splits": {
                split: values["summary"]
                for split, values in selection_report["splits"].items()
            },
        },
        "dicom_cache": {
            "enabled": bool(data_config.get("dicom_cache_dir")),
            "directory": data_config.get("dicom_cache_dir"),
            "storage_dtype": "uint8",
            "preprocessing_version": PREPROCESSING_VERSION,
        },
    }
    write_json(output_dir / "run-summary.json", summary)
    print(f"Wrote {submission_path}", flush=True)
    return 0


def prepare_training_selection_cache(
    config: dict[str, object], output_dir: Path
) -> int:
    """Build the physical-window selection report before GPU workers start."""
    input_path = competition_input(
        config["paths"]["input_root"],  # type: ignore[index]
        config["competition"]["input_slug"],  # type: ignore[index]
    )
    train_table = pd.read_csv(input_path / "train.csv")
    train_series = pd.read_csv(input_path / "train_series.csv")
    study_uids = train_table["StudyInstanceUID"].astype(str).tolist()
    data_config = config["data"]
    print(
        f"Preparing physical-window selection cache for {len(study_uids)} studies",
        flush=True,
    )
    records = build_study_records(
        train_series,
        input_path / "train_series",
        study_uids,
        int(data_config["window_count"]),  # type: ignore[index]
        int(data_config["max_series_per_slot"]),  # type: ignore[index]
        list(data_config["slots"]),  # type: ignore[index]
        str(data_config["series_classification"]),  # type: ignore[index]
    )
    report = slice_selection_report(
        {"all-training-studies": records},
        int(data_config["window_count"]),  # type: ignore[index]
        int(data_config["max_series_per_slot"]),  # type: ignore[index]
    )
    selection_output = output_dir / "slice-selection.json"
    write_json(selection_output, report)
    print(
        f"Prepared physical-window selection cache: {selection_output} "
        f"sha256={file_sha256(selection_output)}",
        flush=True,
    )
    return 0


def prepare_training_image_cache(
    config: dict[str, object],
    output_dir: Path,
    shard_index: int,
    shard_count: int,
) -> int:
    """Preprocess one deterministic shard of every training series on CPU."""
    input_path = competition_input(
        config["paths"]["input_root"],  # type: ignore[index]
        config["competition"]["input_slug"],  # type: ignore[index]
    )
    train_table = pd.read_csv(input_path / "train.csv")
    train_series = pd.read_csv(input_path / "train_series.csv")
    study_uids = train_table["StudyInstanceUID"].astype(str).tolist()
    data_config = config["data"]
    image_size = int(data_config["image_size"])  # type: ignore[index]
    physical_fov_mm = float(data_config["physical_fov_mm"])  # type: ignore[index]
    window_count = int(data_config["window_count"])  # type: ignore[index]
    print(
        f"Preparing persistent image-cache shard {shard_index}/{shard_count} "
        f"for {len(study_uids)} training studies",
        flush=True,
    )
    build_image_cache_shard(
        train_series,
        input_path / "train_series",
        study_uids,
        output_dir,
        shard_index=shard_index,
        shard_count=shard_count,
        image_size=image_size,
        physical_fov_mm=physical_fov_mm,
        window_count=window_count,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    started_at = time.monotonic()
    args = parse_args(argv)
    config = load_config(args.config)
    apply_runtime_overrides(config, args)
    if args.mode == "train":
        apply_study_records_cache_environment(config)
    distributed = initialize_distributed(config, args.mode)
    seed = int(config["run"]["seed"])
    seed_everything(seed + distributed.rank)

    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "train" and distributed.primary:
        write_json(output_dir / "resolved-config.json", config, allow_nan=False)

    if args.mode == "prepare-selection-cache":
        return prepare_training_selection_cache(config, output_dir)
    if args.mode == "prepare-image-cache":
        return prepare_training_image_cache(
            config,
            output_dir,
            args.cache_shard_index,
            args.cache_shard_count,
        )
    if args.mode == "inference":
        return run_inference(config, output_dir, started_at)

    cv_config = config.get("cross_validation", {})
    if isinstance(cv_config, dict) and bool(cv_config.get("enabled", False)):
        try:
            return run_cross_validation_training(
                config, output_dir, started_at, distributed
            )
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()

    if distributed.enabled:
        raise RuntimeError(
            "Distributed training is supported only for cross-validation"
        )

    input_path = competition_input(
        config["paths"]["input_root"], config["competition"]["input_slug"]
    )
    train_table = pd.read_csv(input_path / "train.csv")
    train_series = pd.read_csv(input_path / "train_series.csv")

    missing_targets = [
        target for target in TARGETS if target not in train_table.columns
    ]
    if missing_targets:
        raise ValueError(f"Training table is missing targets: {missing_targets}")
    if "Report" not in train_table.columns:
        raise ValueError("Training table is missing Report")

    pseudo_enabled = bool(config.get("pseudo_labels", {}).get("enabled", False))
    provenance: dict[str, object] = {}
    label_source_validation: dict[str, object] | None = None
    if pseudo_enabled:
        (
            training_table,
            training_labels,
            training_weights,
            validation_table,
            validation_labels,
            validation_weights,
            provenance,
            label_source_validation,
        ) = load_pseudo_supervision(train_table, config, seed)
        (output_dir / "label-source-validation.json").write_text(
            json.dumps(label_source_validation, indent=2, allow_nan=True) + "\n",
            encoding="utf-8",
        )
        if not label_source_validation["acceptance_gate"]["passed"]:
            raise ValueError(
                "Public label source failed its acceptance gate; "
                "see label-source-validation.json"
            )
    else:
        labeled = train_table[TARGETS].notna().any(axis=1)
        supervised = train_table.loc[labeled].reset_index(drop=True)
        max_train_studies = int(config["training"].get("max_train_studies", 0))
        if max_train_studies > 0:
            supervised = supervised.sample(
                n=min(max_train_studies, len(supervised)), random_state=seed
            ).reset_index(drop=True)
        labels = supervised[TARGETS].to_numpy(dtype=np.float32)
        training_indices, validation_indices = split_indices(
            len(supervised),
            float(config["training"]["validation_fraction"]),
            seed,
        )
        training_table = supervised.iloc[training_indices].reset_index(drop=True)
        training_labels = labels[training_indices]
        training_weights = np.isfinite(training_labels).astype(np.float32)
        validation_table = supervised.iloc[validation_indices].reset_index(drop=True)
        validation_labels = labels[validation_indices]
        validation_weights = np.isfinite(validation_labels).astype(np.float32)
    print(
        f"Training studies={len(training_table)}, validation studies="
        f"{len(validation_table)}, available training labels="
        f"{int(np.isfinite(training_labels).sum())}",
        flush=True,
    )

    image_size = int(config["data"]["image_size"])
    physical_fov_mm = float(config["data"]["physical_fov_mm"])
    window_count = int(config["data"]["window_count"])
    augmentation = AugmentationConfig.from_mapping(config["augmentation"])
    slot_names = list(config["data"]["slots"])
    classification_method = str(config["data"]["series_classification"])
    training_records = build_study_records(
        train_series,
        input_path / "train_series",
        training_table["StudyInstanceUID"].astype(str).tolist(),
        window_count,
        int(config["data"]["max_series_per_slot"]),
        slot_names,
        classification_method,
    )
    validation_records = build_study_records(
        train_series,
        input_path / "train_series",
        validation_table["StudyInstanceUID"].astype(str).tolist(),
        window_count,
        int(config["data"]["max_series_per_slot"]),
        slot_names,
        classification_method,
    )
    selection_report = slice_selection_report(
        {
            "training": training_records,
            "validation": validation_records,
        },
        window_count,
        int(config["data"]["max_series_per_slot"]),
    )
    (output_dir / "slice-selection.json").write_text(
        json.dumps(selection_report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    batch_size = int(config["training"]["batch_size_per_device"])
    num_workers = int(config["training"]["num_workers"])
    dicom_cache_dir = config["data"].get("dicom_cache_dir")
    training_loader = make_loader(
        training_records,
        image_size,
        physical_fov_mm,
        training_labels,
        training_weights,
        batch_size,
        num_workers,
        True,
        dicom_cache_dir,
    )
    validation_loader = make_loader(
        validation_records,
        image_size,
        physical_fov_mm,
        validation_labels,
        validation_weights,
        batch_size,
        num_workers,
        False,
        dicom_cache_dir,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Training on {device}; train={len(training_records)}, "
        f"validation={len(validation_records)}",
        flush=True,
    )
    model, model_source, total_parameters, trainable_parameters = (
        build_configured_model(config, device)
    )
    backbone_learning_rate = float(
        config["training"]["backbone_learning_rate"]
    )
    head_learning_rate = float(config["training"]["head_learning_rate"])
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    parameter
                    for parameter in model.backbone.parameters()
                    if parameter.requires_grad
                ],
                "lr": backbone_learning_rate,
            },
            {
                "params": model.head.parameters(),
                "lr": head_learning_rate,
            },
        ],
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = int(config["training"]["epochs"])
    accumulation_steps = int(config["training"]["gradient_accumulation_steps"])
    scheduler = configured_scheduler(
        optimizer, config["training"],
        math.ceil(len(training_loader) / accumulation_steps),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    training_config = config["training"]
    if not isinstance(training_config, dict):
        raise TypeError("training configuration must be a table")
    swa = configured_swa(model, training_config)
    ema = configured_ema(model, training_config)
    early_stopping = EarlyStopping.from_config(training_config)
    target_weights = positive_weights(training_labels, training_weights).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    max_runtime_seconds = float(
        config["training"]["max_runtime_hours"]
    ) * 3600.0

    best_score = -math.inf
    best_epoch = 0
    best_swa_snapshots = 0
    best_state: dict[str, torch.Tensor] | None = None
    history = []
    time_limit_reached = False
    for epoch in range(epochs):
        epoch_started_at = time.monotonic()
        epoch_scheduler = (
            swa.begin_epoch(epoch + 1, optimizer, scheduler)
            if swa is not None else scheduler
        )
        loss = train_epoch(
            model,
            training_loader,
            optimizer,
            epoch_scheduler,
            scaler,
            device,
            target_weights,
            augmentation,
            accumulation_steps,
            ema=ema,
            pairwise_ranking_loss_weight=float(
                config["training"].get("pairwise_ranking_loss_weight", 0.0)
            ),
        )
        evaluation_model = (
            swa.end_epoch(epoch + 1, model) if swa is not None
            else ema.model if ema is not None else model
        )
        swa_snapshots = int(swa.average.n_averaged.item()) if swa is not None else 0
        (
            validation_predictions,
            validation_labels,
            validation_window_attention,
            validation_series_attention,
            validation_slot_attention,
        ) = predict(evaluation_model, validation_loader, device)
        if validation_labels is None:
            raise RuntimeError("Validation labels are unavailable")
        auc = macro_auc(validation_labels, validation_predictions)
        per_target_auc = target_auc_scores(validation_labels, validation_predictions)
        history.append(
            {
                "epoch": epoch + 1,
                "swa_snapshots": swa_snapshots,
                "loss": loss,
                "validation_auc": auc,
                "validation_auc_by_target": per_target_auc,
                "elapsed_seconds": time.monotonic() - epoch_started_at,
            }
        )
        print(
            f"Epoch {epoch + 1}: loss={loss:.5f}, validation_auc={auc:.5f}",
            flush=True,
        )
        selection_score = auc if math.isfinite(auc) else -loss
        if early_stopping.update(selection_score):
            best_score = selection_score
            best_epoch = epoch + 1
            best_swa_snapshots = swa_snapshots
            best_state = checkpoint_state_dict(evaluation_model)
        if early_stopping.stopped:
            print(f"Early stopping after epoch {epoch + 1}; best epoch={best_epoch}", flush=True)
            break
        if time.monotonic() - started_at >= max_runtime_seconds:
            time_limit_reached = True
            print(
                f"Runtime limit reached after epoch {epoch + 1}; "
                "using the best completed epoch",
                flush=True,
            )
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a model state")
    checkpoint_path = output_dir / "best-model.pt"
    torch.save(best_state, checkpoint_path)
    np.savez_compressed(
        output_dir / "attention-weights.npz",
        study_uids=validation_table["StudyInstanceUID"].astype(str).to_numpy(),
        window_attention=validation_window_attention,
        series_attention=validation_series_attention,
        slot_attention=validation_slot_attention,
    )
    artifact_metadata = model_artifact_metadata(config)
    manifest = write_checkpoint_manifest(
        checkpoint_path,
        output_dir / "checkpoint-manifest.json",
        {
            "model": artifact_metadata,
            "best_epoch": best_epoch,
            "best_validation_score": best_score,
            "training_studies": len(training_records),
            "validation_studies": len(validation_records),
            "augmentation": augmentation.metadata(),
            "ema_decay": ema.decay if ema is not None else None,
            "swa": swa.metadata(best_swa_snapshots) if swa is not None else None,
        },
    )
    elapsed_seconds = time.monotonic() - started_at
    summary = {
        "status": "training-complete",
        "device": str(device),
        "pseudo_labels_enabled": pseudo_enabled,
        "training_studies": len(training_records),
        "validation_studies": len(validation_records),
        "targets": TARGETS,
        "elapsed_seconds": elapsed_seconds,
        "peak_gpu_memory_bytes": (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None
        ),
        "completed_epochs": len(history),
        "best_epoch": best_epoch,
        "early_stopping": early_stopping.metadata(),
        "time_limit_reached": time_limit_reached,
        "checkpoint": manifest,
        "augmentation": augmentation.metadata(),
        "model": {
            **artifact_metadata,
            "source": str(model_source),
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "grayscale_to_rgb": "three-ordered-slices-as-channels",
            "within_model_pooling": str(config["model"]["aggregation"]),
            "window_pooling": (
                "per-target-global-masked-attention"
                if config["model"]["aggregation"]
                == "transformer-target-attention"
                else "per-target-masked-attention"
            ),
            "series_pooling": (
                "global-attention-marginal"
                if config["model"]["aggregation"]
                == "transformer-target-attention"
                else "per-target-masked-attention"
            ),
            "slot_pooling": (
                "global-attention-marginal"
                if config["model"]["aggregation"]
                == "transformer-target-attention"
                else "per-target-masked-attention"
            ),
            "backbone_learning_rate": backbone_learning_rate,
            "head_learning_rate": head_learning_rate,
            "weight_decay": float(config["training"]["weight_decay"]),
            "scheduler": (
                "one-cycle-then-swa" if config["training"].get("swa_enabled", False)
                else str(config["training"].get("scheduler", "one-cycle"))
            ),
            "one_cycle_warmup_fraction": float(
                config["training"]["one_cycle_warmup_fraction"]
            ),
            "ema_decay": ema.decay if ema is not None else None,
            "swa": swa.metadata(best_swa_snapshots) if swa is not None else None,
        },
        "slice_selection": {
            "selection_method": selection_report["slice_selection_method"],
            "artifact": "slice-selection.json",
            "splits": {
                split: values["summary"]
                for split, values in selection_report["splits"].items()
            },
        },
        "dicom_cache": {
            "enabled": bool(dicom_cache_dir),
            "directory": dicom_cache_dir,
            "storage_dtype": "uint8",
            "preprocessing_version": PREPROCESSING_VERSION,
        },
        "label_source_validation": label_source_validation,
        "history": history,
        "attention_weights": "attention-weights.npz",
        **provenance,
    }
    write_json(output_dir / "run-summary.json", summary, allow_nan=True)
    print(f"Wrote reusable checkpoint {checkpoint_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
