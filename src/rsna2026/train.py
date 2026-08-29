"""Train, validate, and run inference for the RSNA knee baseline."""

from __future__ import annotations

import argparse
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
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from rsna2026.acquisition import CLASSIFICATION_METHOD, SLOT_NAMES
from rsna2026.checkpoint import (
    file_sha256,
    resolve_input_file,
    write_checkpoint_manifest,
)
from rsna2026.config import load_config
from rsna2026.constants import TARGETS
from rsna2026.cross_validation import (
    bootstrap_macro_auc_interval,
    create_inner_roles,
    create_outer_folds,
    dataframe_sha256,
    weighted_binary_cross_entropy,
)
from rsna2026.data import competition_input
from rsna2026.dataset import (
    PREPROCESSING_VERSION,
    KneeStudyDataset,
    StudyRecord,
    build_study_records,
    load_study_records_cache,
)
from rsna2026.model import build_dinov2_model
from rsna2026.ensemble import (
    ENSEMBLE_FORMAT_VERSION,
    load_ensemble_manifest,
    percentile_rank_mean,
)
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
        "--mode", choices=("train", "inference"), required=True
    )
    return parser.parse_args(argv)


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
    """Initialize the requested two-GPU NCCL process group."""
    training_config = config.get("training", {})
    enabled = (
        mode == "train"
        and isinstance(training_config, dict)
        and bool(training_config.get("distributed", False))
    )
    if not enabled:
        return DistributedContext(enabled=False)

    expected = int(training_config.get("expected_gpu_count", 2))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if expected != 2:
        raise ValueError("This experiment requires exactly two GPUs")
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
    dist.init_process_group(backend="nccl", init_method="env://")
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
    candidate_weights = np.asarray(
        [[confidence_weight(value) for value in row] for row in candidate_labels],
        dtype=np.float32,
    )
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
    augment: bool,
    cache_dir: str | Path | None,
    sampler: Sampler[int] | None = None,
) -> DataLoader:
    dataset = KneeStudyDataset(
        records,
        image_size,
        physical_fov_mm,
        labels,
        label_weights,
        augment,
        cache_dir,
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


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    target_weights: torch.Tensor,
    num_slots: int,
    sampled_slices_per_slot: int,
    window_size: int,
    training_window_stride: int,
    distributed: DistributedContext | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0
    starts = slice_window_starts(
        sampled_slices_per_slot, window_size, training_window_stride
    )
    for images, present, labels, label_weights in loader:
        start_index = torch.randint(len(starts), (1,), device=device)
        if distributed is not None and distributed.enabled:
            dist.broadcast(start_index, src=0)
        start = starts[int(start_index.item())]
        images, present = select_slice_window(
            images,
            present,
            num_slots,
            sampled_slices_per_slot,
            window_size,
            start,
        )
        images = images.to(device, non_blocking=True)
        present = present.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        label_weights = label_weights.to(device, non_blocking=True)
        valid = torch.isfinite(labels) & torch.isfinite(label_weights)
        targets = torch.nan_to_num(labels, nan=0.0)
        weights = torch.where(valid, label_weights, 0.0)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images, present)
            losses = functional.binary_cross_entropy_with_logits(
                logits, targets, reduction="none", pos_weight=target_weights
            )
            loss = (losses * weights).sum() / weights.sum().clamp_min(1.0)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
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
    num_slots: int,
    sampled_slices_per_slot: int,
    window_size: int,
    inference_window_stride: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Average per-window probabilities over an ordered slice stack."""
    model.eval()
    predictions = []
    labels = []
    starts = slice_window_starts(
        sampled_slices_per_slot, window_size, inference_window_stride
    )
    for images, present, batch_labels, _ in loader:
        probability_sum = None
        for start in starts:
            window_images, window_present = select_slice_window(
                images,
                present,
                num_slots,
                sampled_slices_per_slot,
                window_size,
                start,
            )
            with torch.autocast(
                device_type=device.type, enabled=device.type == "cuda"
            ):
                logits = model(
                    window_images.to(device, non_blocking=True),
                    window_present.to(device, non_blocking=True),
                )
            probabilities = torch.sigmoid(logits.float())
            probability_sum = (
                probabilities
                if probability_sum is None
                else probability_sum + probabilities
            )
        if probability_sum is None:
            raise RuntimeError("No inference slice windows were generated")
        predictions.append((probability_sum / len(starts)).cpu().numpy())
        if batch_labels.numel():
            labels.append(batch_labels.numpy())
    prediction_array = (
        np.concatenate(predictions)
        if predictions
        else np.empty((0, len(TARGETS)))
    )
    label_array = np.concatenate(labels) if labels else None
    return prediction_array, label_array


def predict_distributed(
    model: nn.Module,
    loader: DataLoader,
    sampler: DistributedEvaluationSampler,
    expected_size: int,
    device: torch.device,
    num_slots: int,
    sampled_slices_per_slot: int,
    window_size: int,
    inference_window_stride: int,
    distributed: DistributedContext,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Predict one non-overlapping shard and restore order on rank zero."""
    predictions, labels = predict(
        unwrap_model(model),
        loader,
        device,
        num_slots,
        sampled_slices_per_slot,
        window_size,
        inference_window_stride,
    )
    local = (list(sampler), predictions, labels)
    gathered: list[
        tuple[Sequence[int], np.ndarray, np.ndarray | None] | None
    ] = [None] * distributed.world_size
    dist.all_gather_object(gathered, local)
    if not distributed.primary:
        return None, None
    complete = [shard for shard in gathered if shard is not None]
    return merge_prediction_shards(complete, expected_size)


def slice_window_starts(
    sampled_slices_per_slot: int, window_size: int, stride: int
) -> tuple[int, ...]:
    """Return valid starts for fixed-width ordered slice windows."""
    if sampled_slices_per_slot <= 0:
        raise ValueError("Sampled slices per slot must be positive")
    if window_size <= 0 or window_size > sampled_slices_per_slot:
        raise ValueError("Window size must be within the sampled slice count")
    if stride <= 0:
        raise ValueError("Window stride must be positive")
    return tuple(
        range(0, sampled_slices_per_slot - window_size + 1, stride)
    )


def select_slice_window(
    images: torch.Tensor,
    present: torch.Tensor,
    num_slots: int,
    sampled_slices_per_slot: int,
    window_size: int,
    start: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract the same consecutive slice window from every acquisition slot."""
    if images.ndim != 5:
        raise ValueError(f"Expected five-dimensional images, received {images.shape}")
    batch_size, num_views, channels, height, width = images.shape
    expected_views = num_slots * sampled_slices_per_slot
    if num_views != expected_views:
        raise ValueError(
            f"Expected {expected_views} sampled views, received {num_views}"
        )
    if present.shape != (batch_size, num_views):
        raise ValueError(
            f"Expected presence shape {(batch_size, num_views)}, "
            f"received {tuple(present.shape)}"
        )
    if start < 0 or start + window_size > sampled_slices_per_slot:
        raise ValueError("Slice window lies outside the sampled stack")
    window_images = images.reshape(
        batch_size,
        num_slots,
        sampled_slices_per_slot,
        channels,
        height,
        width,
    )[:, :, start : start + window_size]
    window_present = present.reshape(
        batch_size, num_slots, sampled_slices_per_slot
    )[:, :, start : start + window_size]
    return (
        window_images.reshape(
            batch_size, num_slots * window_size, channels, height, width
        ),
        window_present.reshape(batch_size, num_slots * window_size),
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
    sampled_slices_per_slot: int,
    slice_band_start: float,
    slice_band_end: float,
) -> dict[str, object]:
    """Create an auditable record of every selected series and slice."""
    report: dict[str, object] = {
        "series_classification": CLASSIFICATION_METHOD,
        "slots": list(SLOT_NAMES),
        "sampled_slices_per_slot": sampled_slices_per_slot,
        "slice_band": [slice_band_start, slice_band_end],
        "splits": {},
    }
    split_reports: dict[str, object] = {}
    for split, records in records_by_split.items():
        methods = Counter(
            selection.ordering_method
            for record in records
            for selection in record.selections
        )
        selected_slots = Counter(
            selection.slot
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
        entries = [
            {
                "study_uid": record.study_uid,
                "series": [asdict(selection) for selection in record.selections],
                "series_classifications": [
                    asdict(classification)
                    for classification in record.classifications
                ],
            }
            for record in records
        ]
        summary = {
            "studies": len(records),
            "series_by_ordering_method": dict(sorted(methods.items())),
            "selected_series_by_slot": {
                slot: selected_slots.get(slot, 0) for slot in SLOT_NAMES
            },
            "missing_slots_by_slot": {
                slot: len(records) - selected_slots.get(slot, 0)
                for slot in SLOT_NAMES
            },
            "classified_series": sum(
                len(record.classifications) for record in records
            ),
            "unassigned_series": unassigned_series,
            "weighting_sources": dict(sorted(weighting_sources.items())),
            "fat_suppression_sources": dict(
                sorted(fat_suppression_sources.items())
            ),
        }
        split_reports[split] = {"summary": summary, "studies": entries}
        print(f"{split} slice selection: {summary}", flush=True)
    report["splits"] = split_reports
    return report


def model_artifact_metadata(config: dict[str, object]) -> dict[str, object]:
    """Return architecture fields that must match across both notebook stages."""
    model_config = config["model"]
    data_config = config["data"]
    return {
        "name": str(model_config["name"]),  # type: ignore[index]
        "variant": str(model_config["variant"]),  # type: ignore[index]
        "num_targets": len(TARGETS),
        "targets": list(TARGETS),
        "unfreeze_last": int(model_config["unfreeze_last"]),  # type: ignore[index]
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
        "sampled_slices_per_slot": int(  # type: ignore[index]
            data_config["sampled_slices_per_slot"]
        ),
        "window_size": int(data_config["window_size"]),  # type: ignore[index]
        "training_window_stride": int(  # type: ignore[index]
            data_config["training_window_stride"]
        ),
        "inference_window_stride": int(  # type: ignore[index]
            data_config["inference_window_stride"]
        ),
        "inference_window_pooling": str(  # type: ignore[index]
            data_config["inference_window_pooling"]
        ),
        "input_representation": str(  # type: ignore[index]
            model_config["input_representation"]
        ),
        "series_classification": str(  # type: ignore[index]
            data_config["series_classification"]
        ),
        "slots": list(data_config["slots"]),  # type: ignore[index]
    }


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
    if str(model_config["name"]) != "dinov2-small":  # type: ignore[index]
        model_name = model_config["name"]  # type: ignore[index]
        raise ValueError(f"Unsupported model: {model_name}")
    if image_size % 14:
        raise ValueError("DINOv2 image size must be divisible by its patch size 14")
    if str(model_config["aggregation"]) != "target-attention":  # type: ignore[index]
        raise ValueError("Only target-attention aggregation is supported")
    input_representation = str(
        model_config["input_representation"]  # type: ignore[index]
    )
    if input_representation != "2.5d-slice-channels":
        raise ValueError("Only 2.5D slice-channel input is supported")
    sampled_slices_per_slot = int(  # type: ignore[index]
        data_config["sampled_slices_per_slot"]
    )
    window_size = int(data_config["window_size"])  # type: ignore[index]
    training_window_stride = int(  # type: ignore[index]
        data_config["training_window_stride"]
    )
    inference_window_stride = int(  # type: ignore[index]
        data_config["inference_window_stride"]
    )
    slice_window_starts(
        sampled_slices_per_slot, window_size, training_window_stride
    )
    slice_window_starts(
        sampled_slices_per_slot, window_size, inference_window_stride
    )
    inference_window_pooling = str(  # type: ignore[index]
        data_config["inference_window_pooling"]
    )
    if inference_window_pooling != "probability-mean":
        raise ValueError("Only probability-mean inference pooling is supported")
    configured_slots = data_config["slots"]  # type: ignore[index]
    slot_names = tuple(str(name) for name in configured_slots)
    if slot_names != SLOT_NAMES:
        raise ValueError(f"Configured slots must be {list(SLOT_NAMES)}")
    model, model_source = build_dinov2_model(
        input_root=config["paths"]["input_root"],  # type: ignore[index]
        variant=str(model_config["variant"]),  # type: ignore[index]
        num_targets=len(TARGETS),
        unfreeze_last=int(model_config["unfreeze_last"]),  # type: ignore[index]
        dropout=float(model_config["dropout"]),  # type: ignore[index]
        num_slots=len(slot_names),
        slices_per_slot=window_size,
        attention_hidden_size=int(  # type: ignore[index]
            model_config["attention_hidden_size"]
        ),
    )
    model = model.to(device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if int(os.environ.get("RANK", "0")) == 0:
        print(
            f"Loaded {model_config['name']} from {model_source}; "  # type: ignore[index]
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


def supervision_arrays(
    table: pd.DataFrame,
    public_rows: dict[str, tuple[float, ...]],
    gold_loss_weight: float,
    use_gold: bool,
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
            weights.append([confidence_weight(value) for value in values])
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
) -> tuple[dict[str, object] | None, pd.DataFrame | None, bool]:
    """Train one outer fold using only its weak inner validation for selection."""
    run_seed = int(config["run"]["seed"]) + fold  # type: ignore[index]
    seed_everything(run_seed + distributed.rank)
    cv_config = config["cross_validation"]
    training_config = config["training"]
    data_config = config["data"]
    if not isinstance(cv_config, dict):
        raise TypeError("cross_validation configuration must be a table")
    gold_loss_weight = float(cv_config["gold_loss_weight"])

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
        train_table, public_rows, gold_loss_weight, True
    )
    inner_labels, inner_weights = supervision_arrays(
        inner_table, public_rows, gold_loss_weight, False
    )
    outer_labels = outer_table[TARGETS].to_numpy(dtype=np.float32)
    outer_weights = np.ones_like(outer_labels, dtype=np.float32)

    image_size = int(data_config["image_size"])
    physical_fov_mm = float(data_config["physical_fov_mm"])
    sampled_slices_per_slot = int(data_config["sampled_slices_per_slot"])
    window_size = int(data_config["window_size"])
    training_window_stride = int(data_config["training_window_stride"])
    selection_window_stride = int(cv_config["selection_window_stride"])
    inference_window_stride = int(data_config["inference_window_stride"])
    slot_names = list(data_config["slots"])
    batch_size = int(training_config["batch_size_per_device"])
    num_workers = int(training_config["num_workers"])
    cache_dir = data_config.get("dicom_cache_dir")
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
        True,
        cache_dir,
        training_sampler,
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
        False,
        cache_dir,
        inner_sampler,
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
        False,
        cache_dir,
        outer_sampler,
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
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[backbone_learning_rate, head_learning_rate],
        epochs=epochs,
        steps_per_epoch=len(training_loader),
        pct_start=float(training_config["one_cycle_warmup_fraction"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    target_weights = positive_weights(train_labels, train_weights).to(device)
    max_runtime_seconds = float(training_config["max_runtime_hours"]) * 3600.0
    best_score = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history = []
    time_limit_reached = False
    selection_targets = np.where(inner_weights > 0, inner_labels, np.nan)
    for epoch in range(epochs):
        training_sampler.set_epoch(epoch)
        loss = train_epoch(
            model,
            training_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            target_weights,
            len(slot_names),
            sampled_slices_per_slot,
            window_size,
            training_window_stride,
            distributed,
        )
        inner_predictions, _ = predict_distributed(
            model,
            inner_loader,
            inner_sampler,
            len(inner_table),
            device,
            len(slot_names),
            sampled_slices_per_slot,
            window_size,
            selection_window_stride,
            distributed,
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
                    "loss": loss,
                    "inner_validation_auc": auc,
                    "inner_validation_auc_by_target": per_target_auc,
                    "inner_validation_weighted_bce": weighted_bce,
                }
            )
            print(
                f"Fold {fold} epoch {epoch + 1}: loss={loss:.5f}, "
                f"inner_auc={auc:.5f}, inner_bce={weighted_bce:.5f}",
                flush=True,
            )
            if best_state is None or auc > best_score:
                best_score = auc
                best_epoch = epoch + 1
                best_state = checkpoint_state_dict(model)
            time_limit_reached = (
                time.monotonic() - started_at >= max_runtime_seconds
            )
            if time_limit_reached:
                print(f"Global runtime limit reached in fold {fold}", flush=True)
        stop = torch.tensor(
            int(time_limit_reached), dtype=torch.int32, device=device
        )
        dist.broadcast(stop, src=0)
        time_limit_reached = bool(stop.item())
        if time_limit_reached:
            break
    if distributed.primary:
        if best_state is None:
            raise RuntimeError(f"Fold {fold} did not produce a model state")
        raw_model.load_state_dict(best_state)
    for tensor in raw_model.state_dict().values():
        dist.broadcast(tensor, src=0)
    outer_predictions, predicted_outer_labels = predict_distributed(
        model,
        outer_loader,
        outer_sampler,
        len(outer_table),
        device,
        len(slot_names),
        sampled_slices_per_slot,
        window_size,
        inference_window_stride,
        distributed,
    )
    fold_summary: dict[str, object] | None = None
    oof: pd.DataFrame | None = None
    if distributed.primary:
        if outer_predictions is None or predicted_outer_labels is None:
            raise RuntimeError("Primary rank did not receive outer predictions")
        if not np.array_equal(predicted_outer_labels, outer_labels):
            raise RuntimeError(f"Fold {fold} outer labels changed during prediction")
        fold_dir = output_dir / f"fold-{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = fold_dir / "best-model.pt"
        torch.save(best_state, checkpoint_path)
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
                "selection_window_stride": selection_window_stride,
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
            "time_limit_reached": time_limit_reached,
            "history": history,
            "checkpoint": manifest,
            "model_source": str(model_source),
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "distributed": {
                "backend": distributed.backend,
                "world_size": distributed.world_size,
                "batch_size_per_device": batch_size,
                "global_batch_size": batch_size * distributed.world_size,
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
    dist.barrier()
    del raw_model
    del model, optimizer, scheduler, scaler
    del training_loader, inner_loader, outer_loader
    gc.collect()
    torch.cuda.empty_cache()
    return fold_summary, oof, time_limit_reached


def run_cross_validation_training(
    config: dict[str, object],
    output_dir: Path,
    started_at: float,
    distributed: DistributedContext,
) -> int:
    """Train four grouped folds and write auditable OOF artifacts."""
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
    if distributed.primary:
        selection_cache_path = Path(str(data_config["study_records_cache_path"]))
        all_records, selection_summary = load_study_records_cache(
            selection_cache_path,
            input_path / "train_series",
            all_uids,
            int(data_config["sampled_slices_per_slot"]),
            float(data_config["slice_band_start"]),
            float(data_config["slice_band_end"]),
            list(data_config["slots"]),
            str(data_config["series_classification"]),
            str(data_config["study_records_cache_sha256"]),
            str(data_config["study_records_cache_uids_sha256"]),
        )
        temporary_records_cache = records_cache.with_suffix(".temporary")
        with temporary_records_cache.open("wb") as file:
            pickle.dump(all_records, file, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_records_cache, records_cache)
        selection_output = output_dir / "slice-selection.json"
        if selection_cache_path.suffix == ".xz":
            with lzma.open(selection_cache_path, "rb") as source:
                with selection_output.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
        else:
            shutil.copyfile(selection_cache_path, selection_output)
        print(
            f"Loaded pinned study-record cache: {selection_summary}", flush=True
        )
    if not distributed.primary:
        deadline = time.monotonic() + 1800.0
        while not records_cache.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for the study-record cache")
            time.sleep(1.0)
        with records_cache.open("rb") as file:
            all_records = pickle.load(file)
    if len(all_records) != len(all_uids):
        raise RuntimeError("Distributed study-record cache is incomplete")
    records_by_uid = {record.study_uid: record for record in all_records}
    dist.barrier()
    device = torch.device("cuda", distributed.local_rank)
    local_gpu_name = torch.cuda.get_device_name(distributed.local_rank)
    gpu_names: list[str | None] = [None] * distributed.world_size
    dist.all_gather_object(gpu_names, local_gpu_name)
    if distributed.primary:
        print(
            f"Distributed training on {distributed.world_size} GPUs: "
            f"{gpu_names}",
            flush=True,
        )
    fold_summaries = []
    oof_tables = []
    for fold in range(int(cv_config["folds"])):
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
            distributed,
        )
        if distributed.primary:
            if summary is None or oof is None:
                raise RuntimeError("Primary rank did not receive fold artifacts")
            fold_summaries.append(summary)
            oof_tables.append(oof)
        if time_limit and fold < int(cv_config["folds"]) - 1:
            raise RuntimeError("Runtime limit prevented a complete four-fold ensemble")
    if not distributed.primary:
        return 0
    oof = pd.concat(oof_tables, ignore_index=True).sort_values(
        "StudyInstanceUID"
    ).reset_index(drop=True)
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
        "within_model_pooling": str(data_config["inference_window_pooling"]),
    }
    write_json(output_dir / "ensemble-manifest.json", ensemble_manifest)
    elapsed = time.monotonic() - started_at
    run_summary = {
        "status": "cross-validation-training-complete",
        "device": str(device),
        "distributed": {
            "enabled": distributed.enabled,
            "backend": distributed.backend,
            "world_size": distributed.world_size,
            "gpu_names": gpu_names,
            "batch_size_per_device": int(
                config["training"]["batch_size_per_device"]  # type: ignore[index]
            ),
            "global_batch_size": int(
                config["training"]["batch_size_per_device"]  # type: ignore[index]
            )
            * distributed.world_size,
        },
        "optimization": {
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
            "scheduler": "one-cycle",
            "one_cycle_warmup_fraction": float(
                config["training"]["one_cycle_warmup_fraction"]  # type: ignore[index]
            ),
        },
        "elapsed_seconds": elapsed,
        "folds": fold_summaries,
        "cross_validation": cv_summary,
        "ensemble_manifest": "ensemble-manifest.json",
        "fold_audit": "fold-audit.json",
        "fold_assignments": "fold-assignments.csv",
        "inner_splits": "inner-splits.csv",
        "oof_predictions": "oof-predictions.csv",
        "label_source_validation": label_validation,
        "study_records_cache": {
            "sha256": str(data_config["study_records_cache_sha256"]),
            "study_uids_sha256": str(
                data_config["study_records_cache_uids_sha256"]
            ),
            "artifact": "slice-selection.json",
        },
        "pseudo_label_file": str(pseudo_config["file_name"]),
        "pseudo_labels_sha256": public.sha256,
        "model": artifact_metadata,
        "selection_window_stride": int(cv_config["selection_window_stride"]),
        "final_prediction_window_stride": int(data_config["inference_window_stride"]),
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
    sampled_slices_per_slot = int(  # type: ignore[index]
        data_config["sampled_slices_per_slot"]
    )
    window_size = int(data_config["window_size"])  # type: ignore[index]
    inference_window_stride = int(  # type: ignore[index]
        data_config["inference_window_stride"]
    )
    slice_band_start = float(data_config["slice_band_start"])  # type: ignore[index]
    slice_band_end = float(data_config["slice_band_end"])  # type: ignore[index]
    slot_names = list(data_config["slots"])  # type: ignore[index]
    classification_method = str(  # type: ignore[index]
        data_config["series_classification"]
    )
    test_uids = test_table["StudyInstanceUID"].astype(str).tolist()
    test_records = build_study_records(
        test_series,
        input_path / "test_series",
        test_uids,
        sampled_slices_per_slot,
        slice_band_start,
        slice_band_end,
        slot_names,
        classification_method,
    )
    selection_report = slice_selection_report(
        {"test": test_records},
        sampled_slices_per_slot,
        slice_band_start,
        slice_band_end,
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
        predictions, _ = predict(
            model,
            test_loader,
            device,
            len(slot_names),
            sampled_slices_per_slot,
            window_size,
            inference_window_stride,
        )
        member_predictions.append(predictions)
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
        "slice_selection": {
            "slice_band": selection_report["slice_band"],
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


def main(argv: Sequence[str] | None = None) -> int:
    started_at = time.monotonic()
    args = parse_args(argv)
    config = load_config(args.config)
    distributed = initialize_distributed(config, args.mode)
    seed = int(config["run"]["seed"])
    seed_everything(seed + distributed.rank)

    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "inference":
        return run_inference(config, output_dir, started_at)

    cv_config = config.get("cross_validation", {})
    if isinstance(cv_config, dict) and bool(cv_config.get("enabled", False)):
        if not distributed.enabled:
            raise RuntimeError(
                "Cross-validation experiment requires two-GPU distributed training"
            )
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
    sampled_slices_per_slot = int(
        config["data"]["sampled_slices_per_slot"]
    )
    window_size = int(config["data"]["window_size"])
    training_window_stride = int(
        config["data"]["training_window_stride"]
    )
    inference_window_stride = int(
        config["data"]["inference_window_stride"]
    )
    slice_band_start = float(config["data"]["slice_band_start"])
    slice_band_end = float(config["data"]["slice_band_end"])
    slot_names = list(config["data"]["slots"])
    classification_method = str(config["data"]["series_classification"])
    training_records = build_study_records(
        train_series,
        input_path / "train_series",
        training_table["StudyInstanceUID"].astype(str).tolist(),
        sampled_slices_per_slot,
        slice_band_start,
        slice_band_end,
        slot_names,
        classification_method,
    )
    validation_records = build_study_records(
        train_series,
        input_path / "train_series",
        validation_table["StudyInstanceUID"].astype(str).tolist(),
        sampled_slices_per_slot,
        slice_band_start,
        slice_band_end,
        slot_names,
        classification_method,
    )
    selection_report = slice_selection_report(
        {
            "training": training_records,
            "validation": validation_records,
        },
        sampled_slices_per_slot,
        slice_band_start,
        slice_band_end,
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
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[backbone_learning_rate, head_learning_rate],
        epochs=epochs,
        steps_per_epoch=len(training_loader),
        pct_start=float(config["training"]["one_cycle_warmup_fraction"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    target_weights = positive_weights(training_labels, training_weights).to(device)
    max_runtime_seconds = float(
        config["training"]["max_runtime_hours"]
    ) * 3600.0

    best_score = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history = []
    time_limit_reached = False
    for epoch in range(epochs):
        loss = train_epoch(
            model,
            training_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            target_weights,
            len(slot_names),
            sampled_slices_per_slot,
            window_size,
            training_window_stride,
        )
        validation_predictions, validation_labels = predict(
            model,
            validation_loader,
            device,
            len(slot_names),
            sampled_slices_per_slot,
            window_size,
            inference_window_stride,
        )
        if validation_labels is None:
            raise RuntimeError("Validation labels are unavailable")
        auc = macro_auc(validation_labels, validation_predictions)
        per_target_auc = target_auc_scores(validation_labels, validation_predictions)
        history.append(
            {
                "epoch": epoch + 1,
                "loss": loss,
                "validation_auc": auc,
                "validation_auc_by_target": per_target_auc,
            }
        )
        print(
            f"Epoch {epoch + 1}: loss={loss:.5f}, validation_auc={auc:.5f}",
            flush=True,
        )
        selection_score = auc if math.isfinite(auc) else -loss
        if best_state is None or selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
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
        "completed_epochs": len(history),
        "best_epoch": best_epoch,
        "time_limit_reached": time_limit_reached,
        "checkpoint": manifest,
        "model": {
            **artifact_metadata,
            "source": str(model_source),
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "grayscale_to_rgb": "three-ordered-slices-as-channels",
            "slot_pooling": "none-2.5d-encoding",
            "slot_aggregation": "per-target-masked-attention",
            "backbone_learning_rate": backbone_learning_rate,
            "head_learning_rate": head_learning_rate,
            "weight_decay": float(config["training"]["weight_decay"]),
            "scheduler": "one-cycle",
            "one_cycle_warmup_fraction": float(
                config["training"]["one_cycle_warmup_fraction"]
            ),
        },
        "slice_selection": {
            "slice_band": selection_report["slice_band"],
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
        **provenance,
    }
    write_json(output_dir / "run-summary.json", summary, allow_nan=True)
    print(f"Wrote reusable checkpoint {checkpoint_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
