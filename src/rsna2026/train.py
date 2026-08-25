"""Train, validate, and run inference for the RSNA knee baseline."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader

from rsna2026.acquisition import CLASSIFICATION_METHOD, SLOT_NAMES
from rsna2026.checkpoint import (
    load_and_verify_manifest,
    resolve_input_file,
    write_checkpoint_manifest,
)
from rsna2026.config import load_config
from rsna2026.constants import TARGETS
from rsna2026.data import competition_input
from rsna2026.dataset import KneeStudyDataset, StudyRecord, build_study_records
from rsna2026.model import build_dinov2_model
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
    labels: np.ndarray | None,
    label_weights: np.ndarray | None,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    augment: bool,
) -> DataLoader:
    dataset = KneeStudyDataset(
        records, image_size, labels, label_weights, augment
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
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
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0
    for images, present, labels, label_weights in loader:
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
    return total_loss / max(total_batches, 1)


@torch.no_grad()
def predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray | None]:
    model.eval()
    predictions = []
    labels = []
    for images, present, batch_labels, _ in loader:
        with torch.autocast(
            device_type=device.type, enabled=device.type == "cuda"
        ):
            logits = model(
                images.to(device, non_blocking=True),
                present.to(device, non_blocking=True),
            )
        predictions.append(torch.sigmoid(logits).cpu().numpy())
        if batch_labels.numel():
            labels.append(batch_labels.numpy())
    prediction_array = (
        np.concatenate(predictions)
        if predictions
        else np.empty((0, len(TARGETS)))
    )
    label_array = np.concatenate(labels) if labels else None
    return prediction_array, label_array


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
    slices_per_slot: int,
    slice_band_start: float,
    slice_band_end: float,
) -> dict[str, object]:
    """Create an auditable record of every selected series and slice."""
    report: dict[str, object] = {
        "series_classification": CLASSIFICATION_METHOD,
        "slots": list(SLOT_NAMES),
        "slices_per_slot": slices_per_slot,
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
        "slices_per_slot": int(data_config["slices_per_slot"]),  # type: ignore[index]
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
    if str(model_config["name"]) != "dinov2-small":  # type: ignore[index]
        model_name = model_config["name"]  # type: ignore[index]
        raise ValueError(f"Unsupported model: {model_name}")
    if image_size % 14:
        raise ValueError("DINOv2 image size must be divisible by its patch size 14")
    if str(model_config["aggregation"]) != "target-attention":  # type: ignore[index]
        raise ValueError("Only target-attention aggregation is supported")
    slot_names = tuple(str(name) for name in data_config["slots"])  # type: ignore[index]
    if slot_names != SLOT_NAMES:
        raise ValueError(f"Configured slots must be {list(SLOT_NAMES)}")
    model, model_source = build_dinov2_model(
        input_root=config["paths"]["input_root"],  # type: ignore[index]
        variant=str(model_config["variant"]),  # type: ignore[index]
        num_targets=len(TARGETS),
        unfreeze_last=int(model_config["unfreeze_last"]),  # type: ignore[index]
        dropout=float(model_config["dropout"]),  # type: ignore[index]
        num_slots=len(slot_names),
        slices_per_slot=int(data_config["slices_per_slot"]),  # type: ignore[index]
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
    slices_per_slot = int(data_config["slices_per_slot"])  # type: ignore[index]
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
        slices_per_slot,
        slice_band_start,
        slice_band_end,
        slot_names,
        classification_method,
    )
    selection_report = slice_selection_report(
        {"test": test_records},
        slices_per_slot,
        slice_band_start,
        slice_band_end,
    )
    write_json(output_dir / "slice-selection.json", selection_report)
    test_loader = make_loader(
        test_records,
        image_size,
        None,
        None,
        int(config["training"]["batch_size"]),  # type: ignore[index]
        int(config["training"]["num_workers"]),  # type: ignore[index]
        False,
        False,
    )

    inference_config = config["inference"]
    checkpoint_path = resolve_input_file(
        inference_config["checkpoint_path"], input_root  # type: ignore[index]
    )
    manifest_path = resolve_input_file(
        inference_config["checkpoint_manifest_path"], input_root  # type: ignore[index]
    )
    manifest = load_and_verify_manifest(checkpoint_path, manifest_path)
    expected_model = model_artifact_metadata(config)
    if manifest.get("model") != expected_model:
        raise ValueError(
            "Checkpoint model metadata does not match the inference configuration"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_source, total_parameters, trainable_parameters = (
        build_configured_model(config, device)
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    predictions, _ = predict(model, test_loader, device)
    submission_path = output_dir / "submission.csv"
    write_submission(sample_submission, test_uids, predictions, submission_path)
    summary = {
        "status": "inference-complete",
        "device": str(device),
        "test_studies": len(test_records),
        "targets": TARGETS,
        "elapsed_seconds": time.monotonic() - started_at,
        "checkpoint_source": str(checkpoint_path),
        "checkpoint_manifest_source": str(manifest_path),
        "checkpoint": manifest,
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
    }
    write_json(output_dir / "run-summary.json", summary)
    print(f"Wrote {submission_path}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    started_at = time.monotonic()
    args = parse_args(argv)
    config = load_config(args.config)
    seed = int(config["run"]["seed"])
    seed_everything(seed)

    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "inference":
        return run_inference(config, output_dir, started_at)

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
    slices_per_slot = int(config["data"]["slices_per_slot"])
    slice_band_start = float(config["data"]["slice_band_start"])
    slice_band_end = float(config["data"]["slice_band_end"])
    slot_names = list(config["data"]["slots"])
    classification_method = str(config["data"]["series_classification"])
    training_records = build_study_records(
        train_series,
        input_path / "train_series",
        training_table["StudyInstanceUID"].astype(str).tolist(),
        slices_per_slot,
        slice_band_start,
        slice_band_end,
        slot_names,
        classification_method,
    )
    validation_records = build_study_records(
        train_series,
        input_path / "train_series",
        validation_table["StudyInstanceUID"].astype(str).tolist(),
        slices_per_slot,
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
        slices_per_slot,
        slice_band_start,
        slice_band_end,
    )
    (output_dir / "slice-selection.json").write_text(
        json.dumps(selection_report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"]["num_workers"])
    training_loader = make_loader(
        training_records,
        image_size,
        training_labels,
        training_weights,
        batch_size,
        num_workers,
        True,
        True,
    )
    validation_loader = make_loader(
        validation_records,
        image_size,
        validation_labels,
        validation_weights,
        batch_size,
        num_workers,
        False,
        False,
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
        )
        validation_predictions, validation_labels = predict(
            model, validation_loader, device
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
            "grayscale_to_rgb": "channel-repeat",
            "slot_pooling": "masked-slice-mean",
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
        "label_source_validation": label_source_validation,
        "history": history,
        **provenance,
    }
    write_json(output_dir / "run-summary.json", summary, allow_nan=True)
    print(f"Wrote reusable checkpoint {checkpoint_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
