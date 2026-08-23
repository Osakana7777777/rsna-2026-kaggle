"""Train, validate, and run inference for the RSNA knee baseline."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional
from torch.utils.data import DataLoader

from rsna2026.config import load_config
from rsna2026.constants import TARGETS
from rsna2026.data import competition_input
from rsna2026.dataset import KneeStudyDataset, StudyRecord, build_study_records
from rsna2026.model import KneeAbnormalityModel
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
    model: KneeAbnormalityModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
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
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        total_batches += 1
    return total_loss / max(total_batches, 1)


@torch.no_grad()
def predict(
    model: KneeAbnormalityModel, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray | None]:
    model.eval()
    predictions = []
    labels = []
    for images, present, batch_labels, _ in loader:
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


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    seed = int(config["run"]["seed"])
    seed_everything(seed)

    input_path = competition_input(
        config["paths"]["input_root"], config["competition"]["input_slug"]
    )
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_table = pd.read_csv(input_path / "train.csv")
    test_table = pd.read_csv(input_path / "test.csv")
    train_series = pd.read_csv(input_path / "train_series.csv")
    test_series = pd.read_csv(input_path / "test_series.csv")
    sample_submission = pd.read_csv(input_path / "sample_submission.csv")

    missing_targets = [
        target for target in TARGETS if target not in train_table.columns
    ]
    if missing_targets:
        raise ValueError(f"Training table is missing targets: {missing_targets}")
    if "Report" not in train_table.columns:
        raise ValueError("Training table is missing Report")

    benchmark = sample_submission.copy()
    benchmark[TARGETS] = 0.5
    benchmark.to_csv(output_dir / "submission.csv", index=False)

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
    slices_per_plane = int(config["data"]["slices_per_plane"])
    training_records = build_study_records(
        train_series,
        input_path / "train_series",
        training_table["StudyInstanceUID"].astype(str).tolist(),
        slices_per_plane,
    )
    validation_records = build_study_records(
        train_series,
        input_path / "train_series",
        validation_table["StudyInstanceUID"].astype(str).tolist(),
        slices_per_plane,
    )
    test_uids = test_table["StudyInstanceUID"].astype(str).tolist()
    test_records = build_study_records(
        test_series, input_path / "test_series", test_uids, slices_per_plane
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
    test_loader = make_loader(
        test_records,
        image_size,
        None,
        None,
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
    model = KneeAbnormalityModel(len(TARGETS)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    target_weights = positive_weights(training_labels, training_weights).to(device)

    best_score = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    history = []
    for epoch in range(int(config["training"]["epochs"])):
        loss = train_epoch(model, training_loader, optimizer, device, target_weights)
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
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a model state")
    model.load_state_dict(best_state)
    torch.save(best_state, output_dir / "best-model.pt")

    test_predictions, _ = predict(model, test_loader, device)
    write_submission(
        sample_submission, test_uids, test_predictions, output_dir / "submission.csv"
    )
    summary = {
        "status": "train-infer-complete",
        "device": str(device),
        "pseudo_labels_enabled": pseudo_enabled,
        "training_studies": len(training_records),
        "validation_studies": len(validation_records),
        "test_studies": len(test_records),
        "targets": TARGETS,
        "label_source_validation": label_source_validation,
        "history": history,
        **provenance,
    }
    (output_dir / "run-summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output_dir / 'submission.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
