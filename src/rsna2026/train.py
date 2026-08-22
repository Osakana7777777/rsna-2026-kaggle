"""Train, validate, and run inference for the RSNA knee baseline."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from rsna2026.config import load_config
from rsna2026.data import competition_input
from rsna2026.dataset import KneeStudyDataset, StudyRecord, build_study_records
from rsna2026.model import KneeAbnormalityModel


TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


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


def macro_auc(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Compute an unweighted mean ROC AUC without an extra dependency."""
    scores = []
    for target_index in range(labels.shape[1]):
        target = labels[:, target_index]
        prediction = predictions[:, target_index]
        valid = np.isfinite(target) & np.isfinite(prediction)
        target = (target[valid] > 0.5).astype(int)
        prediction = prediction[valid]
        positives = target == 1
        negatives = target == 0
        if not positives.any() or not negatives.any():
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
        scores.append(float(auc))
    return float(np.mean(scores)) if scores else float("nan")


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
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    augment: bool,
) -> DataLoader:
    dataset = KneeStudyDataset(records, image_size, labels, augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def positive_weights(labels: np.ndarray) -> torch.Tensor:
    """Estimate bounded per-target positive weights from available labels."""
    weights = []
    for column in labels.T:
        valid = np.isfinite(column)
        positives = float(np.sum(column[valid] > 0.5))
        negatives = float(np.sum(column[valid] <= 0.5))
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
    for images, present, labels in loader:
        images = images.to(device, non_blocking=True)
        present = present.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        valid = torch.isfinite(labels)
        targets = torch.nan_to_num(labels, nan=0.0)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images, present)
            losses = functional.binary_cross_entropy_with_logits(
                logits, targets, reduction="none", pos_weight=target_weights
            )
            loss = (losses * valid).sum() / valid.sum().clamp_min(1)
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
    for images, present, batch_labels in loader:
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

    missing_targets = [target for target in TARGETS if target not in train_table.columns]
    if missing_targets:
        raise ValueError(f"Training table is missing targets: {missing_targets}")

    benchmark = sample_submission.copy()
    benchmark[TARGETS] = 0.5
    benchmark.to_csv(output_dir / "submission.csv", index=False)

    labeled = train_table[TARGETS].notna().any(axis=1)
    supervised = train_table.loc[labeled].reset_index(drop=True)
    max_train_studies = int(config["training"].get("max_train_studies", 0))
    if max_train_studies > 0:
        supervised = supervised.sample(
            n=min(max_train_studies, len(supervised)), random_state=seed
        ).reset_index(drop=True)
    labels = supervised[TARGETS].to_numpy(dtype=np.float32)
    print(
        f"Using {len(supervised)} labeled studies out of {len(train_table)}; "
        f"available labels={int(np.isfinite(labels).sum())}",
        flush=True,
    )

    image_size = int(config["data"]["image_size"])
    slices_per_plane = int(config["data"]["slices_per_plane"])
    train_records = build_study_records(
        train_series,
        input_path / "train_series",
        supervised["StudyInstanceUID"].astype(str).tolist(),
        slices_per_plane,
    )
    test_uids = test_table["StudyInstanceUID"].astype(str).tolist()
    test_records = build_study_records(
        test_series, input_path / "test_series", test_uids, slices_per_plane
    )

    training_indices, validation_indices = split_indices(
        len(train_records), float(config["training"]["validation_fraction"]), seed
    )
    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"]["num_workers"])
    training_loader = make_loader(
        subset_records(train_records, training_indices),
        image_size,
        labels[training_indices],
        batch_size,
        num_workers,
        True,
        True,
    )
    validation_loader = make_loader(
        subset_records(train_records, validation_indices),
        image_size,
        labels[validation_indices],
        batch_size,
        num_workers,
        False,
        False,
    )
    test_loader = make_loader(
        test_records, image_size, None, batch_size, num_workers, False, False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Training on {device}; train={len(training_indices)}, "
        f"validation={len(validation_indices)}",
        flush=True,
    )
    model = KneeAbnormalityModel(len(TARGETS)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    target_weights = positive_weights(labels[training_indices]).to(device)

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
        history.append({"epoch": epoch + 1, "loss": loss, "validation_auc": auc})
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
        "labeled_studies": len(supervised),
        "training_studies": len(training_indices),
        "validation_studies": len(validation_indices),
        "test_studies": len(test_records),
        "targets": TARGETS,
        "history": history,
    }
    (output_dir / "run-summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output_dir / 'submission.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
