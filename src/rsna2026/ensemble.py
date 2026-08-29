"""Checkpoint-ensemble validation and percentile-rank averaging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rsna2026.checkpoint import file_sha256, load_and_verify_manifest


ENSEMBLE_FORMAT_VERSION = 1


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks, preserving ties."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def percentile_rank_mean(predictions: list[np.ndarray]) -> np.ndarray:
    """Average per-model, per-target percentile ranks."""
    if not predictions:
        raise ValueError("At least one prediction array is required")
    shape = predictions[0].shape
    if len(shape) != 2 or any(array.shape != shape for array in predictions):
        raise ValueError("All prediction arrays must share a two-dimensional shape")
    ranked = np.empty((len(predictions), *shape), dtype=np.float64)
    for model_index, array in enumerate(predictions):
        for target_index in range(shape[1]):
            ranked[model_index, :, target_index] = (
                average_ranks(array[:, target_index]) / shape[0]
            )
    return ranked.mean(axis=0)


def load_ensemble_manifest(path: str | Path) -> dict[str, Any]:
    """Load an ensemble manifest and verify every relative checkpoint."""
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != ENSEMBLE_FORMAT_VERSION:
        raise ValueError("Unsupported ensemble manifest format")
    members = manifest.get("members")
    if not isinstance(members, list) or len(members) != 4:
        raise ValueError("Ensemble manifest must contain four members")
    folds = []
    for member in members:
        checkpoint = manifest_path.parent / str(member["checkpoint_file"])
        checkpoint_manifest = manifest_path.parent / str(
            member["checkpoint_manifest_file"]
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if file_sha256(checkpoint) != member.get("checkpoint_sha256"):
            raise ValueError(f"Checkpoint SHA-256 mismatch: {checkpoint}")
        verified = load_and_verify_manifest(checkpoint, checkpoint_manifest)
        fold = int(member["fold"])
        if int(verified.get("fold", -1)) != fold:
            raise ValueError(f"Checkpoint fold metadata mismatch: {checkpoint}")
        if verified.get("model") != manifest.get("model"):
            raise ValueError(f"Checkpoint model metadata mismatch: {checkpoint}")
        folds.append(fold)
    if sorted(folds) != [0, 1, 2, 3]:
        raise ValueError("Ensemble folds must be exactly 0, 1, 2, and 3")
    return manifest
