"""Validation and deterministic selection for public report-derived labels."""

from __future__ import annotations

import csv
import hashlib
import math
import random
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from rsna2026.constants import TARGETS


@dataclass(frozen=True)
class PublicLabelTable:
    """A fully validated mapping from study identifier to soft targets."""

    rows: dict[str, tuple[float, ...]]
    sha256: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_public_label_table(
    path: Path,
    expected_sha256: str,
    expected_rows: int,
) -> PublicLabelTable:
    """Load the pinned public table and reject any schema or value drift."""
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256.lower():
        raise ValueError(
            f"Public label hash mismatch: expected {expected_sha256}, "
            f"found {actual_sha256}"
        )

    expected_columns = ["StudyInstanceUID", *TARGETS]
    rows: dict[str, tuple[float, ...]] = {}
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != expected_columns:
            raise ValueError(
                f"Unexpected public label columns: {reader.fieldnames}"
            )
        for row in reader:
            study_uid = str(row.get("StudyInstanceUID", "")).strip()
            if not study_uid or study_uid in rows:
                raise ValueError(
                    f"Missing or duplicate public label UID: {study_uid!r}"
                )
            values = []
            for target in TARGETS:
                try:
                    value = float(row[target])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Invalid public label for {study_uid}, {target}"
                    ) from error
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"Public label outside [0, 1] for {study_uid}, {target}: "
                        f"{value}"
                    )
                values.append(value)
            rows[study_uid] = tuple(values)

    if len(rows) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} public label rows, found {len(rows)}"
        )
    return PublicLabelTable(rows=rows, sha256=actual_sha256)


def normalize_report(report: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(report))
    return " ".join(normalized.split()).casefold()


def report_hash(report: str) -> str:
    return hashlib.sha256(normalize_report(report).encode("utf-8")).hexdigest()


def confidence_weight(probability: float) -> float:
    """Map distance from the undecided value to a [0, 1] loss weight."""
    value = float(probability)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"Probability outside [0, 1]: {probability!r}")
    return 2.0 * abs(value - 0.5)


def stable_sample(values: Iterable[str], count: int, seed: int) -> list[str]:
    """Sample identifiers reproducibly, independent of source row order."""
    ordered = sorted(set(values))
    if len(ordered) < count:
        raise ValueError(f"Need {count} eligible studies, found {len(ordered)}")
    return random.Random(seed).sample(ordered, count)


def stable_collection_hash(values: Iterable[str]) -> str:
    material = "\n".join(sorted(values)) + "\n"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def binary_auc(labels: Sequence[float], scores: Sequence[float]) -> float:
    """Compute ROC AUC with average ranks for tied predictions."""
    pairs = [
        (float(score), int(float(label) > 0.5))
        for label, score in zip(labels, scores)
    ]
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        return float("nan")

    ordered = sorted(enumerate(pairs), key=lambda item: item[1][0])
    ranks = [0.0] * len(pairs)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1][0] == ordered[start][1][0]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[ordered[position][0]] = average_rank
        start = end
    positive_ranks = sum(rank for rank, (_, label) in zip(ranks, pairs) if label)
    return (positive_ranks - positives * (positives + 1) / 2) / (
        positives * negatives
    )


def source_validation(
    gold_rows: Iterable[Mapping[str, object]],
    public_rows: Mapping[str, Sequence[float]],
) -> dict[str, object]:
    """Compare public soft labels with every fully annotated study."""
    gold = list(gold_rows)
    per_target: dict[str, dict[str, float | int]] = {}
    aucs: list[float] = []
    matched_uids: set[str] = set()
    for target_index, target in enumerate(TARGETS):
        labels: list[float] = []
        scores: list[float] = []
        for row in gold:
            study_uid = str(row["StudyInstanceUID"])
            public = public_rows.get(study_uid)
            if public is None:
                continue
            matched_uids.add(study_uid)
            labels.append(float(row[target]))
            scores.append(float(public[target_index]))
        auc = binary_auc(labels, scores)
        if math.isfinite(auc):
            aucs.append(auc)
        per_target[target] = {"auc": auc, "matched_studies": len(labels)}
    return {
        "gold_studies": len(gold),
        "matched_gold_studies": len(matched_uids),
        "scored_targets": len(aucs),
        "macro_auc": sum(aucs) / len(aucs) if aucs else float("nan"),
        "per_target": per_target,
    }
