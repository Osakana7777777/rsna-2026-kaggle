"""Map structured report extractions to training-compatible soft-label CSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeAlias

from rsna2026.constants import TARGETS
from rsna2026.weak_label_decode import (
    CompetitionCall,
    DecodedFinding,
    competition_call,
    decode_prediction,
    initial_mask,
    initial_weight,
)

DEFAULT_CALL_PROBABILITIES: Mapping[CompetitionCall, float] = {
    "negative": 0.10,
    "uncertain": 0.50,
    "positive": 0.90,
}
CategoryKey: TypeAlias = tuple[int, int]
CategoryEstimate: TypeAlias = tuple[int, int, float]
CategoryCalibration: TypeAlias = Mapping[
    str, Mapping[CategoryKey, CategoryEstimate]
]


def soft_probability(
    finding: DecodedFinding,
    mapping: Mapping[CompetitionCall, float] = DEFAULT_CALL_PROBABILITIES,
) -> float:
    """Map a decoded finding to a replaceable, explicitly calibrated target."""
    probability = float(mapping[competition_call(finding)])
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"Calibrated probability outside [0, 1]: {probability}")
    return probability


def refined_ordered_probability(finding: DecodedFinding) -> float:
    """Rank structured states more finely than the three-level call mapping."""
    if finding.mention == 2:
        return 0.03
    if finding.mention == 0:
        return 0.12
    if finding.mention == 3:
        probability = 0.38
    else:
        probability = {
            0: 0.52,
            1: 0.22,
            2: 0.68,
            3: 0.92,
            4: 0.32,
        }[finding.threshold]
    certainty_rank = {0: 0.0, 1: 4.0, 2: 3.0, 3: 2.0, 4: 1.0}
    probability += 0.004 * certainty_rank[finding.certainty]
    probability += 0.0005 * finding.severity
    return max(0.0, min(1.0, probability))


def calibrated_probability(
    target: str,
    finding: DecodedFinding,
    calibration: CategoryCalibration | None,
) -> float:
    """Apply target/category calibration or retain the original mapping."""
    if calibration is None:
        return soft_probability(finding)
    estimate = calibration.get(target, {}).get(
        (finding.mention, finding.threshold)
    )
    if estimate is None:
        return refined_ordered_probability(finding)
    positives, count, prior_strength = estimate
    return (
        positives + prior_strength * refined_ordered_probability(finding)
    ) / (count + prior_strength)


def fit_category_calibration(
    predictions_path: Path,
    gold_labels_path: Path,
    prior_strength: float = 5.0,
) -> dict[str, dict[CategoryKey, CategoryEstimate]]:
    """Fit shrinkage estimates by target, mention, and threshold category."""
    if not prior_strength > 0.0:
        raise ValueError("Calibration prior strength must be positive")
    predictions: dict[str, object] = {}
    with predictions_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            study_id = str(record["study_id"]).strip()
            if not study_id or study_id in predictions:
                raise ValueError(
                    f"Missing or duplicate study identifier on line {line_number}"
                )
            predictions[study_id] = decode_prediction(record["prediction"])

    statistics: dict[str, dict[CategoryKey, list[int]]] = {
        target: {} for target in TARGETS
    }
    gold_studies = 0
    with gold_labels_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not {
            "StudyInstanceUID",
            *TARGETS,
        }.issubset(reader.fieldnames):
            raise ValueError("Gold label CSV does not contain the required columns")
        for row in reader:
            if any(str(row[target]).strip() == "" for target in TARGETS):
                continue
            study_id = str(row["StudyInstanceUID"]).strip()
            prediction = predictions.get(study_id)
            if prediction is None:
                raise ValueError(f"Gold study is absent from predictions: {study_id}")
            gold_studies += 1
            for target, finding in zip(TARGETS, prediction.findings):  # type: ignore[attr-defined]
                label = int(float(row[target]))
                if label not in (0, 1):
                    raise ValueError(f"Gold label must be binary: {study_id}, {target}")
                key = (finding.mention, finding.threshold)
                values = statistics[target].setdefault(key, [0, 0])
                values[0] += label
                values[1] += 1
    if not gold_studies:
        raise ValueError("Gold label CSV contains no fully labeled studies")

    calibration: dict[str, dict[CategoryKey, CategoryEstimate]] = {}
    for target, target_statistics in statistics.items():
        calibration[target] = {}
        for key, (positives, count) in target_statistics.items():
            calibration[target][key] = (positives, count, prior_strength)
    return calibration


def load_soft_label_rows(
    path: Path,
    calibration: CategoryCalibration | None = None,
    impute_synovitis_from_effusion: bool = False,
) -> list[dict[str, str | float]]:
    """Read canonical JSONL and return the exact CSV schema used by training."""
    rows: list[dict[str, str | float]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                study_id = str(record["study_id"]).strip()
                prediction = decode_prediction(record["prediction"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Invalid JSONL line {line_number}: {error}"
                ) from error
            if not study_id:
                raise ValueError(f"Missing study identifier on JSONL line {line_number}")
            if study_id in seen:
                raise ValueError(f"Duplicate study identifier: {study_id}")
            seen.add(study_id)
            row: dict[str, str | float] = {"StudyInstanceUID": study_id}
            row.update(
                {
                    target: calibrated_probability(target, finding, calibration)
                    for target, finding in zip(TARGETS, prediction.findings)
                }
            )
            if (
                impute_synovitis_from_effusion
                and prediction.findings[TARGETS.index("Synovitis")].mention == 0
            ):
                effusion = prediction.findings[TARGETS.index("Effusion")]
                row["Synovitis"] = 0.30 + 0.10 * refined_ordered_probability(
                    effusion
                )
            rows.append(row)
    return rows


def load_weak_supervision_rows(
    path: Path,
    calibration: CategoryCalibration | None = None,
    impute_synovitis_from_effusion: bool = False,
) -> list[dict[str, str | float | int]]:
    """Return probabilities, independent confidence weights, and global masks."""
    rows: list[dict[str, str | float | int]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                study_id = str(record["study_id"]).strip()
                prediction = decode_prediction(record["prediction"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Invalid JSONL line {line_number}: {error}"
                ) from error
            if not study_id:
                raise ValueError(f"Missing study identifier on JSONL line {line_number}")
            if study_id in seen:
                raise ValueError(f"Duplicate study identifier: {study_id}")
            seen.add(study_id)
            row: dict[str, str | float | int] = {"StudyInstanceUID": study_id}
            row.update(
                {
                    target: calibrated_probability(target, finding, calibration)
                    for target, finding in zip(TARGETS, prediction.findings)
                }
            )
            if (
                impute_synovitis_from_effusion
                and prediction.findings[TARGETS.index("Synovitis")].mention == 0
            ):
                effusion = prediction.findings[TARGETS.index("Effusion")]
                row["Synovitis"] = 0.30 + 0.10 * refined_ordered_probability(
                    effusion
                )
            row.update(
                {
                    f"{target}_confidence_weight": initial_weight(finding)
                    for target, finding in zip(TARGETS, prediction.findings)
                }
            )
            row["mask"] = initial_mask(prediction)
            rows.append(row)
    return rows


def write_soft_label_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    """Write soft labels in the column order required by the training loader."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["StudyInstanceUID", *TARGETS]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_weak_supervision_csv(
    rows: Sequence[Mapping[str, object]], path: Path
) -> None:
    """Write the complete weak-supervision table without conflating its signals."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "StudyInstanceUID",
        *TARGETS,
        *(f"{target}_confidence_weight" for target in TARGETS),
        "mask",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gold-labels",
        type=Path,
        help="Fit target/category shrinkage calibration from fully labeled rows",
    )
    parser.add_argument(
        "--calibration-prior-strength",
        type=float,
        default=5.0,
        help="Ordered-rule pseudo-count used to shrink gold category rates",
    )
    parser.add_argument(
        "--impute-synovitis-from-effusion",
        action="store_true",
        help="Rank not-mentioned Synovitis cases using the Effusion extraction",
    )
    parser.add_argument(
        "--soft-label-only",
        action="store_true",
        help="Write only StudyInstanceUID and the 12 probability columns",
    )
    args = parser.parse_args(argv)
    calibration = None
    if args.gold_labels is not None:
        calibration = fit_category_calibration(
            args.input,
            args.gold_labels,
            args.calibration_prior_strength,
        )
    loader = load_soft_label_rows if args.soft_label_only else load_weak_supervision_rows
    rows = loader(
        args.input,
        calibration,
        args.impute_synovitis_from_effusion,
    )
    if not rows:
        raise ValueError("Input contains no predictions")
    if args.soft_label_only:
        write_soft_label_csv(rows, args.output)
    else:
        write_weak_supervision_csv(rows, args.output)
    print(f"Wrote {len(rows)} weak-label rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
