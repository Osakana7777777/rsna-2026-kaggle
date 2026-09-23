"""Evaluate report weak-label ranking rules against the 58 gold studies."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold

from rsna2026.constants import TARGETS
from rsna2026.public_labels import binary_auc
from rsna2026.weak_label_decode import competition_call, decode_prediction


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--reference-labels", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def load_predictions(path: Path) -> dict[str, object]:
    predictions: dict[str, object] = {}
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            study_id = str(record["study_id"])
            if study_id in predictions:
                raise ValueError(f"Duplicate study ID on line {line_number}: {study_id}")
            predictions[study_id] = decode_prediction(record["prediction"])
    return predictions


def current_probability(finding: object) -> float:
    return {"negative": 0.1, "uncertain": 0.5, "positive": 0.9}[
        competition_call(finding)  # type: ignore[arg-type]
    ]


def ordered_probability(finding: object) -> float:
    mention = finding.mention  # type: ignore[attr-defined]
    threshold = finding.threshold  # type: ignore[attr-defined]
    if mention == 2:
        return 0.03
    if mention == 0:
        return 0.12
    if mention == 3:
        return 0.38
    return {0: 0.52, 1: 0.22, 2: 0.68, 3: 0.92, 4: 0.32}[threshold]


def refined_ordered_probability(finding: object) -> float:
    """Break ordered-rule ties using report evidence strength."""
    score = ordered_probability(finding)
    mention = finding.mention  # type: ignore[attr-defined]
    if mention not in (1, 3):
        return score
    certainty_rank = {0: 0.0, 1: 4.0, 2: 3.0, 3: 2.0, 4: 1.0}
    score += 0.004 * certainty_rank[finding.certainty]  # type: ignore[attr-defined]
    score += 0.0005 * finding.severity  # type: ignore[attr-defined]
    return score


def finding_features(prediction: object, target_index: int) -> np.ndarray:
    finding = prediction.findings[target_index]  # type: ignore[attr-defined]
    values: list[float] = []
    for value, size in (
        (finding.mention, 4),
        (finding.certainty, 5),
        (finding.severity, 13),
        (finding.acuity, 7),
        (finding.threshold, 5),
    ):
        values.extend(float(value == index) for index in range(size))
    values.extend(float(code in finding.features) for code in range(1, 30))
    values.extend(float(code in finding.review_reasons) for code in range(1, 10))
    values.extend(
        float(code in prediction.global_flags)  # type: ignore[attr-defined]
        for code in range(1, 6)
    )
    values.extend(
        [
            len(finding.evidence) / 2.0,
            float(finding.needs_review),
        ]
    )
    return np.asarray(values, dtype=np.float64)


def auc_summary(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[float, dict[str, float]]:
    per_target = {
        target: binary_auc(labels[:, index], scores[:, index])
        for index, target in enumerate(TARGETS)
    }
    finite = [value for value in per_target.values() if np.isfinite(value)]
    return float(np.mean(finite)), per_target


def targetwise_oof(
    features: list[np.ndarray], labels: np.ndarray, regularization: float
) -> np.ndarray:
    scores = np.zeros_like(labels, dtype=np.float64)
    for target_index in range(len(TARGETS)):
        x = features[target_index]
        y = labels[:, target_index].astype(np.int64)
        splits = min(5, int(y.sum()), int((1 - y).sum()))
        if splits < 2:
            scores[:, target_index] = float(y.mean())
            continue
        fold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
        for train_index, valid_index in fold.split(x, y):
            model = LogisticRegression(
                C=regularization,
                class_weight="balanced",
                max_iter=5000,
                solver="liblinear",
            )
            model.fit(x[train_index], y[train_index])
            scores[valid_index, target_index] = model.predict_proba(
                x[valid_index]
            )[:, 1]
    return scores


def pooled_features(
    predictions: list[object], base_features: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    groups: list[int] = []
    for study_index, prediction in enumerate(predictions):
        for target_index in range(len(TARGETS)):
            target = np.zeros(len(TARGETS), dtype=np.float64)
            target[target_index] = 1.0
            base = base_features[target_index][study_index]
            interaction = np.concatenate(
                [
                    target * prediction.findings[target_index].mention,
                    target * prediction.findings[target_index].threshold,
                    target * prediction.findings[target_index].severity,
                ]
            )
            rows.append(np.concatenate([target, base, interaction]))
            groups.append(study_index)
    return np.stack(rows), np.asarray(groups)


def pooled_oof(
    x: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    regularization: float,
) -> np.ndarray:
    flat_labels = labels.reshape(-1)
    flat_scores = np.zeros_like(flat_labels, dtype=np.float64)
    fold = GroupKFold(n_splits=5)
    for train_index, valid_index in fold.split(x, flat_labels, groups):
        model = LogisticRegression(
            C=regularization,
            class_weight="balanced",
            max_iter=5000,
            solver="liblinear",
        )
        model.fit(x[train_index], flat_labels[train_index])
        flat_scores[valid_index] = model.predict_proba(x[valid_index])[:, 1]
    return flat_scores.reshape(labels.shape)


def category_calibration_scores(
    predictions: list[object],
    labels: np.ndarray,
    prior_strength: float,
    out_of_fold: bool,
) -> np.ndarray:
    """Calibrate mention/threshold categories toward the ordered-rule prior."""
    scores = np.zeros_like(labels, dtype=np.float64)
    for target_index in range(len(TARGETS)):
        y = labels[:, target_index].astype(np.int64)
        if out_of_fold:
            splits = min(5, int(y.sum()), int((1 - y).sum()))
            fold_indices = StratifiedKFold(
                n_splits=splits, shuffle=True, random_state=42
            ).split(np.zeros(len(y)), y)
        else:
            all_indices = np.arange(len(y))
            fold_indices = [(all_indices, all_indices)]
        for train_index, valid_index in fold_indices:
            categories: dict[tuple[int, int], list[int]] = {}
            for index in train_index:
                finding = predictions[index].findings[target_index]
                categories.setdefault(
                    (finding.mention, finding.threshold), []
                ).append(int(y[index]))
            for index in valid_index:
                finding = predictions[index].findings[target_index]
                values = categories.get((finding.mention, finding.threshold), [])
                prior = refined_ordered_probability(finding)
                scores[index, target_index] = (
                    sum(values) + prior_strength * prior
                ) / (len(values) + prior_strength)
    return scores


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    train = pd.read_csv(args.train_csv)
    train["StudyInstanceUID"] = train["StudyInstanceUID"].astype(str)
    gold = train.loc[train[TARGETS].notna().all(axis=1)].reset_index(drop=True)
    predictions_by_id = load_predictions(args.predictions)
    predictions = [predictions_by_id[study_id] for study_id in gold.StudyInstanceUID]
    labels = gold[TARGETS].to_numpy(dtype=np.float64)

    current = np.asarray(
        [
            [current_probability(finding) for finding in prediction.findings]
            for prediction in predictions
        ]
    )
    ordered = np.asarray(
        [
            [ordered_probability(finding) for finding in prediction.findings]
            for prediction in predictions
        ]
    )
    refined = np.asarray(
        [
            [
                refined_ordered_probability(finding)
                for finding in prediction.findings
            ]
            for prediction in predictions
        ]
    )
    base_features = [
        np.stack(
            [finding_features(prediction, target_index) for prediction in predictions]
        )
        for target_index in range(len(TARGETS))
    ]
    pooled_x, groups = pooled_features(predictions, base_features)

    result: dict[str, object] = {
        "gold_studies": len(gold),
        "positive_counts": {
            target: int(labels[:, index].sum())
            for index, target in enumerate(TARGETS)
        },
        "methods": {},
    }
    methods = result["methods"]
    assert isinstance(methods, dict)
    for name, scores in (
        ("current_three_level", current),
        ("ordered_rule", ordered),
        ("refined_ordered_rule", refined),
    ):
        macro, per_target = auc_summary(labels, scores)
        methods[name] = {"macro_auc": macro, "per_target_auc": per_target}

    if args.reference_labels is not None:
        reference_table = pd.read_csv(args.reference_labels)
        reference_table["StudyInstanceUID"] = reference_table[
            "StudyInstanceUID"
        ].astype(str)
        reference_by_id = reference_table.set_index("StudyInstanceUID")
        reference = reference_by_id.loc[gold.StudyInstanceUID, TARGETS].to_numpy(
            dtype=np.float64
        )
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
            scores = alpha * refined + (1.0 - alpha) * reference
            macro, per_target = auc_summary(labels, scores)
            methods[f"refined_reference_blend_alpha={alpha}"] = {
                "macro_auc": macro,
                "per_target_auc": per_target,
            }

    category_statistics: dict[str, list[dict[str, object]]] = {}
    for target_index, target in enumerate(TARGETS):
        categories: dict[tuple[int, int], list[int]] = {}
        for prediction, label in zip(predictions, labels[:, target_index]):
            finding = prediction.findings[target_index]
            categories.setdefault((finding.mention, finding.threshold), []).append(
                int(label)
            )
        category_statistics[target] = [
            {
                "mention": mention,
                "threshold": threshold,
                "count": len(values),
                "positives": sum(values),
                "positive_rate": sum(values) / len(values),
            }
            for (mention, threshold), values in sorted(categories.items())
        ]
    result["mention_threshold_statistics"] = category_statistics

    for prior_strength in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0):
        for out_of_fold in (False, True):
            scores = category_calibration_scores(
                predictions, labels, prior_strength, out_of_fold
            )
            macro, per_target = auc_summary(labels, scores)
            split_name = "oof" if out_of_fold else "fit"
            methods[
                f"mention_threshold_calibration_{split_name}_prior={prior_strength}"
            ] = {"macro_auc": macro, "per_target_auc": per_target}

    pooled_scores_by_regularization: dict[float, np.ndarray] = {}
    for regularization in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
        target_scores = targetwise_oof(base_features, labels, regularization)
        macro, per_target = auc_summary(labels, target_scores)
        methods[f"targetwise_logistic_oof_c={regularization}"] = {
            "macro_auc": macro,
            "per_target_auc": per_target,
        }
        pooled_scores = pooled_oof(pooled_x, labels, groups, regularization)
        pooled_scores_by_regularization[regularization] = pooled_scores
        macro, per_target = auc_summary(labels, pooled_scores)
        methods[f"pooled_logistic_oof_c={regularization}"] = {
            "macro_auc": macro,
            "per_target_auc": per_target,
        }

    for alpha in (0.25, 0.5, 0.75):
        scores = (
            alpha * refined
            + (1.0 - alpha) * pooled_scores_by_regularization[0.1]
        )
        macro, per_target = auc_summary(labels, scores)
        methods[f"refined_pooled_oof_blend_alpha={alpha}"] = {
            "macro_auc": macro,
            "per_target_auc": per_target,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
