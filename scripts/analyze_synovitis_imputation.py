"""Test whether other report findings improve Synovitis weak labels."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from rsna2026.constants import TARGETS
from rsna2026.public_labels import binary_auc
from rsna2026.weak_label_calibration import (
    calibrated_probability,
    fit_category_calibration,
    refined_ordered_probability,
)
from rsna2026.weak_label_decode import decode_prediction


SYNOVITIS_INDEX = TARGETS.index("Synovitis")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def load_predictions(path: Path) -> dict[str, object]:
    predictions: dict[str, object] = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            study_id = str(record["study_id"])
            if study_id in predictions:
                raise ValueError(f"Duplicate study identifier: {study_id}")
            predictions[study_id] = decode_prediction(record["prediction"])
    return predictions


def synovitis_features(prediction: object, include_other_targets: bool) -> np.ndarray:
    synovitis = prediction.findings[SYNOVITIS_INDEX]  # type: ignore[attr-defined]
    values: list[float] = []
    for value, size in (
        (synovitis.mention, 4),
        (synovitis.certainty, 5),
        (synovitis.severity, 13),
        (synovitis.acuity, 7),
        (synovitis.threshold, 5),
    ):
        values.extend(float(value == index) for index in range(size))
    values.extend(float(code in synovitis.features) for code in range(1, 30))
    values.extend(float(code in synovitis.review_reasons) for code in range(1, 10))
    values.append(refined_ordered_probability(synovitis))
    if include_other_targets:
        values.extend(
            refined_ordered_probability(finding)
            for index, finding in enumerate(prediction.findings)  # type: ignore[attr-defined]
            if index != SYNOVITIS_INDEX
        )
    return np.asarray(values, dtype=np.float64)


def logistic_oof_auc(x: np.ndarray, labels: np.ndarray, c: float) -> float:
    scores = np.zeros(len(labels), dtype=np.float64)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, valid_index in folds.split(x, labels):
        model = LogisticRegression(
            C=c,
            class_weight="balanced",
            max_iter=5000,
            solver="liblinear",
        )
        model.fit(x[train_index], labels[train_index])
        scores[valid_index] = model.predict_proba(x[valid_index])[:, 1]
    return binary_auc(labels, scores)


def bootstrap_auc_difference(
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    resamples: int = 20_000,
) -> dict[str, float | int]:
    """Return a paired study-bootstrap interval for the AUC improvement."""
    generator = np.random.default_rng(42)
    differences: list[float] = []
    for _ in range(resamples):
        indices = generator.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[indices]
        if sampled_labels.min() == sampled_labels.max():
            continue
        differences.append(
            binary_auc(sampled_labels, candidate[indices])
            - binary_auc(sampled_labels, baseline[indices])
        )
    values = np.asarray(differences, dtype=np.float64)
    return {
        "valid_resamples": len(values),
        "mean_difference": float(values.mean()),
        "ci_2_5": float(np.quantile(values, 0.025)),
        "ci_97_5": float(np.quantile(values, 0.975)),
        "probability_improvement": float(np.mean(values > 0.0)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    train = pd.read_csv(args.train_csv)
    train["StudyInstanceUID"] = train["StudyInstanceUID"].astype(str)
    gold = train.loc[train[TARGETS].notna().all(axis=1)].reset_index(drop=True)
    predictions_by_id = load_predictions(args.predictions)
    predictions = [predictions_by_id[study_id] for study_id in gold.StudyInstanceUID]
    labels = gold["Synovitis"].to_numpy(dtype=np.int64)
    extracted = np.asarray(
        [
            [refined_ordered_probability(finding) for finding in prediction.findings]
            for prediction in predictions
        ],
        dtype=np.float64,
    )
    direct = extracted[:, SYNOVITIS_INDEX]
    missing = np.asarray(
        [prediction.findings[SYNOVITIS_INDEX].mention == 0 for prediction in predictions]
    )

    result: dict[str, object] = {
        "gold_studies": len(gold),
        "synovitis_positives": int(labels.sum()),
        "synovitis_not_mentioned": int(missing.sum()),
        "synovitis_positive_when_not_mentioned": int(labels[missing].sum()),
        "direct_auc": binary_auc(labels, direct),
        "proxy_results": {},
        "composite_results": {},
        "logistic_oof": {},
    }
    proxy_results = result["proxy_results"]
    assert isinstance(proxy_results, dict)
    for target_index, target in enumerate(TARGETS):
        if target == "Synovitis":
            continue
        proxy = extracted[:, target_index]
        gold_proxy = gold[target].to_numpy(dtype=np.float64)
        blends = {}
        missing_only = {}
        for alpha in (0.1, 0.25, 0.5, 1.0, 2.0):
            blends[str(alpha)] = binary_auc(labels, direct + alpha * proxy)
            imputed = direct.copy()
            imputed[missing] = 0.12 + alpha * proxy[missing]
            missing_only[str(alpha)] = binary_auc(labels, imputed)
        proxy_results[target] = {
            "gold_phi": float(np.corrcoef(labels, gold_proxy)[0, 1]),
            "gold_proxy_auc": binary_auc(labels, gold_proxy),
            "gold_proxy_missing_subset_auc": binary_auc(
                labels[missing], gold_proxy[missing]
            ),
            "extracted_proxy_auc": binary_auc(labels, proxy),
            "missing_subset_auc": binary_auc(labels[missing], proxy[missing]),
            "direct_plus_proxy_auc": blends,
            "missing_only_imputation_auc": missing_only,
        }

    proxy_indices = {
        "effusion": [TARGETS.index("Effusion")],
        "effusion_bakers": [TARGETS.index("Effusion"), TARGETS.index("Baker's")],
        "inflammatory_context": [
            TARGETS.index("Effusion"),
            TARGETS.index("Baker's"),
            TARGETS.index("Medial OA"),
            TARGETS.index("Lateral OA"),
            TARGETS.index("PF OA"),
        ],
        "all_other_targets": [
            index for index in range(len(TARGETS)) if index != SYNOVITIS_INDEX
        ],
    }
    composite_results = result["composite_results"]
    assert isinstance(composite_results, dict)
    for name, indices in proxy_indices.items():
        proxy = extracted[:, indices].mean(axis=1)
        composite_results[name] = {
            "proxy_auc": binary_auc(labels, proxy),
            "missing_subset_auc": binary_auc(labels[missing], proxy[missing]),
            "direct_plus_proxy_auc": {
                str(alpha): binary_auc(labels, direct + alpha * proxy)
                for alpha in (0.1, 0.25, 0.5, 1.0, 2.0)
            },
        }

    calibration = fit_category_calibration(
        args.predictions, args.train_csv, prior_strength=5.0
    )
    calibrated_direct = np.asarray(
        [
            calibrated_probability(
                "Synovitis",
                prediction.findings[SYNOVITIS_INDEX],
                calibration,
            )
            for prediction in predictions
        ]
    )
    effusion = extracted[:, TARGETS.index("Effusion")]
    effusion_imputed = calibrated_direct.copy()
    effusion_imputed[missing] = 0.30 + 0.1 * effusion[missing]
    result["recommended_rule"] = {
        "description": (
            "Keep explicit Synovitis scores; rank not-mentioned cases by "
            "0.30 + 0.10 * refined Effusion probability"
        ),
        "auc": binary_auc(labels, effusion_imputed),
        "baseline_auc": binary_auc(labels, calibrated_direct),
        "auc_difference": binary_auc(labels, effusion_imputed)
        - binary_auc(labels, calibrated_direct),
        "paired_bootstrap": bootstrap_auc_difference(
            labels, calibrated_direct, effusion_imputed
        ),
    }

    direct_features = np.stack(
        [synovitis_features(prediction, False) for prediction in predictions]
    )
    all_features = np.stack(
        [synovitis_features(prediction, True) for prediction in predictions]
    )
    logistic_results = result["logistic_oof"]
    assert isinstance(logistic_results, dict)
    for c in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
        logistic_results[str(c)] = {
            "synovitis_only_auc": logistic_oof_auc(direct_features, labels, c),
            "with_other_findings_auc": logistic_oof_auc(all_features, labels, c),
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
