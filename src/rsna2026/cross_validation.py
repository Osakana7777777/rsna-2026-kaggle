"""Deterministic grouped cross-validation and OOF evaluation helpers."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rsna2026.constants import TARGETS
from rsna2026.public_labels import report_hash, stable_collection_hash


SPLIT_FORMAT_VERSION = 1


def selected_training_folds(cv_config: Mapping[str, object]) -> tuple[int, ...]:
    """Select existing folds without changing the complete split design."""
    fold_count = int(cv_config["folds"])
    requested = cv_config.get("run_folds", list(range(fold_count)))
    if not isinstance(requested, list) or not requested:
        raise ValueError("run_folds must be a non-empty list of fold indices")
    if any(type(fold) is not int or not 0 <= fold < fold_count for fold in requested):
        raise ValueError("run_folds contains an invalid fold index")
    if len(set(requested)) != len(requested):
        raise ValueError("run_folds must not contain duplicate fold indices")
    return tuple(sorted(requested))


@dataclass(frozen=True)
class FoldDesign:
    """Outer assignments and their reproducibility audit."""

    assignments: pd.DataFrame
    audit: dict[str, object]


def dataframe_sha256(table: pd.DataFrame) -> str:
    """Hash a CSV table using its current, explicitly controlled row order."""
    material = table.to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _group_rows(
    table: pd.DataFrame,
    public_rows: Mapping[str, Sequence[float]],
) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    for group_hash, rows in table.groupby("report_hash", sort=True):
        uids = sorted(rows["StudyInstanceUID"].astype(str))
        gold = rows[TARGETS].notna().all(axis=1).to_numpy()
        gold_labels = rows.loc[gold, TARGETS].to_numpy(dtype=np.float64)
        teacher = np.asarray([public_rows[uid] for uid in uids], dtype=np.float64)
        groups.append(
            {
                "report_hash": str(group_hash),
                "uids": uids,
                "study_count": len(uids),
                "gold_count": int(gold.sum()),
                "gold_positive": (
                    gold_labels.sum(axis=0)
                    if len(gold_labels)
                    else np.zeros(len(TARGETS), dtype=np.float64)
                ),
                "gold_pairs": (
                    np.asarray(
                        [
                            gold_labels[:, left] * gold_labels[:, right]
                            for left in range(len(TARGETS))
                            for right in range(left + 1, len(TARGETS))
                        ]
                    ).sum(axis=1)
                    if len(gold_labels)
                    else np.zeros(66, dtype=np.float64)
                ),
                "teacher_mass": teacher.sum(axis=0),
            }
        )
    return groups


def _gold_assignment_score(
    fold_positive: np.ndarray,
    fold_pairs: np.ndarray,
    fold_gold: np.ndarray,
) -> float:
    positive_target = fold_positive.sum(axis=0) / len(fold_gold)
    pair_target = fold_pairs.sum(axis=0) / len(fold_gold)
    positive_scale = np.maximum(positive_target, 1.0)
    pair_scale = np.maximum(pair_target, 1.0)
    return float(
        np.square((fold_positive - positive_target) / positive_scale).sum()
        + 0.25 * np.square((fold_pairs - pair_target) / pair_scale).sum()
        + 0.05 * np.square(fold_gold - fold_gold.mean()).sum()
    )


def _assign_gold_groups(
    groups: Sequence[dict[str, object]], folds: int, seed: int
) -> dict[str, int]:
    gold_groups = [group for group in groups if int(group["gold_count"]) > 0]
    total_gold = sum(int(group["gold_count"]) for group in gold_groups)
    capacities = np.asarray(
        [total_gold // folds + (fold < total_gold % folds) for fold in range(folds)]
    )
    rng = np.random.default_rng(seed)
    best: tuple[float, tuple[int, ...]] | None = None
    best_assignment: dict[str, int] | None = None
    base_order = sorted(
        range(len(gold_groups)),
        key=lambda index: (
            -int(gold_groups[index]["gold_count"]),
            -float(np.asarray(gold_groups[index]["gold_positive"]).sum()),
            str(gold_groups[index]["report_hash"]),
        ),
    )
    for attempt in range(1000):
        if attempt == 0:
            order = base_order
        else:
            jitter = rng.random(len(gold_groups))
            order = sorted(
                range(len(gold_groups)),
                key=lambda index: (
                    -int(gold_groups[index]["gold_count"]), jitter[index]
                ),
            )
        fold_gold = np.zeros(folds, dtype=np.int64)
        fold_positive = np.zeros((folds, len(TARGETS)), dtype=np.float64)
        fold_pairs = np.zeros((folds, 66), dtype=np.float64)
        assignment: dict[str, int] = {}
        feasible = True
        for index in order:
            group = gold_groups[index]
            count = int(group["gold_count"])
            candidates = [
                fold
                for fold in range(folds)
                if fold_gold[fold] + count <= capacities[fold]
            ]
            if not candidates:
                feasible = False
                break
            rng.shuffle(candidates)
            scored = []
            for fold in candidates:
                next_gold = fold_gold.copy()
                next_positive = fold_positive.copy()
                next_pairs = fold_pairs.copy()
                next_gold[fold] += count
                next_positive[fold] += np.asarray(group["gold_positive"])
                next_pairs[fold] += np.asarray(group["gold_pairs"])
                scored.append(
                    (
                        _gold_assignment_score(next_positive, next_pairs, next_gold),
                        int(next_gold[fold]),
                        fold,
                    )
                )
            chosen = min(scored)[2]
            fold_gold[chosen] += count
            fold_positive[chosen] += np.asarray(group["gold_positive"])
            fold_pairs[chosen] += np.asarray(group["gold_pairs"])
            assignment[str(group["report_hash"])] = chosen
        if not feasible or not np.array_equal(fold_gold, capacities):
            continue
        fold_negative = fold_gold[:, None] - fold_positive
        if np.any(fold_positive < 1) or np.any(fold_negative < 1):
            continue
        score = _gold_assignment_score(fold_positive, fold_pairs, fold_gold)
        signature = tuple(assignment[str(group["report_hash"])] for group in gold_groups)
        candidate = (score, signature)
        if best is None or candidate < best:
            best = candidate
            best_assignment = assignment
    if best_assignment is None:
        raise ValueError(
            "Unable to create four grouped gold folds with both classes for every target"
        )
    return best_assignment


def _assign_balanced_groups(
    groups: Sequence[dict[str, object]], folds: int, seed: int
) -> dict[str, int]:
    """Greedily balance study counts and public-teacher probability mass."""
    counts = np.zeros(folds, dtype=np.float64)
    mass = np.zeros((folds, len(TARGETS)), dtype=np.float64)
    target_count = sum(int(group["study_count"]) for group in groups) / folds
    total_mass = sum(
        (np.asarray(group["teacher_mass"]) for group in groups),
        start=np.zeros(len(TARGETS), dtype=np.float64),
    )
    target_mass = total_mass / folds
    assignment: dict[str, int] = {}

    def salted(group_hash: str) -> str:
        return hashlib.sha256(f"{seed}:{group_hash}".encode()).hexdigest()

    ordered = sorted(
        groups,
        key=lambda group: (
            -int(group["study_count"]),
            -float(np.asarray(group["teacher_mass"]).sum()),
            salted(str(group["report_hash"])),
        ),
    )
    for group in ordered:
        size = int(group["study_count"])
        group_mass = np.asarray(group["teacher_mass"])
        choices = []
        for fold in range(folds):
            next_counts = counts.copy()
            next_mass = mass.copy()
            next_counts[fold] += size
            next_mass[fold] += group_mass
            count_error = np.square(
                (next_counts - target_count) / max(target_count, 1.0)
            ).sum()
            mass_error = np.square(
                (next_mass - target_mass) / np.maximum(target_mass, 1.0)
            ).sum()
            choices.append((float(count_error + mass_error), counts[fold], fold))
        chosen = min(choices)[2]
        counts[chosen] += size
        mass[chosen] += group_mass
        assignment[str(group["report_hash"])] = chosen
    return assignment


def create_outer_folds(
    train_table: pd.DataFrame,
    public_rows: Mapping[str, Sequence[float]],
    folds: int = 4,
    seed: int = 42,
    algorithm: str = "report-group-multilabel-v1",
) -> FoldDesign:
    """Create deterministic report-grouped outer folds for every study."""
    if folds != 4:
        raise ValueError("This experiment requires exactly four outer folds")
    table = train_table.copy()
    table["StudyInstanceUID"] = table["StudyInstanceUID"].astype(str)
    if set(table["StudyInstanceUID"]) != set(public_rows):
        raise ValueError("Competition and public-label UID sets differ")
    table["report_hash"] = table["Report"].fillna("").astype(str).map(report_hash)
    table["is_gold"] = table[TARGETS].notna().all(axis=1)
    groups = _group_rows(table, public_rows)
    assignment = _assign_gold_groups(groups, folds, seed)
    weak_groups = [group for group in groups if int(group["gold_count"]) == 0]
    assignment.update(_assign_balanced_groups(weak_groups, folds, seed + 1))
    table["outer_fold"] = table["report_hash"].map(assignment).astype(int)
    group_sizes = table.groupby("report_hash").size()
    result = table[
        ["StudyInstanceUID", "report_hash", "is_gold", "outer_fold"]
    ].copy()
    result.insert(2, "report_group_size", result["report_hash"].map(group_sizes))
    result = result.sort_values("StudyInstanceUID").reset_index(drop=True)

    per_fold: dict[str, object] = {}
    acceptance = {
        "all_studies_assigned_once": len(result) == len(table)
        and result["StudyInstanceUID"].is_unique,
        "report_groups_do_not_cross_folds": bool(
            (result.groupby("report_hash")["outer_fold"].nunique() == 1).all()
        ),
        "all_gold_studies_assigned_once": int(result["is_gold"].sum())
        == int(table["is_gold"].sum()),
        "every_gold_fold_has_both_classes": True,
    }
    for fold in range(folds):
        fold_uids = set(result.loc[result["outer_fold"] == fold, "StudyInstanceUID"])
        rows = table.loc[table["StudyInstanceUID"].isin(fold_uids)]
        gold = rows.loc[rows["is_gold"]]
        positive = gold[TARGETS].sum(axis=0).astype(int)
        negative = len(gold) - positive
        if bool((positive < 1).any() or (negative < 1).any()):
            acceptance["every_gold_fold_has_both_classes"] = False
        teacher = np.asarray(
            [public_rows[uid] for uid in rows["StudyInstanceUID"]], dtype=np.float64
        )
        per_fold[str(fold)] = {
            "studies": len(rows),
            "report_groups": int(rows["report_hash"].nunique()),
            "gold_studies": len(gold),
            "weak_studies": len(rows) - len(gold),
            "gold_positive_by_target": dict(zip(TARGETS, positive.tolist())),
            "gold_negative_by_target": dict(zip(TARGETS, negative.tolist())),
            "teacher_probability_sum_by_target": dict(
                zip(TARGETS, teacher.sum(axis=0).tolist())
            ),
            "teacher_probability_mean_by_target": dict(
                zip(TARGETS, teacher.mean(axis=0).tolist())
            ),
        }
    acceptance["passed"] = all(acceptance.values())
    if not acceptance["passed"]:
        raise ValueError(f"Outer fold acceptance failed: {acceptance}")
    audit = {
        "format_version": SPLIT_FORMAT_VERSION,
        "algorithm": algorithm,
        "report_normalization": "NFKC, whitespace-collapse, casefold, SHA-256",
        "outer_seed": seed,
        "folds": folds,
        "study_uid_set_sha256": stable_collection_hash(result["StudyInstanceUID"]),
        "assignment_sha256": dataframe_sha256(result),
        "studies": len(result),
        "report_groups": int(result["report_hash"].nunique()),
        "duplicate_report_groups": int((group_sizes > 1).sum()),
        "gold_studies": int(result["is_gold"].sum()),
        "per_fold": per_fold,
        "acceptance": acceptance,
    }
    return FoldDesign(result, audit)


def create_inner_roles(
    assignments: pd.DataFrame,
    public_rows: Mapping[str, Sequence[float]],
    outer_fold: int,
    fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Assign weak-only report groups to a grouped inner validation fifth."""
    if not 0.0 < fraction < 1.0 or not math.isclose(fraction, 0.2):
        raise ValueError("This experiment requires a 20% inner validation fraction")
    rows = assignments.copy()
    rows["role"] = "training"
    rows.loc[rows["outer_fold"] == outer_fold, "role"] = "outer-holdout"
    outer_training = rows.loc[rows["outer_fold"] != outer_fold]
    gold_hashes = set(outer_training.loc[outer_training["is_gold"], "report_hash"])
    candidates = outer_training.loc[~outer_training["report_hash"].isin(gold_hashes)]
    groups = []
    for group_hash, group_rows in candidates.groupby("report_hash", sort=True):
        uids = group_rows["StudyInstanceUID"].astype(str).tolist()
        teacher = np.asarray([public_rows[uid] for uid in uids], dtype=np.float64)
        groups.append(
            {
                "report_hash": str(group_hash),
                "study_count": len(uids),
                "teacher_mass": teacher.sum(axis=0),
            }
        )
    inner_assignment = _assign_balanced_groups(groups, 5, seed + outer_fold)
    validation_hashes = {
        group_hash for group_hash, fold in inner_assignment.items() if fold == 0
    }
    rows.loc[
        (rows["outer_fold"] != outer_fold)
        & rows["report_hash"].isin(validation_hashes),
        "role",
    ] = "inner-validation"
    inner = rows.loc[rows["role"] == "inner-validation"]
    labels = np.asarray(
        [public_rows[uid] for uid in inner["StudyInstanceUID"]], dtype=np.float64
    )
    usable = labels != 0.5
    positives = ((labels > 0.5) & usable).sum(axis=0)
    negatives = ((labels < 0.5) & usable).sum(axis=0)
    if np.any(positives < 1) or np.any(negatives < 1):
        raise ValueError(
            f"Inner fold {outer_fold} lacks a teacher class for at least one target"
        )
    if rows.groupby("report_hash")["role"].nunique().max() != 1:
        raise ValueError("A report group crosses roles in an inner split")
    rows.insert(0, "model_fold", outer_fold)
    return rows[
        [
            "model_fold",
            "StudyInstanceUID",
            "report_hash",
            "is_gold",
            "outer_fold",
            "role",
        ]
    ].sort_values("StudyInstanceUID").reset_index(drop=True)


def weighted_binary_cross_entropy(
    targets: np.ndarray, predictions: np.ndarray, weights: np.ndarray
) -> float:
    """Return confidence-weighted BCE for probability predictions."""
    valid = np.isfinite(targets) & np.isfinite(predictions) & (weights > 0)
    clipped = np.clip(predictions, 1e-7, 1.0 - 1e-7)
    losses = -(targets * np.log(clipped) + (1.0 - targets) * np.log(1.0 - clipped))
    return float((losses[valid] * weights[valid]).sum() / weights[valid].sum())


def bootstrap_macro_auc_interval(
    labels: np.ndarray,
    predictions: np.ndarray,
    auc_function,
    resamples: int,
    seed: int,
) -> dict[str, object]:
    """Bootstrap studies with replacement and summarize macro AUC uncertainty."""
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(resamples):
        indices = rng.integers(0, len(labels), len(labels))
        score = float(auc_function(labels[indices], predictions[indices]))
        if math.isfinite(score):
            scores.append(score)
    if not scores:
        raise ValueError("No finite bootstrap macro AUC values were produced")
    return {
        "unit": "study",
        "requested_resamples": resamples,
        "finite_resamples": len(scores),
        "seed": seed,
        "lower_95": float(np.quantile(scores, 0.025)),
        "median": float(np.quantile(scores, 0.5)),
        "upper_95": float(np.quantile(scores, 0.975)),
    }
