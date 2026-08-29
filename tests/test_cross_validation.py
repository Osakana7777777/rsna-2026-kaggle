from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from rsna2026.constants import TARGETS
from rsna2026.cross_validation import (
    bootstrap_macro_auc_interval,
    create_inner_roles,
    create_outer_folds,
    weighted_binary_cross_entropy,
)
from rsna2026.public_labels import binary_auc


def macro_auc(labels, predictions) -> float:
    scores = [
        binary_auc(labels[:, index], predictions[:, index])
        for index in range(labels.shape[1])
    ]
    finite = [score for score in scores if np.isfinite(score)]
    return float(np.mean(finite)) if finite else float("nan")


def synthetic_tables() -> tuple[pd.DataFrame, dict[str, tuple[float, ...]]]:
    rows = []
    public = {}
    for index in range(460):
        uid = f"study-{index:03d}"
        is_gold = index < 60
        labels = {
            target: float((index + target_index * 3) % (5 + target_index % 3) == 0)
            if is_gold
            else np.nan
            for target_index, target in enumerate(TARGETS)
        }
        report = "duplicate weak report" if index in (70, 71) else f"report {index}"
        rows.append({"StudyInstanceUID": uid, "Report": report, **labels})
        public[uid] = tuple(
            0.8 if (index + target_index * 3) % (5 + target_index % 3) == 0 else 0.2
            for target_index in range(len(TARGETS))
        )
    return pd.DataFrame(rows), public


def test_outer_folds_are_order_independent_grouped_and_stratified() -> None:
    table, public = synthetic_tables()
    first = create_outer_folds(table, public, seed=42)
    second = create_outer_folds(
        table.sample(frac=1.0, random_state=9).reset_index(drop=True), public, seed=42
    )

    pd.testing.assert_frame_equal(first.assignments, second.assignments)
    assert first.audit["assignment_sha256"] == second.audit["assignment_sha256"]
    assert first.assignments.groupby("report_hash")["outer_fold"].nunique().max() == 1
    assert first.assignments["is_gold"].sum() == 60
    assert first.audit["acceptance"]["passed"] is True


def test_inner_roles_keep_gold_out_of_selection_and_groups_together() -> None:
    table, public = synthetic_tables()
    design = create_outer_folds(table, public, seed=42)

    roles = create_inner_roles(design.assignments, public, 0, 0.2, 1042)

    assert not roles.loc[roles["role"] == "inner-validation", "is_gold"].any()
    assert roles.groupby("report_hash")["role"].nunique().max() == 1
    assert set(roles.loc[roles["role"] == "outer-holdout", "outer_fold"]) == {0}
    assert set(roles["role"]) == {
        "training",
        "inner-validation",
        "outer-holdout",
    }


def test_weighted_bce_and_bootstrap_interval_are_finite() -> None:
    labels = np.asarray([[0.0] * 12, [1.0] * 12, [0.0] * 12, [1.0] * 12])
    predictions = np.asarray(
        [[0.1] * 12, [0.9] * 12, [0.2] * 12, [0.8] * 12]
    )
    weights = np.ones_like(labels)

    assert weighted_binary_cross_entropy(labels, predictions, weights) > 0
    interval = bootstrap_macro_auc_interval(
        labels, predictions, macro_auc, resamples=100, seed=42
    )
    assert 0.0 <= interval["lower_95"] <= interval["upper_95"] <= 1.0


def test_rejects_non_twenty_percent_inner_validation() -> None:
    table, public = synthetic_tables()
    design = create_outer_folds(table, public, seed=42)
    with pytest.raises(ValueError, match="20%"):
        create_inner_roles(design.assignments, public, 0, 0.1, 1042)
