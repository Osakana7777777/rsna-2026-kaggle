import csv
import math

import pytest

from rsna2026.constants import TARGETS
from rsna2026.public_labels import (
    binary_auc,
    confidence_weight,
    file_sha256,
    load_public_label_table,
    report_hash,
    source_validation,
    stable_collection_hash,
    stable_sample,
)


def write_table(path, rows, columns=None) -> None:
    fieldnames = columns or ["StudyInstanceUID", *TARGETS]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def label_row(study_uid: str, value: float) -> dict[str, object]:
    return {
        "StudyInstanceUID": study_uid,
        **{target: value for target in TARGETS},
    }


def test_loads_exact_soft_label_schema_and_preserves_probabilities(tmp_path) -> None:
    path = tmp_path / "labels.csv"
    write_table(path, [label_row("a", 0.25), label_row("b", 0.75)])
    table = load_public_label_table(path, file_sha256(path), expected_rows=2)
    assert table.rows["a"] == (0.25,) * len(TARGETS)
    assert table.rows["b"] == (0.75,) * len(TARGETS)
    assert table.confidence_weights is None
    assert table.masks is None


def test_loads_explicit_confidence_weights_and_masks(tmp_path) -> None:
    path = tmp_path / "weak-supervision.csv"
    columns = [
        "StudyInstanceUID",
        *TARGETS,
        *(f"{target}_confidence_weight" for target in TARGETS),
        "mask",
    ]
    row = label_row("a", 0.9)
    row.update({f"{target}_confidence_weight": 0.7 for target in TARGETS})
    row["mask"] = 0
    write_table(path, [row], columns)

    table = load_public_label_table(path, file_sha256(path), expected_rows=1)
    assert table.rows["a"] == (0.9,) * len(TARGETS)
    assert table.confidence_weights == {"a": (0.7,) * len(TARGETS)}
    assert table.masks == {"a": 0}


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf")])
def test_rejects_nonfinite_or_out_of_range_values(tmp_path, value) -> None:
    path = tmp_path / "labels.csv"
    write_table(path, [label_row("a", value)])
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        load_public_label_table(path, file_sha256(path), expected_rows=1)


def test_rejects_hash_schema_row_count_and_duplicate_drift(tmp_path) -> None:
    path = tmp_path / "labels.csv"
    write_table(path, [label_row("a", 0.5)])
    digest = file_sha256(path)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_public_label_table(path, "0" * 64, expected_rows=1)
    with pytest.raises(ValueError, match="Expected 2"):
        load_public_label_table(path, digest, expected_rows=2)

    duplicate = tmp_path / "duplicate.csv"
    write_table(duplicate, [label_row("a", 0.1), label_row("a", 0.9)])
    with pytest.raises(ValueError, match="duplicate"):
        load_public_label_table(duplicate, file_sha256(duplicate), 2)

    wrong_schema = tmp_path / "wrong.csv"
    write_table(wrong_schema, [{"StudyInstanceUID": "a"}], ["StudyInstanceUID"])
    with pytest.raises(ValueError, match="columns"):
        load_public_label_table(wrong_schema, file_sha256(wrong_schema), 1)


def test_confidence_weight_uses_distance_from_undecided() -> None:
    assert confidence_weight(0.0) == 2.0
    assert confidence_weight(0.25) == pytest.approx(1.15)
    assert confidence_weight(0.5) == 0.30
    assert confidence_weight(0.75) == pytest.approx(1.15)
    assert confidence_weight(1.0) == 2.0


@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0.0, 1.0), (0.25, 0.5), (0.5, 0.0), (0.75, 0.5), (1.0, 1.0)],
)
def test_linear_confidence_weight(probability: float, expected: float) -> None:
    assert confidence_weight(probability, "linear-distance") == expected


def test_rejects_unknown_confidence_weight_method() -> None:
    with pytest.raises(ValueError, match="Unsupported confidence weight method"):
        confidence_weight(0.75, "unknown")


def test_report_hash_normalizes_case_spacing_and_unicode() -> None:
    assert report_hash("  ACL  INTACT ") == report_hash("acl intact")
    assert report_hash("ＡＣＬ") == report_hash("acl")


def test_sampling_and_collection_hash_are_order_independent() -> None:
    values = ["study-c", "study-a", "study-b", "study-d"]
    assert stable_sample(values, 3, 42) == stable_sample(reversed(values), 3, 42)
    assert stable_collection_hash(values) == stable_collection_hash(reversed(values))
    with pytest.raises(ValueError, match="Need 5"):
        stable_sample(values, 5, 42)


def test_source_validation_reports_all_targets_and_macro_auc() -> None:
    negative = {"StudyInstanceUID": "negative", **{target: 0 for target in TARGETS}}
    positive = {"StudyInstanceUID": "positive", **{target: 1 for target in TARGETS}}
    rows = {
        "negative": (0.1,) * len(TARGETS),
        "positive": (0.9,) * len(TARGETS),
    }
    metrics = source_validation([negative, positive], rows)
    assert metrics["gold_studies"] == 2
    assert metrics["matched_gold_studies"] == 2
    assert metrics["scored_targets"] == len(TARGETS)
    assert metrics["macro_auc"] == 1.0
    assert binary_auc([0, 1], [0.5, 0.5]) == 0.5
    assert math.isnan(binary_auc([1, 1], [0.1, 0.9]))
