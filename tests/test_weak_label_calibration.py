import csv
import json

import pytest

from rsna2026.weak_label_calibration import (
    fit_category_calibration,
    load_soft_label_rows,
    load_weak_supervision_rows,
    write_soft_label_csv,
    write_weak_supervision_csv,
)
from rsna2026.constants import TARGETS


def finding(mention: int, threshold: int) -> list[object]:
    return [mention, 1 if mention else 0, 0, 0, threshold, [], [], []]


def prediction() -> dict[str, object]:
    values = [
        finding(1, 3),
        finding(2, 4),
        finding(0, 0),
    ]
    values.extend(finding(3, 2) for _ in range(len(TARGETS) - len(values)))
    return {"l": "en", "s": 1, "g": [], "x": values}


def test_writes_training_compatible_soft_label_csv(tmp_path) -> None:
    source = tmp_path / "predictions.jsonl"
    source.write_text(
        json.dumps({"study_id": "study-1", "prediction": prediction()}) + "\n",
        encoding="utf-8",
    )
    rows = load_soft_label_rows(source)
    assert rows[0]["ACL"] == 0.9
    assert rows[0]["MCL"] == 0.1
    assert rows[0]["Medial Meniscus"] == 0.5

    output = tmp_path / "labels.csv"
    write_soft_label_csv(rows, output)
    with output.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == ["StudyInstanceUID", *TARGETS]
        assert next(reader)["StudyInstanceUID"] == "study-1"


def test_writes_confidence_weights_and_mask_separately(tmp_path) -> None:
    source = tmp_path / "predictions.jsonl"
    value = prediction()
    value["g"] = [2]
    source.write_text(
        json.dumps({"study_id": "study-1", "prediction": value}) + "\n",
        encoding="utf-8",
    )
    rows = load_weak_supervision_rows(source)
    assert rows[0]["ACL"] == 0.9
    assert rows[0]["ACL_confidence_weight"] == 1.0
    assert rows[0]["mask"] == 0

    output = tmp_path / "weak-supervision.csv"
    write_weak_supervision_csv(rows, output)
    with output.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == [
            "StudyInstanceUID",
            *TARGETS,
            *(f"{target}_confidence_weight" for target in TARGETS),
            "mask",
        ]


def test_rejects_duplicate_study_identifiers(tmp_path) -> None:
    source = tmp_path / "predictions.jsonl"
    line = json.dumps({"study_id": "duplicate", "prediction": prediction()})
    source.write_text(f"{line}\n{line}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate study identifier"):
        load_soft_label_rows(source)


def test_fits_shrunk_target_category_probabilities(tmp_path) -> None:
    source = tmp_path / "predictions.jsonl"
    source.write_text(
        json.dumps({"study_id": "gold-1", "prediction": prediction()}) + "\n",
        encoding="utf-8",
    )
    gold = tmp_path / "train.csv"
    with gold.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["StudyInstanceUID", *TARGETS])
        writer.writeheader()
        writer.writerow(
            {
                "StudyInstanceUID": "gold-1",
                **{target: int(target == "ACL") for target in TARGETS},
            }
        )

    calibration = fit_category_calibration(source, gold, prior_strength=5.0)
    rows = load_weak_supervision_rows(source, calibration)
    expected = (1.0 + 5.0 * (0.92 + 0.004 * 4.0)) / 6.0
    assert rows[0]["ACL"] == pytest.approx(expected)
    assert rows[0]["ACL_confidence_weight"] == 1.0


def test_imputes_only_unmentioned_synovitis_from_effusion(tmp_path) -> None:
    value = prediction()
    effusion_index = TARGETS.index("Effusion")
    synovitis_index = TARGETS.index("Synovitis")
    value["x"][effusion_index] = finding(1, 3)  # type: ignore[index]
    value["x"][synovitis_index] = finding(0, 0)  # type: ignore[index]
    source = tmp_path / "predictions.jsonl"
    source.write_text(
        json.dumps({"study_id": "study-1", "prediction": value}) + "\n",
        encoding="utf-8",
    )

    original = load_weak_supervision_rows(source)[0]
    imputed = load_weak_supervision_rows(
        source, impute_synovitis_from_effusion=True
    )[0]
    assert original["Synovitis"] == 0.5
    assert imputed["Synovitis"] == pytest.approx(0.30 + 0.10 * 0.936)
    assert imputed["Synovitis_confidence_weight"] == original[
        "Synovitis_confidence_weight"
    ]
    assert imputed["mask"] == original["mask"]
