import copy
import csv
import json
import math
from pathlib import Path

import pytest

from rsna2026.constants import TARGETS
from rsna2026.report_labels import (
    CERTAINTIES,
    LABEL_STATES,
    average_precision,
    build_audit,
    empty_report_labels,
    evidence_matches_report,
    group_reports,
    load_label_config,
    output_json_schema,
    paired_bootstrap_auc_difference,
    probability_for,
    prompt_hash,
    read_train_csv,
    report_hash,
    validate_checkpoint_metadata,
    validate_response,
)

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "report-labels.toml"


def valid_response() -> dict[str, object]:
    return {
        target: {
            "state": "not_mentioned",
            "certainty": "high",
            "evidence": None,
            "reason": "The target is not discussed in the report.",
        }
        for target in TARGETS
    }


def test_schema_requires_exact_targets_and_finding_fields() -> None:
    schema = output_json_schema()
    assert schema["additionalProperties"] is False
    assert schema["required"] == TARGETS
    finding = schema["properties"]["ACL"]
    assert finding["additionalProperties"] is False
    assert finding["required"] == ["state", "certainty", "evidence", "reason"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.pop("ACL"), "missing"),
        (lambda payload: payload.update({"extra": {}}), "extra"),
        (
            lambda payload: payload["ACL"].update({"state": "maybe"}),
            "invalid state",
        ),
        (
            lambda payload: payload["ACL"].update({"certainty": "certain"}),
            "invalid certainty",
        ),
    ],
)
def test_rejects_missing_extra_and_invalid_enumerations(mutation, message) -> None:
    payload = valid_response()
    mutation(payload)
    with pytest.raises(ValueError, match=message):
        validate_response(payload, "Normal knee MRI.")


def test_evidence_may_differ_from_report_but_null_rule_remains() -> None:
    report = "Complete ACL tear is present."
    payload = valid_response()
    payload["ACL"] = {
        "state": "positive",
        "certainty": "high",
        "evidence": "Complete ACL tear",
        "reason": "A complete tear meets the challenge threshold.",
    }
    parsed = validate_response(json.dumps(payload), report)
    assert parsed["ACL"]["evidence"] == "Complete ACL tear"

    payload["ACL"]["evidence"] = "complete ACL tear"
    parsed = validate_response(payload, report)
    assert parsed["ACL"]["evidence"] == "complete ACL tear"
    assert evidence_matches_report("complete ACL tear", report)
    assert evidence_matches_report("Complete ACL—tear", report)
    assert not evidence_matches_report("ACL reconstruction graft", report)

    payload["ACL"]["evidence"] = "   "
    with pytest.raises(ValueError, match="non-empty"):
        validate_response(payload, report)

    payload = valid_response()
    payload["MCL"]["evidence"] = "MCL"
    with pytest.raises(ValueError, match="must be null"):
        validate_response(payload, report)


def test_empty_report_has_complete_deterministic_labels() -> None:
    labels = empty_report_labels()
    assert set(labels) == set(TARGETS)
    assert all(value["state"] == "not_mentioned" for value in labels.values())
    assert all(value["evidence"] is None for value in labels.values())


def test_all_probability_anchors_are_frozen() -> None:
    config = load_label_config(CONFIG_PATH)
    expected = {
        "positive": {"high": 0.95, "moderate": 0.80, "low": 0.65},
        "negative": {"high": 0.05, "moderate": 0.20, "low": 0.35},
        "uncertain": {certainty: 0.50 for certainty in CERTAINTIES},
        "not_mentioned": {certainty: 0.50 for certainty in CERTAINTIES},
    }
    for state in LABEL_STATES:
        for certainty in CERTAINTIES:
            assert probability_for(state, certainty, config) == expected[state][certainty]
    assert len(prompt_hash()) == 64


def _write_train_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["StudyInstanceUID", "Report", *TARGETS])
        writer.writeheader()
        writer.writerows(rows)


def test_transport_projection_never_contains_gold_labels(tmp_path) -> None:
    path = tmp_path / "train.csv"
    row = {"StudyInstanceUID": "1.2.3", "Report": "ACL intact.\nMCL intact."}
    row.update({target: 1 for target in TARGETS})
    _write_train_csv(path, [row])
    source, transport, encoding = read_train_csv(path, expected_studies=1)
    assert source[0]["ACL"] == "1"
    assert transport == [
        {"StudyInstanceUID": "1.2.3", "Report": "ACL intact.\nMCL intact."}
    ]
    assert set(transport[0]) == {"StudyInstanceUID", "Report"}
    assert encoding == "utf-8-sig"


def test_duplicate_reports_are_inferred_once_and_fanned_out() -> None:
    reports = [
        {"StudyInstanceUID": "b", "Report": "  ACL INTACT  "},
        {"StudyInstanceUID": "a", "Report": "acl intact"},
        {"StudyInstanceUID": "c", "Report": "MCL tear"},
    ]
    unique, members = group_reports(reports)
    assert len(unique) == 2
    assert members[report_hash("acl intact")] == ["a", "b"]


def test_resume_rejects_any_signature_change() -> None:
    validate_checkpoint_metadata({"signature": "same"}, "same")
    with pytest.raises(ValueError, match="different input, model, or prompt"):
        validate_checkpoint_metadata({"signature": "old"}, "new")


def _audit_fixture():
    train_rows = []
    structured = []
    public = {}
    for index, label in enumerate((0, 1, 0, 1)):
        uid = f"study-{index}"
        train = {"StudyInstanceUID": uid, "Report": "finding"}
        train.update({target: str(label) for target in TARGETS})
        train_rows.append(train)
        state = "positive" if label else "negative"
        findings = {
            target: {
                "state": state,
                "certainty": "high",
                "evidence": "finding",
                "reason": "The report explicitly determines this target.",
            }
            for target in TARGETS
        }
        probabilities = {target: 0.95 if label else 0.05 for target in TARGETS}
        structured.append(
            {
                "StudyInstanceUID": uid,
                "findings": findings,
                "probabilities": probabilities,
            }
        )
        public[uid] = {target: 0.9 if not label else 0.1 for target in TARGETS}
    return train_rows, structured, public


def test_audit_reports_paired_candidate_and_public_metrics() -> None:
    config = copy.deepcopy(load_label_config(CONFIG_PATH))
    config["run"]["expected_gold_studies"] = 4
    config["run"]["bootstrap_resamples"] = 100
    train_rows, structured, public = _audit_fixture()
    summary, rows = build_audit(train_rows, structured, public, config)
    assert summary["candidate_macro_roc_auc"] == 1.0
    assert summary["public_macro_roc_auc"] == 0.0
    assert summary["macro_roc_auc_difference"] == 1.0
    assert len(rows) == len(TARGETS)
    assert rows[0]["candidate"]["average_precision"] == 1.0
    assert rows[0]["explicit"]["f1"] == 1.0


def test_auc_helpers_handle_ties_single_class_and_reproducible_bootstrap() -> None:
    assert average_precision([0, 1], [0.5, 0.5]) == 0.5
    assert math.isnan(average_precision([0, 0], [0.1, 0.2]))
    labels = {target: [0, 1, 0, 1] for target in TARGETS}
    candidate = {target: [0.1, 0.9, 0.2, 0.8] for target in TARGETS}
    baseline = {target: [0.9, 0.1, 0.8, 0.2] for target in TARGETS}
    first = paired_bootstrap_auc_difference(labels, candidate, baseline, 100, 42)
    second = paired_bootstrap_auc_difference(labels, candidate, baseline, 100, 42)
    assert first == second
    assert first["median"] == 1.0
