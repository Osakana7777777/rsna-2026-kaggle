import ast
import csv
import json
import sys
import types
from pathlib import Path

import pytest

from rsna2026.weak_label_codes import TARGETS
from rsna2026.weak_label_decode import (
    competition_call,
    decode_prediction,
    flatten_prediction,
    initial_weight,
    needs_second_pass,
    semantic_warnings,
)
from rsna2026.weak_label_infer import (
    InferenceSettings,
    Report,
    completed_study_ids,
    create_engine,
    messages,
    read_reports,
)
from rsna2026.weak_label_postprocess import load_flat_rows, write_parquet
from rsna2026.weak_label_schema import (
    PredictionValidationError,
    RSNA_LABEL_SCHEMA,
    VLLM_RSNA_LABEL_SCHEMA,
    validate_prediction,
)


def absent_finding():
    return [0, 0, 0, 0, 0, [], [], []]


def prediction_with(target, finding):
    findings = [absent_finding() for _ in TARGETS]
    findings[TARGETS.index(target)] = finding
    return {"l": "en", "s": 0, "g": [], "x": findings}


@pytest.mark.parametrize(
    ("target", "finding", "expected_call", "feature"),
    [
        ("ACL", [1, 1, 12, 1, 3, [1], [], ["Complete ACL rupture"]], "positive", 1),
        (
            "Effusion",
            [1, 1, 4, 1, 1, [15], [], ["Small joint effusion"]],
            "negative",
            15,
        ),
        (
            "Effusion",
            [1, 1, 7, 1, 3, [14], [], ["Moderate joint effusion"]],
            "positive",
            14,
        ),
        (
            "Medial_Meniscus",
            [
                1,
                1,
                5,
                4,
                1,
                [9],
                [],
                [
                    "Mild degenerative signal within the medial meniscus without definite tear"
                ],
            ],
            "negative",
            9,
        ),
        (
            "Contusion",
            [
                1,
                1,
                0,
                1,
                1,
                [20],
                [],
                ["Bone marrow edema related to osteoarthritis"],
            ],
            "negative",
            20,
        ),
        (
            "Fracture",
            [2, 1, 1, 0, 1, [24], [], ["No acute fracture"]],
            "negative",
            24,
        ),
    ],
)
def test_acceptance_examples_decode_to_expected_competition_call(
    target, finding, expected_call, feature
):
    decoded = decode_prediction(prediction_with(target, finding))
    decoded_finding = decoded.findings[TARGETS.index(target)]
    assert competition_call(decoded_finding) == expected_call
    assert feature in decoded_finding.features


def test_unmentioned_findings_have_no_evidence_and_are_uncertain():
    decoded = decode_prediction(prediction_with("ACL", absent_finding()))
    acl = decoded.findings[0]
    assert acl.mention == 0
    assert acl.evidence == ()
    assert competition_call(acl) == "uncertain"


def test_schema_rejects_wrong_lengths_ranges_duplicates_and_extra_keys():
    valid = prediction_with("ACL", absent_finding())
    assert validate_prediction(valid) is valid

    wrong_count = {**valid, "x": valid["x"][:-1]}
    with pytest.raises(PredictionValidationError, match="12 findings"):
        validate_prediction(wrong_count)

    invalid_feature = prediction_with("ACL", [1, 1, 12, 1, 3, [30], [], ["tear"]])
    with pytest.raises(PredictionValidationError, match="between 1 and 29"):
        validate_prediction(invalid_feature)

    duplicate_review = prediction_with("ACL", [1, 1, 12, 1, 3, [1], [1, 1], ["tear"]])
    with pytest.raises(PredictionValidationError, match="unique"):
        validate_prediction(duplicate_review)

    with pytest.raises(PredictionValidationError, match="keys"):
        validate_prediction({**valid, "probability": 0.9})


def test_vllm_schema_omits_unsupported_unique_items_but_runtime_keeps_it():
    assert "uniqueItems" in json.dumps(RSNA_LABEL_SCHEMA)
    assert "uniqueItems" not in json.dumps(VLLM_RSNA_LABEL_SCHEMA)

    duplicate_global_flag = prediction_with("ACL", absent_finding())
    duplicate_global_flag["g"] = [1, 1]
    with pytest.raises(PredictionValidationError, match="unique"):
        validate_prediction(duplicate_global_flag)


def test_semantic_warnings_do_not_rewrite_prediction():
    value = prediction_with("ACL", [0, 0, 0, 0, 3, [1], [], ["complete tear"]])
    decoded = decode_prediction(value)
    warnings = semantic_warnings(decoded)
    assert "ACL:not_mentioned_with_evidence" in warnings
    assert "ACL:not_mentioned_clearly_above" in warnings
    assert decoded.findings[0].mention == 0
    assert needs_second_pass(decoded)


def test_weight_is_independent_of_probability_and_applies_review_penalties():
    base = [1, 1, 12, 2, 3, [1], [], ["complete tear"]]
    uncertain = [1, 1, 12, 2, 2, [1], [4, 6], ["tear"]]
    assert initial_weight(base) == 1.0
    assert initial_weight(uncertain) == pytest.approx(0.0735)


def test_flatten_adds_calls_weights_review_mask_and_second_pass():
    value = prediction_with("ACL", [1, 2, 8, 6, 2, [2], [1], ["partial tear"]])
    value["g"] = [2]
    row = flatten_prediction("study-1", value)
    assert row["StudyInstanceUID"] == "study-1"
    assert row["ACL_competition_call"] == "uncertain"
    assert row["ACL_initial_weight"] == pytest.approx(0.49)
    assert row["ACL_needs_review"] is True
    assert row["mask"] == 0
    assert row["needs_second_pass"] is True


def test_messages_have_identical_system_prefix_and_variable_user_suffix():
    first = messages("fixed", Report("uid-1", "report one"))
    second = messages("fixed", Report("uid-2", "report two"))
    assert first[0] == second[0] == {"role": "system", "content": "fixed"}
    assert first[1] != second[1]
    assert first[1]["content"].endswith("report one")


def test_read_reports_rejects_duplicate_uids(tmp_path):
    path = tmp_path / "reports.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["StudyInstanceUID", "report"])
        writer.writeheader()
        writer.writerows(
            [
                {"StudyInstanceUID": "same", "report": "one"},
                {"StudyInstanceUID": "same", "report": "two"},
            ]
        )
    with pytest.raises(ValueError, match="Duplicate"):
        read_reports(path, "StudyInstanceUID", "report")


def test_resume_and_postprocess_revalidate_canonical_jsonl(tmp_path):
    path = tmp_path / "raw.jsonl"
    record = {
        "study_id": "study-1",
        "prompt_version": "v1",
        "model": "test-model",
        "prediction": prediction_with("ACL", absent_finding()),
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert completed_study_ids(path) == {"study-1"}
    rows = load_flat_rows(path)
    assert rows[0]["prompt_version"] == "v1"
    assert rows[0]["model"] == "test-model"

    with pytest.raises(ValueError, match="settings differ"):
        completed_study_ids(path, {"generation_config": "auto"})

    path.write_text('{"study_id":"broken"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSONL line"):
        completed_study_ids(path)


def test_flat_rows_round_trip_through_parquet(tmp_path):
    parquet = pytest.importorskip("pyarrow.parquet")
    row = flatten_prediction("study-1", prediction_with("ACL", absent_finding()))
    path = tmp_path / "labels.parquet"
    write_parquet([row], path)
    table = parquet.read_table(path)
    assert table.num_rows == 1
    assert table.column("StudyInstanceUID").to_pylist() == ["study-1"]
    assert table.column("ACL_competition_call").to_pylist() == ["uncertain"]


def test_modal_model_prepare_is_cpu_only_and_inference_is_h100():
    source_path = Path(__file__).resolve().parents[1] / "modal_weak_labels.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    prepare_decorator = functions["prepare_model"].decorator_list[0]
    check_decorator = functions["check_prepared_model"].decorator_list[0]
    extract_decorator = functions["extract"].decorator_list[0]
    assert isinstance(prepare_decorator, ast.Call)
    assert isinstance(check_decorator, ast.Call)
    assert isinstance(extract_decorator, ast.Call)
    assert "gpu" not in {keyword.arg for keyword in prepare_decorator.keywords}
    assert "gpu" not in {keyword.arg for keyword in check_decorator.keywords}
    extract_keywords = {
        keyword.arg: keyword.value for keyword in extract_decorator.keywords
    }
    assert ast.literal_eval(extract_keywords["gpu"]) == "H100"
    assert '"HF_HUB_OFFLINE": "1"' in source
    assert '"TRANSFORMERS_OFFLINE": "1"' in source
    assert "snapshot_path = _prepared_snapshot(model, revision)" in source
    assert 'if action == "prepare-model":' in source
    assert "check_prepared_model.remote(" in source
    assert "input_path=input_path" in source
    assert source.index(".env(") < source.index(".add_local_dir(")


def test_vllm_engine_prefers_prepared_local_snapshot(monkeypatch):
    captured = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            captured["engine"] = kwargs

        def get_default_sampling_params(self):
            sampling = types.SimpleNamespace(
                temperature=1.0,
                top_p=0.95,
                top_k=20,
                max_tokens=None,
                structured_outputs=None,
            )
            captured["sampling"] = sampling
            return sampling

    class FakeStructuredOutputsParams:
        def __init__(self, **kwargs):
            captured["structured"] = kwargs

    vllm = types.ModuleType("vllm")
    vllm.LLM = FakeLLM
    sampling_params = types.ModuleType("vllm.sampling_params")
    sampling_params.StructuredOutputsParams = FakeStructuredOutputsParams
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling_params)

    create_engine(
        InferenceSettings(
            model="remote/model",
            model_revision="fixed-revision",
            model_path="/cache/snapshots/fixed-revision",
        )
    )
    assert captured["engine"]["model"] == "/cache/snapshots/fixed-revision"
    assert captured["engine"]["tensor_parallel_size"] == 1
    assert captured["engine"]["max_num_seqs"] == 64
    assert captured["engine"]["enable_prefix_caching"] is True
    assert captured["engine"]["generation_config"] == "auto"
    assert captured["sampling"].temperature == 1.0
    assert captured["sampling"].top_p == 0.95
    assert captured["sampling"].top_k == 20
    assert captured["sampling"].max_tokens == 1_200
    assert captured["sampling"].structured_outputs is not None
    assert captured["structured"]["json"]["required"] == ["l", "s", "g", "x"]
