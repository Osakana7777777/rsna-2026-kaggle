import json
import re
import types
from dataclasses import asdict

import httpx
import pytest
from openai import BadRequestError, OpenAI

from rsna2026.weak_label_codes import TARGETS
from rsna2026.weak_label_infer import Report
from rsna2026.weak_label_openai import (
    DEFAULT_OPENAI_MODEL,
    OpenAIInferenceSettings,
    generate_one,
    main,
    openai_prompt,
    read_error_study_ids,
    response_request,
    retry_error_output_path,
    run_inference,
    select_retry_reports,
)
from rsna2026.weak_label_openai_schema_v2 import (
    EVIDENCE_PATTERN,
    OPENAI_RSNA_LABEL_SCHEMA_V2,
    OPENAI_SCHEMA_VERSION,
)


def api_finding():
    return {"m": 0, "c": 0, "v": 0, "a": 0, "t": 0, "k": [], "r": [], "e": []}


def api_prediction():
    return {
        "l": "en",
        "s": 0,
        "g": [],
        "x": [api_finding() for _ in TARGETS],
    }


def test_request_uses_luna_strict_schema_and_explicit_prefix_cache():
    settings = OpenAIInferenceSettings()
    request = response_request(
        "fixed extraction instructions",
        Report("study-1", "normal knee report"),
        settings,
        "v1",
    )

    assert request["model"] == DEFAULT_OPENAI_MODEL == "gpt-5.6-luna"
    assert request["store"] is False
    assert request["reasoning"] == {"effort": "medium"}
    assert request["text"]["format"]["strict"] is True
    assert request["text"]["format"]["schema"] is OPENAI_RSNA_LABEL_SCHEMA_V2
    assert request["input"][0]["role"] == "user"
    assert request["input"][0]["content"][0]["text"] == "fixed extraction instructions"
    assert request["input"][0]["content"][0]["prompt_cache_breakpoint"] == {
        "mode": "explicit"
    }
    assert request["input"][1]["role"] == "user"
    assert request["input"][1]["content"].endswith("normal knee report")
    assert request["prompt_cache_key"] == (
        f"rsna-knee-weak-label:v1:{OPENAI_SCHEMA_VERSION}"
    )
    assert request["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}


def test_openai_prompt_changes_only_finding_wire_format():
    source = "before\nEach finding is exactly [m,c,v,a,t,k,r,e].\nafter"
    adapted = openai_prompt(source)
    assert "exactly an object with the named fields m,c,v,a,t,k,r,e" in adapted
    assert adapted.startswith("before\n")
    assert adapted.endswith("\nafter")


def test_openai_schema_uses_supported_named_findings():
    serialized = json.dumps(OPENAI_RSNA_LABEL_SCHEMA_V2)
    assert "prefixItems" not in serialized
    assert "uniqueItems" not in serialized
    assert "minLength" not in serialized
    assert "maxLength" not in serialized
    finding = OPENAI_RSNA_LABEL_SCHEMA_V2["properties"]["x"]["items"]
    assert finding["additionalProperties"] is False
    assert finding["required"] == ["m", "c", "v", "a", "t", "k", "r", "e"]


def test_openai_schema_pattern_enforces_160_character_evidence_limit():
    evidence_schema = OPENAI_RSNA_LABEL_SCHEMA_V2["properties"]["x"]["items"][
        "properties"
    ]["e"]["items"]
    assert evidence_schema == {"type": "string", "pattern": EVIDENCE_PATTERN}
    assert re.fullmatch(EVIDENCE_PATTERN, "a" * 160)
    assert re.fullmatch(EVIDENCE_PATTERN, "line one\nline two")
    assert not re.fullmatch(EVIDENCE_PATTERN, "a" * 161)


def test_generate_one_compacts_and_deduplicates_api_output():
    prediction = api_prediction()
    prediction["g"] = [1, 1]
    prediction["x"][0]["k"] = [1, 1, 2]
    usage = types.SimpleNamespace(
        input_tokens=2_000,
        output_tokens=200,
        total_tokens=2_200,
        input_tokens_details=types.SimpleNamespace(
            cached_tokens=1_500,
            cache_write_tokens=300,
        ),
    )
    response = types.SimpleNamespace(
        id="resp_test",
        status="completed",
        output_text=json.dumps(prediction),
        usage=usage,
    )

    class FakeResponses:
        def create(self, **kwargs):
            self.request = kwargs
            return response

    responses = FakeResponses()
    client = types.SimpleNamespace(responses=responses)
    row = generate_one(
        client,
        Report("study-1", "report"),
        "fixed prompt",
        "v1",
        OpenAIInferenceSettings(reasoning_effort="none"),
    )

    assert row["prediction"]["g"] == [1]
    assert row["prediction"]["x"][0][5] == [1, 2]
    assert row["normalized_unique_items"] == ["g", "x[0][5]"]
    assert row["schema_version"] == OPENAI_SCHEMA_VERSION
    assert row["generation_settings"]["schema_version"] == OPENAI_SCHEMA_VERSION
    assert row["openai_response_id"] == "resp_test"
    assert row["usage"] == {
        "input_tokens": 2_000,
        "cached_input_tokens": 1_500,
        "cache_write_tokens": 300,
        "output_tokens": 200,
        "total_tokens": 2_200,
    }
    assert responses.request["reasoning"] == {"effort": "none"}


def test_installed_openai_sdk_serializes_cache_and_schema_fields():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            400,
            request=request,
            json={"error": {"message": "test", "type": "invalid_request_error"}},
        )

    client = OpenAI(
        api_key="test",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    request = response_request(
        "fixed extraction instructions",
        Report("study-1", "report"),
        OpenAIInferenceSettings(),
        "v1",
    )
    with pytest.raises(BadRequestError):
        client.responses.create(**request)

    assert captured["body"]["prompt_cache_options"] == {
        "mode": "explicit",
        "ttl": "30m",
    }
    assert captured["body"]["input"][0]["content"][0]["prompt_cache_breakpoint"] == {
        "mode": "explicit"
    }
    assert captured["body"]["text"]["format"]["strict"] is True


def test_fully_resumed_run_does_not_require_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    metrics = run_inference(
        [],
        tmp_path / "predictions.jsonl",
        tmp_path / "errors.jsonl",
        "prompt",
        "openai-v1",
        OpenAIInferenceSettings(),
    )
    assert metrics["processed"] == 0
    assert metrics["failures"] == 0
    assert metrics["cached_input_tokens"] == 0


def compact_prediction():
    prediction = api_prediction()
    prediction["x"] = [
        [
            finding["m"],
            finding["c"],
            finding["v"],
            finding["a"],
            finding["t"],
            finding["k"],
            finding["r"],
            finding["e"],
        ]
        for finding in prediction["x"]
    ]
    return prediction


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_read_error_study_ids_deduplicates_repeated_attempts(tmp_path):
    error_path = tmp_path / "errors.jsonl"
    write_jsonl(
        error_path,
        [
            {"study_id": "study-2", "error": "first"},
            {"study_id": "study-1", "error": "second"},
            {"study_id": "study-2", "error": "third"},
        ],
    )

    assert read_error_study_ids(error_path) == ["study-2", "study-1"]


def test_select_retry_reports_uses_only_unresolved_error_ids(tmp_path):
    reports = [
        Report("study-1", "one"),
        Report("study-2", "two"),
        Report("study-3", "three"),
    ]
    error_path = tmp_path / "errors.jsonl"
    output_path = tmp_path / "predictions.jsonl"
    write_jsonl(
        error_path,
        [
            {"study_id": "study-2", "error": "failed"},
            {"study_id": "study-3", "error": "failed"},
        ],
    )
    write_jsonl(
        output_path,
        [
            {
                "study_id": "study-3",
                "prediction": compact_prediction(),
                "inference_settings": asdict(OpenAIInferenceSettings()),
            }
        ],
    )

    pending, requested, already_completed = select_retry_reports(
        reports,
        error_path,
        output_path,
        asdict(OpenAIInferenceSettings()),
    )

    assert pending == [Report("study-2", "two")]
    assert requested == 2
    assert already_completed == 1


def test_select_retry_reports_rejects_mixed_inference_settings(tmp_path):
    reports = [Report("study-1", "one"), Report("study-2", "two")]
    error_path = tmp_path / "errors.jsonl"
    output_path = tmp_path / "predictions.jsonl"
    write_jsonl(error_path, [{"study_id": "study-2", "error": "failed"}])
    write_jsonl(
        output_path,
        [
            {
                "study_id": "study-1",
                "prediction": compact_prediction(),
                "inference_settings": asdict(OpenAIInferenceSettings()),
            }
        ],
    )

    with pytest.raises(ValueError, match="inference settings differ"):
        select_retry_reports(
            reports,
            error_path,
            output_path,
            asdict(OpenAIInferenceSettings(reasoning_effort="none")),
        )


def test_retry_mode_requires_explicit_resume():
    with pytest.raises(ValueError, match="specify --resume"):
        main(
            [
                "--input",
                "missing.csv",
                "--output",
                "missing.jsonl",
                "--retry-errors-from",
                "missing.errors.jsonl",
            ]
        )


def test_retry_mode_appends_only_failed_studies_and_preserves_error_log(
    tmp_path, monkeypatch
):
    input_path = tmp_path / "reports.csv"
    input_path.write_text(
        "StudyInstanceUID,report\n"
        "study-1,one\n"
        "study-2,two\n"
        "study-3,three\n",
        encoding="utf-8",
    )
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("prompt", encoding="utf-8")
    output_path = tmp_path / "predictions.jsonl"
    write_jsonl(
        output_path,
        [
            {
                "study_id": "study-1",
                "prediction": compact_prediction(),
                "inference_settings": asdict(OpenAIInferenceSettings()),
            }
        ],
    )
    error_path = tmp_path / "predictions.errors.jsonl"
    write_jsonl(
        error_path,
        [
            {"study_id": "study-1", "error": "old failure"},
            {"study_id": "study-2", "error": "current failure"},
        ],
    )
    original_error_log = error_path.read_bytes()
    captured = {}

    def fake_run_inference(
        reports,
        output_path,
        error_path,
        prompt,
        prompt_version,
        settings,
    ):
        captured.update(
            reports=list(reports),
            output_path=output_path,
            error_path=error_path,
        )
        return {
            "number_of_reports": len(reports),
            "processed": len(reports),
            "failures": 0,
        }

    monkeypatch.setattr(
        "rsna2026.weak_label_openai.run_inference", fake_run_inference
    )

    result = main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--prompt",
            str(prompt_path),
            "--retry-errors-from",
            str(error_path),
            "--resume",
        ]
    )

    assert result == 0
    assert captured["reports"] == [Report("study-2", "two")]
    assert captured["output_path"] == output_path
    assert captured["error_path"] == retry_error_output_path(error_path)
    assert captured["error_path"] != error_path
    assert error_path.read_bytes() == original_error_log
