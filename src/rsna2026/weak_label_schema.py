"""JSON schema and strict runtime validation for compact LLM output."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from rsna2026.weak_label_codes import TARGETS

FINDING_SCHEMA: dict[str, Any] = {
    "type": "array",
    "minItems": 8,
    "maxItems": 8,
    "prefixItems": [
        {"type": "integer", "enum": [0, 1, 2, 3]},
        {"type": "integer", "enum": [0, 1, 2, 3, 4]},
        {"type": "integer", "minimum": 0, "maximum": 12},
        {"type": "integer", "minimum": 0, "maximum": 6},
        {"type": "integer", "enum": [0, 1, 2, 3, 4]},
        {
            "type": "array",
            "uniqueItems": True,
            "maxItems": 5,
            "items": {"type": "integer", "minimum": 1, "maximum": 29},
        },
        {
            "type": "array",
            "uniqueItems": True,
            "maxItems": 4,
            "items": {"type": "integer", "minimum": 1, "maximum": 9},
        },
        {
            "type": "array",
            "maxItems": 2,
            "items": {"type": "string", "maxLength": 160},
        },
    ],
    "items": False,
}

RSNA_LABEL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["l", "s", "g", "x"],
    "properties": {
        "l": {"type": "string", "minLength": 2, "maxLength": 8},
        "s": {"type": "integer", "enum": [0, 1, 2, 3]},
        "g": {
            "type": "array",
            "uniqueItems": True,
            "maxItems": 5,
            "items": {"type": "integer", "minimum": 1, "maximum": 5},
        },
        "x": {
            "type": "array",
            "minItems": len(TARGETS),
            "maxItems": len(TARGETS),
            "items": FINDING_SCHEMA,
        },
    },
}


def _schema_without_unique_items(value: object) -> object:
    """Remove a keyword unsupported by vLLM guidance; runtime validation keeps it."""
    if isinstance(value, dict):
        return {
            key: _schema_without_unique_items(item)
            for key, item in value.items()
            if key != "uniqueItems"
        }
    if isinstance(value, list):
        return [_schema_without_unique_items(item) for item in value]
    return value


# vLLM's guidance backend rejects ``uniqueItems``. The constrained decoder
# enforces every other structural rule, and ``validate_prediction`` below still
# rejects duplicate global, feature, and review IDs before anything is persisted.
VLLM_RSNA_LABEL_SCHEMA: dict[str, Any] = _schema_without_unique_items(
    deepcopy(RSNA_LABEL_SCHEMA)
)

OPENAI_FINDING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["m", "c", "v", "a", "t", "k", "r", "e"],
    "properties": {
        "m": {"type": "integer", "enum": [0, 1, 2, 3]},
        "c": {"type": "integer", "enum": [0, 1, 2, 3, 4]},
        "v": {"type": "integer", "minimum": 0, "maximum": 12},
        "a": {"type": "integer", "minimum": 0, "maximum": 6},
        "t": {"type": "integer", "enum": [0, 1, 2, 3, 4]},
        "k": {
            "type": "array",
            "maxItems": 5,
            "items": {"type": "integer", "minimum": 1, "maximum": 29},
        },
        "r": {
            "type": "array",
            "maxItems": 4,
            "items": {"type": "integer", "minimum": 1, "maximum": 9},
        },
        "e": {
            "type": "array",
            "maxItems": 2,
            "items": {"type": "string"},
        },
    },
}

# The OpenAI Structured Outputs subset does not support tuple-style
# ``prefixItems`` or ``uniqueItems``. Named finding fields keep every other
# constraint strict; they are compacted and de-duplicated before validation.
OPENAI_RSNA_LABEL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["l", "s", "g", "x"],
    "properties": {
        "l": {"type": "string"},
        "s": {"type": "integer", "enum": [0, 1, 2, 3]},
        "g": {
            "type": "array",
            "maxItems": 5,
            "items": {"type": "integer", "minimum": 1, "maximum": 5},
        },
        "x": {
            "type": "array",
            "minItems": len(TARGETS),
            "maxItems": len(TARGETS),
            "items": OPENAI_FINDING_SCHEMA,
        },
    },
}


class PredictionValidationError(ValueError):
    """Raised when model output violates the production schema."""


def _deduplicate_integer_list(value: object) -> tuple[object, bool]:
    """Remove repeated integer codes while preserving their first-seen order."""
    if not isinstance(value, list) or not all(type(item) is int for item in value):
        return value, False
    deduplicated = list(dict.fromkeys(value))
    return deduplicated, deduplicated != value


def normalize_unique_items(value: object) -> tuple[object, list[str]]:
    """Normalize only arrays whose production schema declares ``uniqueItems``."""
    normalized = deepcopy(value)
    changed_paths: list[str] = []
    if not isinstance(normalized, dict):
        return normalized, changed_paths

    global_flags, changed = _deduplicate_integer_list(normalized.get("g"))
    if changed:
        normalized["g"] = global_flags
        changed_paths.append("g")

    findings = normalized.get("x")
    if not isinstance(findings, list):
        return normalized, changed_paths
    for target_index, finding in enumerate(findings):
        if not isinstance(finding, list):
            continue
        for item_index in (5, 6):
            if len(finding) <= item_index:
                continue
            codes, changed = _deduplicate_integer_list(finding[item_index])
            if changed:
                finding[item_index] = codes
                changed_paths.append(f"x[{target_index}][{item_index}]")
    return normalized, changed_paths


def compact_openai_prediction(value: object) -> object:
    """Convert the API's named finding objects to the canonical compact tuples."""
    if not isinstance(value, dict):
        return value
    compact = deepcopy(value)
    findings = compact.get("x")
    if not isinstance(findings, list):
        return compact
    field_order = ("m", "c", "v", "a", "t", "k", "r", "e")
    compact["x"] = [
        [finding.get(field) for field in field_order]
        if isinstance(finding, dict)
        else finding
        for finding in findings
    ]
    return compact


def _integer(value: object, path: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PredictionValidationError(f"{path} must be an integer")
    if not minimum <= value <= maximum:
        raise PredictionValidationError(
            f"{path} must be between {minimum} and {maximum}"
        )
    return value


def _unique_integer_list(
    value: object,
    path: str,
    maximum_items: int,
    maximum_value: int,
) -> list[int]:
    if not isinstance(value, list) or len(value) > maximum_items:
        raise PredictionValidationError(
            f"{path} must be an array with at most {maximum_items} items"
        )
    result = [
        _integer(item, f"{path}[{index}]", 1, maximum_value)
        for index, item in enumerate(value)
    ]
    if len(result) != len(set(result)):
        raise PredictionValidationError(f"{path} must contain unique items")
    return result


def validate_prediction(value: object) -> dict[str, Any]:
    """Validate and return a compact prediction without coercing any value."""
    if not isinstance(value, dict):
        raise PredictionValidationError("prediction must be an object")
    expected_keys = {"l", "s", "g", "x"}
    if set(value) != expected_keys:
        raise PredictionValidationError(
            f"prediction keys must be exactly {sorted(expected_keys)}"
        )

    language = value["l"]
    if not isinstance(language, str) or not 2 <= len(language) <= 8:
        raise PredictionValidationError("l must be a string of length 2 through 8")
    _integer(value["s"], "s", 0, 3)
    _unique_integer_list(value["g"], "g", 5, 5)

    findings = value["x"]
    if not isinstance(findings, list) or len(findings) != len(TARGETS):
        raise PredictionValidationError(f"x must contain {len(TARGETS)} findings")
    for target_index, finding in enumerate(findings):
        path = f"x[{target_index}]"
        if not isinstance(finding, list) or len(finding) != 8:
            raise PredictionValidationError(f"{path} must contain 8 items")
        _integer(finding[0], f"{path}[0]", 0, 3)
        _integer(finding[1], f"{path}[1]", 0, 4)
        _integer(finding[2], f"{path}[2]", 0, 12)
        _integer(finding[3], f"{path}[3]", 0, 6)
        _integer(finding[4], f"{path}[4]", 0, 4)
        _unique_integer_list(finding[5], f"{path}[5]", 5, 29)
        _unique_integer_list(finding[6], f"{path}[6]", 4, 9)
        evidence = finding[7]
        if not isinstance(evidence, list) or len(evidence) > 2:
            raise PredictionValidationError(
                f"{path}[7] must be an array with at most 2 items"
            )
        for evidence_index, excerpt in enumerate(evidence):
            if not isinstance(excerpt, str) or len(excerpt) > 160:
                raise PredictionValidationError(
                    f"{path}[7][{evidence_index}] must be a string of at most 160 characters"
                )
    return value
