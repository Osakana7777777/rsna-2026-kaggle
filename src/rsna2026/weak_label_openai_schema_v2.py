"""OpenAI Structured Outputs schema for report-derived weak labels."""

from __future__ import annotations

from typing import Any

from rsna2026.weak_label_codes import TARGETS

OPENAI_SCHEMA_VERSION = "openai-v2"

# OpenAI Structured Outputs supports ``pattern`` but not ``maxLength``. Match
# every character class (including newlines) so the runtime's 160-character
# evidence limit is also enforced during constrained generation.
EVIDENCE_PATTERN = r"^[\s\S]{0,160}$"
LANGUAGE_PATTERN = r"^[\s\S]{2,8}$"

OPENAI_FINDING_SCHEMA_V2: dict[str, Any] = {
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
            "items": {"type": "string", "pattern": EVIDENCE_PATTERN},
        },
    },
}

OPENAI_RSNA_LABEL_SCHEMA_V2: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["l", "s", "g", "x"],
    "properties": {
        "l": {"type": "string", "pattern": LANGUAGE_PATTERN},
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
            "items": OPENAI_FINDING_SCHEMA_V2,
        },
    },
}
