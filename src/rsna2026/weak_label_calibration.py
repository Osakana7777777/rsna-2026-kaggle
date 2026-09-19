"""Replaceable mapping from structured extraction to soft probabilities."""

from __future__ import annotations

from collections.abc import Mapping

from rsna2026.weak_label_decode import CompetitionCall, DecodedFinding, competition_call

DEFAULT_CALL_PROBABILITIES: Mapping[CompetitionCall, float] = {
    "negative": 0.10,
    "uncertain": 0.50,
    "positive": 0.90,
}


def soft_probability(
    finding: DecodedFinding,
    mapping: Mapping[CompetitionCall, float] = DEFAULT_CALL_PROBABILITIES,
) -> float:
    """Map a decoded finding to a replaceable, explicitly calibrated target."""
    probability = float(mapping[competition_call(finding)])
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"Calibrated probability outside [0, 1]: {probability}")
    return probability
