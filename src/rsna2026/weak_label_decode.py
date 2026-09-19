"""Decode, sanity-check, and flatten structured report extractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from rsna2026.weak_label_codes import TARGETS
from rsna2026.weak_label_schema import validate_prediction

FindingValue: TypeAlias = list[Any]
CompetitionCall: TypeAlias = Literal["positive", "negative", "uncertain"]


@dataclass(frozen=True)
class DecodedFinding:
    mention: int
    certainty: int
    severity: int
    acuity: int
    threshold: int
    features: tuple[int, ...]
    review_reasons: tuple[int, ...]
    evidence: tuple[str, ...]

    @property
    def needs_review(self) -> bool:
        return bool(self.review_reasons)


@dataclass(frozen=True)
class DecodedPrediction:
    language: str
    side: int
    global_flags: tuple[int, ...]
    findings: tuple[DecodedFinding, ...]


def decode_prediction(value: object) -> DecodedPrediction:
    prediction = validate_prediction(value)
    findings = tuple(
        DecodedFinding(
            mention=finding[0],
            certainty=finding[1],
            severity=finding[2],
            acuity=finding[3],
            threshold=finding[4],
            features=tuple(finding[5]),
            review_reasons=tuple(finding[6]),
            evidence=tuple(finding[7]),
        )
        for finding in prediction["x"]
    )
    return DecodedPrediction(
        language=prediction["l"],
        side=prediction["s"],
        global_flags=tuple(prediction["g"]),
        findings=findings,
    )


def competition_call(finding: DecodedFinding | FindingValue) -> CompetitionCall:
    """Map extracted facts to the competition decision without using a model."""
    if isinstance(finding, DecodedFinding):
        mention, threshold = finding.mention, finding.threshold
    else:
        mention, threshold = finding[0], finding[4]
    if mention == 0:
        return "uncertain"
    if mention == 2:
        return "negative"
    if mention == 3:
        return "uncertain"
    if threshold == 3:
        return "positive"
    if threshold == 1:
        return "negative"
    return "uncertain"


def initial_weight(finding: DecodedFinding | FindingValue) -> float:
    """Return the uncalibrated confidence weight specified for first-pass labels."""
    if isinstance(finding, DecodedFinding):
        mention = finding.mention
        certainty = finding.certainty
        threshold = finding.threshold
        review = finding.review_reasons
    else:
        mention, certainty, _, _, threshold, _, review, _ = finding

    if mention == 0:
        weight = 0.20
    elif certainty == 1 and threshold in (1, 3):
        weight = 1.00
    elif certainty == 2:
        weight = 0.70
    elif certainty == 3:
        weight = 0.45
    else:
        weight = 0.35
    for reason, multiplier in ((6, 0.5), (4, 0.6), (5, 0.5), (8, 0.7)):
        if reason in review:
            weight *= multiplier
    if threshold == 2:
        weight *= 0.7
    return max(0.05, min(1.0, weight))


def semantic_warnings(prediction: DecodedPrediction) -> list[str]:
    """Report contradictions without mutating the extraction."""
    warnings: list[str] = []
    positive_features = {1, 2, 3, 6, 7, 8, 10, 11, 12, 14, 16, 18, 21, 22}
    for target, finding in zip(TARGETS, prediction.findings):
        if finding.mention == 0 and finding.evidence:
            warnings.append(f"{target}:not_mentioned_with_evidence")
        if finding.mention == 0 and finding.threshold == 3:
            warnings.append(f"{target}:not_mentioned_clearly_above")
        if finding.mention == 2 and positive_features.intersection(finding.features):
            warnings.append(f"{target}:negative_with_positive_feature")
        if finding.mention == 2 and finding.threshold == 3:
            warnings.append(f"{target}:negative_clearly_above")
        if finding.mention != 0 and not finding.evidence:
            warnings.append(f"{target}:mentioned_without_evidence")
    return warnings


def initial_mask(prediction: DecodedPrediction) -> int:
    """Mask reports whose content or target laterality is unusable."""
    return 0 if {2, 5}.intersection(prediction.global_flags) else 1


def needs_second_pass(prediction: DecodedPrediction) -> bool:
    return bool(
        {3, 4}.intersection(prediction.global_flags)
        or semantic_warnings(prediction)
        or any(
            finding.review_reasons or finding.threshold == 2
            for finding in prediction.findings
        )
    )


def flatten_prediction(study_id: str, value: object) -> dict[str, object]:
    prediction = decode_prediction(value)
    warnings = semantic_warnings(prediction)
    row: dict[str, object] = {
        "StudyInstanceUID": study_id,
        "language": prediction.language,
        "side": prediction.side,
        "global_flags": list(prediction.global_flags),
        "semantic_warning": bool(warnings),
        "semantic_warnings": warnings,
        "needs_second_pass": needs_second_pass(prediction),
        "mask": initial_mask(prediction),
    }
    for target, finding in zip(TARGETS, prediction.findings):
        prefix = f"{target}_"
        row.update(
            {
                prefix + "m": finding.mention,
                prefix + "c": finding.certainty,
                prefix + "v": finding.severity,
                prefix + "a": finding.acuity,
                prefix + "t": finding.threshold,
                prefix + "k": list(finding.features),
                prefix + "r": list(finding.review_reasons),
                prefix + "e": list(finding.evidence),
                prefix + "competition_call": competition_call(finding),
                prefix + "initial_weight": initial_weight(finding),
                prefix + "needs_review": finding.needs_review,
            }
        )
    return row
