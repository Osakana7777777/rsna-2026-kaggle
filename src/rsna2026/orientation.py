"""Patient-coordinate orientation normalization for knee DICOM images."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


ORIENTATION_NORMALIZATION_METHOD = "dicom-patient-medial-right-v1"
ORIENTATION_TAGS = (
    "ImageOrientationPatient",
    "ImagePositionPatient",
    "PixelSpacing",
    "Laterality",
    "Rows",
    "Columns",
)


@dataclass(frozen=True)
class OrientationDecision:
    """Auditable laterality and horizontal-flip decision for one DICOM image."""

    plane: str
    laterality_tag: str | None
    geometry_laterality: str | None
    resolved_laterality: str | None
    laterality_source: str
    horizontal_flip: bool
    tag_geometry_disagreement: bool
    audit_reason: str | None


def read_orientation_header(path: Path) -> OrientationDecision:
    """Read only DICOM values needed for an auditable orientation decision."""
    import pydicom

    dataset = pydicom.dcmread(
        path,
        force=True,
        stop_before_pixels=True,
        specific_tags=list(ORIENTATION_TAGS),
    )
    rows = int(getattr(dataset, "Rows", 0) or 0)
    columns = int(getattr(dataset, "Columns", 0) or 0)
    return decide_orientation(
        {name: getattr(dataset, name, None) for name in ORIENTATION_TAGS},
        (rows, columns),
    )


def _values(value: object, length: int, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must contain {length} numeric values")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain {length} numeric values") from error
    if len(result) != length or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain {length} finite numeric values")
    return result


def normalize_laterality(value: object) -> str | None:
    """Normalize standard DICOM laterality spellings without guessing."""
    text = str(value or "").strip().upper()
    if text in {"L", "LEFT"}:
        return "L"
    if text in {"R", "RIGHT"}:
        return "R"
    return None


def _plane_from_orientation(orientation: tuple[float, ...]) -> str:
    column = orientation[:3]
    row = orientation[3:]
    normal = (
        column[1] * row[2] - column[2] * row[1],
        column[2] * row[0] - column[0] * row[2],
        column[0] * row[1] - column[1] * row[0],
    )
    axis = max(range(3), key=lambda index: abs(normal[index]))
    if abs(normal[axis]) < 0.8:
        return "oblique"
    return ("sagittal", "coronal", "axial")[axis]


def decide_orientation(
    header: Mapping[str, object],
    pixel_shape: tuple[int, int],
) -> OrientationDecision:
    """Resolve knee side and whether medial anatomy must be moved image-right.

    DICOM patient X increases toward the patient's left. Geometry laterality is
    accepted only when the complete pixel-center extent lies on one side of the
    patient midline. When a valid tag disagrees with geometry, geometry controls
    normalization and the decision is explicitly marked for audit.
    """
    laterality_tag = normalize_laterality(header.get("Laterality"))
    try:
        orientation = _values(
            header.get("ImageOrientationPatient"), 6, "ImageOrientationPatient"
        )
        position = _values(
            header.get("ImagePositionPatient"), 3, "ImagePositionPatient"
        )
        spacing = _values(header.get("PixelSpacing"), 2, "PixelSpacing")
        if any(value <= 0.0 for value in spacing):
            raise ValueError("PixelSpacing values must be positive")
        rows, columns = (int(pixel_shape[0]), int(pixel_shape[1]))
        if rows <= 0 or columns <= 0:
            raise ValueError("Pixel shape must be positive")
    except (TypeError, ValueError):
        orientation = ()
        geometry_laterality = None
        plane = "unknown"
    else:
        plane = _plane_from_orientation(orientation)
        column_direction = orientation[:3]
        row_direction = orientation[3:]
        x_coordinates = (
            position[0],
            position[0] + (columns - 1) * spacing[1] * column_direction[0],
            position[0] + (rows - 1) * spacing[0] * row_direction[0],
            position[0]
            + (columns - 1) * spacing[1] * column_direction[0]
            + (rows - 1) * spacing[0] * row_direction[0],
        )
        if min(x_coordinates) > 0.0:
            geometry_laterality = "L"
        elif max(x_coordinates) < 0.0:
            geometry_laterality = "R"
        else:
            geometry_laterality = None

    disagreement = (
        laterality_tag is not None
        and geometry_laterality is not None
        and laterality_tag != geometry_laterality
    )
    if geometry_laterality is not None:
        resolved = geometry_laterality
        source = "geometry"
    elif laterality_tag is not None:
        resolved = laterality_tag
        source = "dicom-tag"
    else:
        resolved = None
        source = "unresolved"

    horizontal_flip = False
    audit_reason = "laterality-tag-geometry-disagreement" if disagreement else None
    if plane in {"coronal", "axial"} and resolved is not None and orientation:
        patient_left_toward_image_right = orientation[0]
        if abs(patient_left_toward_image_right) < 0.5:
            audit_reason = audit_reason or "ambiguous-horizontal-patient-axis"
        else:
            desired_sign = 1.0 if resolved == "R" else -1.0
            horizontal_flip = patient_left_toward_image_right * desired_sign < 0.0

    return OrientationDecision(
        plane=plane,
        laterality_tag=laterality_tag,
        geometry_laterality=geometry_laterality,
        resolved_laterality=resolved,
        laterality_source=source,
        horizontal_flip=horizontal_flip,
        tag_geometry_disagreement=disagreement,
        audit_reason=audit_reason,
    )
