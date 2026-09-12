"""MRI acquisition classification and fixed diagnostic series slots."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


CLASSIFICATION_METHOD = "competition-series-table-v1"
CLASSIFICATION_SOURCE = "competition-series-table"


@dataclass(frozen=True)
class AcquisitionClass:
    """Acquisition properties recovered from one series' DICOM header."""

    weighting: str
    fluid_sensitive: bool
    fat_suppressed: bool
    weighting_source: str
    fat_suppression_source: str


@dataclass(frozen=True)
class AcquisitionSlot:
    """One clinically meaningful plane and acquisition combination."""

    name: str
    plane: str
    fluid_sensitive: bool | None
    fat_suppressed: bool

    def matches(
        self,
        plane: str,
        fluid_sensitive: bool,
        fat_suppressed: bool,
    ) -> bool:
        if plane != self.plane:
            return False
        return (
            fluid_sensitive == self.fluid_sensitive
            and fat_suppressed == self.fat_suppressed
        )


ACQUISITION_SLOTS = (
    AcquisitionSlot("sagittal-fluid", "sagittal", True, True),
    AcquisitionSlot("coronal-fluid", "coronal", True, True),
    AcquisitionSlot("axial-fluid", "axial", True, True),
    AcquisitionSlot("sagittal-structural", "sagittal", False, False),
    AcquisitionSlot("coronal-structural", "coronal", False, False),
    AcquisitionSlot("axial-structural", "axial", False, False),
)
SLOT_NAMES = tuple(slot.name for slot in ACQUISITION_SLOTS)
SLOTS_BY_NAME = {slot.name: slot for slot in ACQUISITION_SLOTS}


_SEPARATORS = re.compile(r"[_\-.]+")
_T1 = re.compile(r"\bt1w?\b")
_T2 = re.compile(r"\bt2w?\b")
_PD = re.compile(r"\bpdw?\b|\bproton\b|\bdens(?:ity)?\b|\bdp\b")
_FAT_SUPPRESSION = re.compile(
    r"\bfs\b|\bfatsat\b|\bfat sat\b|\bfat sup(?:pression)?\b|"
    r"\bstir\b|\bspair\b|\bspir\b|\bwater excit(?:ation)?\b|"
    r"\btirm\b|\bsting\b|\bfatsup\b|\bwe\b"
)
_FAT_SUPPRESSION_OPTIONS = {"FS", "FATSAT", "FAT_SAT", "FSAT"}


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence):
        return " ".join(str(item) for item in value)
    return str(value)


def _finite_number(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _scan_options(value: object) -> set[str]:
    if isinstance(value, str):
        raw = re.split(r"[|\\, ]+", value)
    elif isinstance(value, Sequence):
        raw = [str(item) for item in value]
    else:
        raw = []
    return {item.strip().upper() for item in raw if item.strip()}


def classify_acquisition(header: Mapping[str, object]) -> AcquisitionClass:
    """Classify weighting and fat suppression from normalized header values."""
    description = " ".join(
        _text(header.get(name))
        for name in ("SeriesDescription", "SequenceName", "ProtocolName")
    ).lower()
    description = _SEPARATORS.sub(" ", description)
    has_t1 = bool(_T1.search(description))
    has_t2 = bool(_T2.search(description))
    has_pd = bool(_PD.search(description))
    repetition_time = _finite_number(header.get("RepetitionTime"))
    echo_time = _finite_number(header.get("EchoTime"))
    scanning_sequence = _text(header.get("ScanningSequence")).upper()

    if has_t1 and not has_t2 and not has_pd:
        weighting, weighting_source = "t1", "series-text:t1"
    elif has_t2 and not has_pd:
        weighting, weighting_source = "t2", "series-text:t2"
    elif has_pd:
        weighting, weighting_source = "pd", "series-text:pd"
    elif "GR" in scanning_sequence:
        weighting, weighting_source = "gre", "scanning-sequence:gr"
    elif repetition_time is not None and repetition_time < 800.0:
        weighting, weighting_source = "t1", "timing:tr<800"
    elif echo_time is not None and echo_time > 60.0:
        weighting, weighting_source = "t2", "timing:te>60"
    elif repetition_time is not None and repetition_time >= 800.0:
        weighting, weighting_source = "pd", "timing:tr>=800"
    else:
        weighting, weighting_source = "unknown", "insufficient-header"

    text_fat_suppressed = bool(_FAT_SUPPRESSION.search(description))
    option_fat_suppressed = bool(
        _scan_options(header.get("ScanOptions")) & _FAT_SUPPRESSION_OPTIONS
    )
    fat_suppressed = text_fat_suppressed or option_fat_suppressed
    if text_fat_suppressed and option_fat_suppressed:
        fat_source = "series-text+scan-options"
    elif text_fat_suppressed:
        fat_source = "series-text"
    elif option_fat_suppressed:
        fat_source = "scan-options"
    else:
        fat_source = "not-detected"

    return AcquisitionClass(
        weighting=weighting,
        fluid_sensitive=weighting in {"pd", "t2"},
        fat_suppressed=fat_suppressed,
        weighting_source=weighting_source,
        fat_suppression_source=fat_source,
    )


def read_acquisition_header(path: Path) -> AcquisitionClass:
    """Read only the DICOM values required for acquisition classification."""
    import pydicom

    tags = [
        "SeriesDescription",
        "SequenceName",
        "ProtocolName",
        "ScanOptions",
        "RepetitionTime",
        "EchoTime",
        "ScanningSequence",
    ]
    dataset = pydicom.dcmread(
        path,
        force=True,
        stop_before_pixels=True,
        specific_tags=tags,
    )
    return classify_acquisition(
        {name: getattr(dataset, name, None) for name in tags}
    )


def binary_series_attribute(value: object, name: str) -> bool:
    """Normalize one competition-series binary attribute without guessing."""
    if isinstance(value, bool):
        return value
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {name}: {value!r}") from error
    if not math.isfinite(number) or number not in {0.0, 1.0}:
        raise ValueError(f"Invalid {name}: {value!r}")
    return bool(number)


def match_series_table_slot(
    plane: str,
    fluid_sensitive: object,
    fat_suppression: object,
) -> tuple[str | None, str | None]:
    """Map the competition table's 11/00 acquisition values to one slot."""
    normalized_plane = str(plane).strip().lower()
    if normalized_plane not in {slot.plane for slot in ACQUISITION_SLOTS}:
        return None, "unsupported-plane"
    try:
        fluid = binary_series_attribute(fluid_sensitive, "Fluid_Sensitive")
        fat = binary_series_attribute(fat_suppression, "Fat_Suppression")
    except ValueError:
        return None, "invalid-series-attributes"
    if fluid != fat:
        return None, "mixed-series-attributes"
    matching = [
        slot.name
        for slot in ACQUISITION_SLOTS
        if slot.matches(normalized_plane, fluid, fat)
    ]
    if len(matching) != 1:
        return None, "unmapped-series-attributes"
    return matching[0], None


def resolve_slots(names: Sequence[str]) -> tuple[AcquisitionSlot, ...]:
    """Resolve and validate the configured slot order."""
    requested = tuple(str(name) for name in names)
    if requested != SLOT_NAMES:
        raise ValueError(
            "Acquisition slots must use the fixed six-slot order: "
            f"{list(SLOT_NAMES)}"
        )
    return tuple(SLOTS_BY_NAME[name] for name in requested)


def choose_largest_series(candidates: Sequence[tuple[str, int]]) -> str:
    """Choose the thickest series with a stable UID tie-break."""
    return rank_series_by_size(candidates)[0]


def rank_series_by_size(candidates: Sequence[tuple[str, int]]) -> tuple[str, ...]:
    """Rank series by descending slice count with a stable UID tie-break."""
    if not candidates:
        raise ValueError("At least one candidate series is required")
    return tuple(
        series_uid
        for series_uid, _ in sorted(
            candidates, key=lambda item: (-item[1], item[0])
        )
    )
