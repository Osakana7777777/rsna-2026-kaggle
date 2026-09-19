"""Study-level DICOM loading for the RSNA knee baseline."""

from __future__ import annotations

import hashlib
import json
import lzma
import math
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pydicom
import torch
from torch.nn import functional
from torch.utils.data import Dataset

from rsna2026.acquisition import (
    CLASSIFICATION_SOURCE,
    CLASSIFICATION_METHOD,
    AcquisitionClass,
    binary_series_attribute,
    classify_acquisition,
    match_series_table_slot,
    rank_series_by_size,
    read_acquisition_header,
    resolve_slots,
)
from rsna2026.orientation import (
    ORIENTATION_NORMALIZATION_METHOD,
    OrientationDecision,
    decide_orientation,
    read_orientation_header,
)
from rsna2026.slice_selection import select_physical_windows

if TYPE_CHECKING:
    from rsna2026.image_cache import ImageCacheReader


ORDERING_WORKERS = 16
PREPROCESSING_VERSION = "dicom-physical-fov-window-orientation-uint8-v5"
SLICE_SELECTION_METHOD = "physical-adjacent-three-window-v1"
SERIES_SELECTION_METHOD = "top-k-slice-count-v1"


@dataclass(frozen=True)
class StudyRecord:
    """Paths sampled from one MRI study."""

    study_uid: str
    window_paths: tuple[tuple[Path | None, Path | None, Path | None], ...]
    position_features: tuple[tuple[float, float, float], ...]
    num_slots: int
    max_series_per_slot: int
    selections: tuple["SeriesSelection", ...] = ()
    classifications: tuple["SeriesClassification", ...] = ()
    orientation_audits: tuple["SeriesOrientationAudit", ...] = ()


@dataclass(frozen=True)
class SeriesOrientationAudit:
    """Patient-coordinate orientation decision for one candidate series."""

    series_uid: str
    representative_path: str
    plane: str
    laterality_tag: str | None
    geometry_laterality: str | None
    resolved_laterality: str | None
    laterality_source: str
    horizontal_flip: bool
    tag_geometry_disagreement: bool
    audit_reason: str | None


@dataclass(frozen=True)
class SeriesClassification:
    """Auditable acquisition classification for one candidate series."""

    series_uid: str
    plane: str
    total_slices: int
    weighting: str
    fluid_sensitive: bool | None
    fat_suppressed: bool | None
    classification_source: str
    unassigned_reason: str | None
    dicom_fluid_sensitive: bool | None
    dicom_fat_suppressed: bool | None
    weighting_source: str
    fat_suppression_source: str
    representative_path: str
    matching_slots: tuple[str, ...]


@dataclass(frozen=True)
class SeriesSelection:
    """Auditable slice selection metadata for one acquisition slot."""

    slot: str
    series_rank: int
    plane: str
    series_uid: str | None
    weighting: str | None
    fluid_sensitive: bool | None
    fat_suppressed: bool | None
    classification_source: str
    unassigned_reason: str | None
    dicom_fluid_sensitive: bool | None
    dicom_fat_suppressed: bool | None
    weighting_source: str
    fat_suppression_source: str
    representative_path: str | None
    ordering_method: str
    total_slices: int
    physical_coverage_mm: float
    median_spacing_mm: float
    duplicate_positions: int
    large_gaps: int
    all_positions_mm: tuple[float, ...]
    target_positions_mm: tuple[float, ...]
    selected_center_positions_mm: tuple[float, ...]
    selected_windows: tuple[tuple[str, str, str], ...]
    position_features: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class _SeriesCandidate:
    study_uid: str
    series_uid: str
    plane: str
    series_dir: Path
    paths: tuple[Path, ...]
    acquisition: AcquisitionClass
    fluid_sensitive: bool | None
    fat_suppressed: bool | None
    unassigned_reason: str | None
    representative_path: Path
    matching_slots: tuple[str, ...]
    orientation: OrientationDecision


def build_study_records(
    series_table: pd.DataFrame,
    image_root: Path,
    study_uids: Sequence[str],
    window_count: int,
    max_series_per_slot: int,
    slot_names: Sequence[str],
    classification_method: str,
) -> list[StudyRecord]:
    """Classify series, fill fixed acquisition slots, and sample each stack."""
    if window_count <= 0:
        raise ValueError("Window count must be positive")
    if max_series_per_slot <= 0:
        raise ValueError("Maximum series per slot must be positive")
    if classification_method != CLASSIFICATION_METHOD:
        raise ValueError(
            f"Unsupported series classification method: {classification_method}"
        )
    slots = resolve_slots(slot_names)
    required = {
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "Anatomical_Plane",
        "Fluid_Sensitive",
        "Fat_Suppression",
    }
    missing = required.difference(series_table.columns)
    if missing:
        raise ValueError(f"Series table is missing columns: {sorted(missing)}")

    wanted = {str(uid) for uid in study_uids}
    series_inputs: dict[
        tuple[str, str], tuple[str, object, object, Path, tuple[Path, ...]]
    ] = {}
    for row in series_table.itertuples(index=False):
        study_uid = str(row.StudyInstanceUID)
        if study_uid not in wanted:
            continue
        plane = str(row.Anatomical_Plane).strip().lower()
        series_uid = str(row.SeriesInstanceUID)
        series_dir = image_root / study_uid / series_uid
        paths = (
            tuple(sorted(series_dir.glob("*.dcm")))
            if series_dir.is_dir()
            else ()
        )
        if paths:
            series_inputs[(study_uid, series_uid)] = (
                plane,
                row.Fluid_Sensitive,
                row.Fat_Suppression,
                series_dir,
                paths,
            )

    def classify_series(
        item: tuple[
            tuple[str, str],
            tuple[str, object, object, Path, tuple[Path, ...]],
        ]
    ) -> _SeriesCandidate:
        (study_uid, series_uid), values = item
        plane, raw_fluid, raw_fat, series_dir, paths = values
        representative = paths[len(paths) // 2]
        try:
            acquisition = read_acquisition_header(representative)
        except Exception as error:
            acquisition = AcquisitionClass(
                weighting="unknown",
                fluid_sensitive=False,
                fat_suppressed=False,
                weighting_source=f"read-error:{type(error).__name__}",
                fat_suppression_source=f"read-error:{type(error).__name__}",
            )
        try:
            orientation = read_orientation_header(representative)
        except Exception as error:
            orientation = OrientationDecision(
                plane="unknown",
                laterality_tag=None,
                geometry_laterality=None,
                resolved_laterality=None,
                laterality_source="unresolved",
                horizontal_flip=False,
                tag_geometry_disagreement=False,
                audit_reason=f"read-error:{type(error).__name__}",
            )
        slot_name, unassigned_reason = match_series_table_slot(
            plane, raw_fluid, raw_fat
        )
        try:
            fluid_sensitive = binary_series_attribute(
                raw_fluid, "Fluid_Sensitive"
            )
            fat_suppressed = binary_series_attribute(
                raw_fat, "Fat_Suppression"
            )
        except ValueError:
            fluid_sensitive = None
            fat_suppressed = None
        matching = (slot_name,) if slot_name in {slot.name for slot in slots} else ()
        return _SeriesCandidate(
            study_uid=study_uid,
            series_uid=series_uid,
            plane=plane,
            series_dir=series_dir,
            paths=paths,
            acquisition=acquisition,
            fluid_sensitive=fluid_sensitive,
            fat_suppressed=fat_suppressed,
            unassigned_reason=unassigned_reason,
            representative_path=representative,
            matching_slots=matching,
            orientation=orientation,
        )

    ordered_inputs = sorted(series_inputs.items())
    with ThreadPoolExecutor(max_workers=ORDERING_WORKERS) as pool:
        classified = list(pool.map(classify_series, ordered_inputs))

    candidates: dict[str, dict[str, list[_SeriesCandidate]]] = {}
    classifications: dict[str, list[SeriesClassification]] = {}
    orientation_audits: dict[str, list[SeriesOrientationAudit]] = {}
    for candidate in classified:
        relative_header = (
            f"{candidate.study_uid}/{candidate.series_uid}/"
            f"{candidate.representative_path.name}"
        )
        classifications.setdefault(candidate.study_uid, []).append(
            SeriesClassification(
                series_uid=candidate.series_uid,
                plane=candidate.plane,
                total_slices=len(candidate.paths),
                weighting=candidate.acquisition.weighting,
                fluid_sensitive=candidate.fluid_sensitive,
                fat_suppressed=candidate.fat_suppressed,
                classification_source=CLASSIFICATION_SOURCE,
                unassigned_reason=candidate.unassigned_reason,
                dicom_fluid_sensitive=candidate.acquisition.fluid_sensitive,
                dicom_fat_suppressed=candidate.acquisition.fat_suppressed,
                weighting_source=candidate.acquisition.weighting_source,
                fat_suppression_source=(
                    candidate.acquisition.fat_suppression_source
                ),
                representative_path=relative_header,
                matching_slots=candidate.matching_slots,
            )
        )
        orientation_audits.setdefault(candidate.study_uid, []).append(
            SeriesOrientationAudit(
                series_uid=candidate.series_uid,
                representative_path=relative_header,
                plane=candidate.orientation.plane,
                laterality_tag=candidate.orientation.laterality_tag,
                geometry_laterality=candidate.orientation.geometry_laterality,
                resolved_laterality=candidate.orientation.resolved_laterality,
                laterality_source=candidate.orientation.laterality_source,
                horizontal_flip=candidate.orientation.horizontal_flip,
                tag_geometry_disagreement=(
                    candidate.orientation.tag_geometry_disagreement
                ),
                audit_reason=candidate.orientation.audit_reason,
            )
        )
        for slot_name in candidate.matching_slots:
            candidates.setdefault(candidate.study_uid, {}).setdefault(
                slot_name, []
            ).append(candidate)

    chosen: dict[str, dict[str, tuple[_SeriesCandidate, ...]]] = {}
    for study_uid, slots_for_study in candidates.items():
        for slot_name, slot_candidates in slots_for_study.items():
            selected_uids = rank_series_by_size(
                [
                    (candidate.series_uid, len(candidate.paths))
                    for candidate in slot_candidates
                ]
            )[:max_series_per_slot]
            by_uid = {
                candidate.series_uid: candidate for candidate in slot_candidates
            }
            chosen.setdefault(study_uid, {})[slot_name] = tuple(
                by_uid[series_uid] for series_uid in selected_uids
            )

    series_directories = sorted(
        {
            candidate.series_dir
            for slots_for_study in chosen.values()
            for slot_candidates in slots_for_study.values()
            for candidate in slot_candidates
        }
    )

    def sample_series(series_dir: Path):
        paths = list(series_dir.glob("*.dcm"))
        return paths, select_physical_windows(paths, window_count)

    with ThreadPoolExecutor(max_workers=ORDERING_WORKERS) as pool:
        sampled_series = dict(
            zip(series_directories, pool.map(sample_series, series_directories))
        )

    records = []
    for uid in study_uids:
        study_uid = str(uid)
        selected: list[tuple[Path | None, Path | None, Path | None]] = []
        position_features: list[tuple[float, float, float]] = []
        selections = []
        for slot in slots:
            slot_candidates = chosen.get(study_uid, {}).get(slot.name, ())
            for series_index in range(max_series_per_slot):
                candidate = (
                    slot_candidates[series_index]
                    if series_index < len(slot_candidates)
                    else None
                )
                if candidate is None:
                    paths = []
                    sample = select_physical_windows([], window_count)
                else:
                    paths, sample = sampled_series[candidate.series_dir]
                padded_windows = list(sample.windows)
                padded_features = list(sample.position_features)
                padded_windows.extend(
                    [(None, None, None)] * (window_count - len(padded_windows))
                )
                padded_features.extend(
                    [(0.0, 0.0, 0.0)] * (window_count - len(padded_features))
                )
                selected.extend(padded_windows)
                position_features.extend(padded_features)
                series_uid = (
                    candidate.series_uid if candidate is not None else None
                )
                acquisition = (
                    candidate.acquisition if candidate is not None else None
                )
                selections.append(
                    SeriesSelection(
                        slot=slot.name,
                        series_rank=series_index + 1,
                        plane=slot.plane,
                        series_uid=series_uid,
                        weighting=(
                            acquisition.weighting if acquisition else None
                        ),
                        fluid_sensitive=(
                            candidate.fluid_sensitive if candidate else None
                        ),
                        fat_suppressed=(
                            candidate.fat_suppressed if candidate else None
                        ),
                        classification_source=(
                            CLASSIFICATION_SOURCE if candidate else "missing"
                        ),
                        unassigned_reason=(
                            candidate.unassigned_reason if candidate else "missing"
                        ),
                        dicom_fluid_sensitive=(
                            acquisition.fluid_sensitive if acquisition else None
                        ),
                        dicom_fat_suppressed=(
                            acquisition.fat_suppressed if acquisition else None
                        ),
                        weighting_source=(
                            acquisition.weighting_source
                            if acquisition
                            else "missing"
                        ),
                        fat_suppression_source=(
                            acquisition.fat_suppression_source
                            if acquisition
                            else "missing"
                        ),
                        representative_path=(
                            f"{study_uid}/{series_uid}/"
                            f"{candidate.representative_path.name}"
                            if candidate is not None
                            else None
                        ),
                        ordering_method=sample.method,
                        total_slices=len(paths),
                        physical_coverage_mm=sample.coverage_mm,
                        median_spacing_mm=sample.median_spacing_mm,
                        duplicate_positions=sample.duplicate_positions,
                        large_gaps=sample.large_gaps,
                        all_positions_mm=sample.all_positions_mm,
                        target_positions_mm=sample.target_positions_mm,
                        selected_center_positions_mm=sample.center_positions_mm,
                        selected_windows=tuple(
                            tuple(
                                f"{study_uid}/{series_uid}/{path.name}"
                                for path in window
                            )
                            for window in sample.windows
                        ),
                        position_features=sample.position_features,
                    )
                )
        records.append(
            StudyRecord(
                study_uid=study_uid,
                window_paths=tuple(selected),
                position_features=tuple(position_features),
                num_slots=len(slots),
                max_series_per_slot=max_series_per_slot,
                selections=tuple(selections),
                classifications=tuple(
                    sorted(
                        classifications.get(study_uid, []),
                        key=lambda item: item.series_uid,
                    )
                ),
                orientation_audits=tuple(
                    sorted(
                        orientation_audits.get(study_uid, []),
                        key=lambda item: item.series_uid,
                    )
                ),
            )
        )
    return records


def build_cached_study_records(
    series_table: pd.DataFrame,
    cache_reader: "ImageCacheReader",
    study_uids: Sequence[str],
    window_count: int,
    max_series_per_slot: int,
    slot_names: Sequence[str],
    classification_method: str,
) -> list[StudyRecord]:
    """Build slot selections from competition rows and cached physical windows."""
    if classification_method != CLASSIFICATION_METHOD:
        raise ValueError(
            f"Unsupported series classification method: {classification_method}"
        )
    if window_count <= 0 or max_series_per_slot <= 0:
        raise ValueError("Window and maximum-series counts must be positive")
    slots = resolve_slots(slot_names)
    required = {
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "Anatomical_Plane",
        "Fluid_Sensitive",
        "Fat_Suppression",
    }
    missing = required.difference(series_table.columns)
    if missing:
        raise ValueError(f"Series table is missing columns: {sorted(missing)}")

    wanted = {str(uid) for uid in study_uids}
    table_rows: dict[tuple[str, str], object] = {}
    for row in series_table.itertuples(index=False):
        key = (str(row.StudyInstanceUID), str(row.SeriesInstanceUID))
        if key[0] not in wanted:
            continue
        if key in table_rows:
            raise ValueError(f"Duplicate series-table row: {key}")
        table_rows[key] = row

    records = []
    for uid in study_uids:
        study_uid = str(uid)
        study_metadata = cache_reader.study_metadata(study_uid)
        cached_series = {
            str(series["series_uid"]): series
            for series in study_metadata["series"]  # type: ignore[index]
        }
        expected_keys = {
            series_uid
            for row_study_uid, series_uid in table_rows
            if row_study_uid == study_uid
        }
        if set(cached_series) != expected_keys:
            raise ValueError(f"Cached series set changed for study {study_uid}")
        cached_paths = {
            str(path) for path in study_metadata["relative_paths"]  # type: ignore[index]
        }

        classifications = []
        orientation_audits = []
        candidates: dict[str, list[tuple[str, Mapping[str, object], object]]] = {}
        classified_values: dict[
            str, tuple[object, AcquisitionClass, str | None, bool | None, bool | None]
        ] = {}
        for series_uid, metadata in sorted(cached_series.items()):
            row = table_rows[(study_uid, series_uid)]
            slot_name, unassigned_reason = match_series_table_slot(
                row.Anatomical_Plane,
                row.Fluid_Sensitive,
                row.Fat_Suppression,
            )
            try:
                fluid_sensitive = binary_series_attribute(
                    row.Fluid_Sensitive, "Fluid_Sensitive"
                )
                fat_suppressed = binary_series_attribute(
                    row.Fat_Suppression, "Fat_Suppression"
                )
            except ValueError:
                fluid_sensitive = None
                fat_suppressed = None
            acquisition = classify_acquisition(metadata["acquisition_header"])
            orientation_header = metadata["acquisition_header"]
            orientation = decide_orientation(
                orientation_header,
                (
                    int(orientation_header.get("Rows", 0) or 0),
                    int(orientation_header.get("Columns", 0) or 0),
                ),
            )
            matching_slots = (
                (slot_name,)
                if slot_name in {slot.name for slot in slots}
                else ()
            )
            classified_values[series_uid] = (
                row,
                acquisition,
                unassigned_reason,
                fluid_sensitive,
                fat_suppressed,
            )
            classifications.append(
                SeriesClassification(
                    series_uid=series_uid,
                    plane=str(row.Anatomical_Plane).strip().lower(),
                    total_slices=int(metadata["total_slices"]),
                    weighting=acquisition.weighting,
                    fluid_sensitive=fluid_sensitive,
                    fat_suppressed=fat_suppressed,
                    classification_source=CLASSIFICATION_SOURCE,
                    unassigned_reason=unassigned_reason,
                    dicom_fluid_sensitive=acquisition.fluid_sensitive,
                    dicom_fat_suppressed=acquisition.fat_suppressed,
                    weighting_source=acquisition.weighting_source,
                    fat_suppression_source=acquisition.fat_suppression_source,
                    representative_path=str(metadata["representative_path"]),
                    matching_slots=matching_slots,
                )
            )
            orientation_audits.append(
                SeriesOrientationAudit(
                    series_uid=series_uid,
                    representative_path=str(metadata["representative_path"]),
                    plane=orientation.plane,
                    laterality_tag=orientation.laterality_tag,
                    geometry_laterality=orientation.geometry_laterality,
                    resolved_laterality=orientation.resolved_laterality,
                    laterality_source=orientation.laterality_source,
                    horizontal_flip=orientation.horizontal_flip,
                    tag_geometry_disagreement=(
                        orientation.tag_geometry_disagreement
                    ),
                    audit_reason=orientation.audit_reason,
                )
            )
            if slot_name in matching_slots:
                candidates.setdefault(slot_name, []).append(
                    (series_uid, metadata, row)
                )

        selected_windows: list[
            tuple[Path | None, Path | None, Path | None]
        ] = []
        position_features: list[tuple[float, float, float]] = []
        selections = []
        for slot in slots:
            slot_candidates = candidates.get(slot.name, [])
            ordered_uids = rank_series_by_size(
                [
                    (series_uid, int(metadata["total_slices"]))
                    for series_uid, metadata, _ in slot_candidates
                ]
            ) if slot_candidates else ()
            by_uid = {
                series_uid: (metadata, row)
                for series_uid, metadata, row in slot_candidates
            }
            chosen_uids = ordered_uids[:max_series_per_slot]
            for series_index in range(max_series_per_slot):
                if series_index < len(chosen_uids):
                    series_uid = chosen_uids[series_index]
                    metadata, row = by_uid[series_uid]
                    (
                        _,
                        acquisition,
                        unassigned_reason,
                        fluid_sensitive,
                        fat_suppressed,
                    ) = classified_values[series_uid]
                    windows = tuple(
                        tuple(Path(str(path)) for path in window)
                        for window in metadata["selected_windows"]
                    )
                    features = tuple(
                        tuple(float(value) for value in feature)
                        for feature in metadata["position_features"]
                    )
                    if len(windows) > window_count or len(features) != len(windows):
                        raise ValueError(
                            f"Cached physical-window layout changed for {series_uid}"
                        )
                    referenced_paths = {
                        path.as_posix() for window in windows for path in window
                    }
                    if any(len(window) != 3 for window in windows):
                        raise ValueError(
                            f"Cached window size changed for {series_uid}"
                        )
                    if not referenced_paths.issubset(cached_paths):
                        raise ValueError(
                            f"Cached window paths changed for {series_uid}"
                        )
                else:
                    series_uid = None
                    metadata = None
                    acquisition = None
                    unassigned_reason = "missing"
                    fluid_sensitive = None
                    fat_suppressed = None
                    windows = ()
                    features = ()
                selected_windows.extend(windows)
                selected_windows.extend(
                    [(None, None, None)] * (window_count - len(windows))
                )
                position_features.extend(features)
                position_features.extend(
                    [(0.0, 0.0, 0.0)] * (window_count - len(features))
                )
                selections.append(
                    SeriesSelection(
                        slot=slot.name,
                        series_rank=series_index + 1,
                        plane=slot.plane,
                        series_uid=series_uid,
                        weighting=(acquisition.weighting if acquisition else None),
                        fluid_sensitive=fluid_sensitive,
                        fat_suppressed=fat_suppressed,
                        classification_source=(
                            CLASSIFICATION_SOURCE if metadata else "missing"
                        ),
                        unassigned_reason=unassigned_reason,
                        dicom_fluid_sensitive=(
                            acquisition.fluid_sensitive if acquisition else None
                        ),
                        dicom_fat_suppressed=(
                            acquisition.fat_suppressed if acquisition else None
                        ),
                        weighting_source=(
                            acquisition.weighting_source if acquisition else "missing"
                        ),
                        fat_suppression_source=(
                            acquisition.fat_suppression_source
                            if acquisition
                            else "missing"
                        ),
                        representative_path=(
                            str(metadata["representative_path"])
                            if metadata
                            else None
                        ),
                        ordering_method=(
                            str(metadata["ordering_method"])
                            if metadata
                            else "missing"
                        ),
                        total_slices=(
                            int(metadata["total_slices"]) if metadata else 0
                        ),
                        physical_coverage_mm=(
                            float(metadata["physical_coverage_mm"])
                            if metadata
                            else 0.0
                        ),
                        median_spacing_mm=(
                            float(metadata["median_spacing_mm"])
                            if metadata
                            else 0.0
                        ),
                        duplicate_positions=(
                            int(metadata["duplicate_positions"])
                            if metadata
                            else 0
                        ),
                        large_gaps=(
                            int(metadata["large_gaps"]) if metadata else 0
                        ),
                        all_positions_mm=(
                            tuple(float(value) for value in metadata["all_positions_mm"])
                            if metadata
                            else ()
                        ),
                        target_positions_mm=(
                            tuple(
                                float(value)
                                for value in metadata["target_positions_mm"]
                            )
                            if metadata
                            else ()
                        ),
                        selected_center_positions_mm=(
                            tuple(
                                float(value)
                                for value in metadata["selected_center_positions_mm"]
                            )
                            if metadata
                            else ()
                        ),
                        selected_windows=tuple(
                            tuple(path.as_posix() for path in window)
                            for window in windows
                        ),
                        position_features=features,
                    )
                )
        records.append(
            StudyRecord(
                study_uid=study_uid,
                window_paths=tuple(selected_windows),
                position_features=tuple(position_features),
                num_slots=len(slots),
                max_series_per_slot=max_series_per_slot,
                selections=tuple(selections),
                classifications=tuple(classifications),
                orientation_audits=tuple(orientation_audits),
            )
        )
    return records


def load_study_records_cache(
    cache_path: Path,
    image_root: Path,
    study_uids: Sequence[str],
    window_count: int,
    max_series_per_slot: int,
    slot_names: Sequence[str],
    classification_method: str,
    expected_sha256: str,
    expected_study_uids_sha256: str,
) -> tuple[list[StudyRecord], dict[str, object]]:
    """Restore and validate a pinned slice-selection report without DICOM scans."""
    stored = cache_path.read_bytes()
    raw = lzma.decompress(stored) if cache_path.suffix == ".xz" else stored
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Study-record cache SHA-256 mismatch: {actual_sha256}"
        )
    report = json.loads(raw)
    if report.get("series_classification") != classification_method:
        raise ValueError("Study-record cache classification method changed")
    if report.get("series_selection_method") != SERIES_SELECTION_METHOD:
        raise ValueError("Study-record cache series selection method changed")
    if report.get("slots") != list(slot_names):
        raise ValueError("Study-record cache slot order changed")
    if int(report.get("max_series_per_slot", -1)) != max_series_per_slot:
        raise ValueError("Study-record cache maximum series per slot changed")
    if report.get("slice_selection_method") != SLICE_SELECTION_METHOD:
        raise ValueError("Study-record cache selection method changed")
    if int(report.get("window_count", -1)) != window_count:
        raise ValueError("Study-record cache window count changed")
    if int(report.get("window_size", -1)) != 3:
        raise ValueError("Study-record cache window size changed")
    if (
        report.get("orientation_normalization")
        != ORIENTATION_NORMALIZATION_METHOD
        or report.get("medial_side") != "image-right"
    ):
        raise ValueError("Study-record cache orientation normalization changed")

    split = report.get("splits", {}).get("all-training-studies", {})
    entries = split.get("studies", [])
    entries_by_uid = {str(entry["study_uid"]): entry for entry in entries}
    expected_uids = [str(uid) for uid in study_uids]
    actual_uids = set(entries_by_uid)
    if actual_uids != set(expected_uids) or len(entries) != len(expected_uids):
        raise ValueError("Study-record cache UID set changed")
    uid_material = "\n".join(sorted(actual_uids)) + "\n"
    actual_uid_sha256 = hashlib.sha256(uid_material.encode("utf-8")).hexdigest()
    if actual_uid_sha256 != expected_study_uids_sha256:
        raise ValueError(
            f"Study-record cache UID SHA-256 mismatch: {actual_uid_sha256}"
        )

    records = []
    for study_uid in expected_uids:
        entry = entries_by_uid[study_uid]
        selections = []
        window_paths: list[tuple[Path | None, Path | None, Path | None]] = []
        position_features: list[tuple[float, float, float]] = []
        series_rows = entry.get("series", [])
        expected_layout = [
            (slot, rank)
            for slot in slot_names
            for rank in range(1, max_series_per_slot + 1)
        ]
        actual_layout = [
            (row.get("slot"), row.get("series_rank")) for row in series_rows
        ]
        if actual_layout != expected_layout:
            raise ValueError(f"Cached series layout changed for study {study_uid}")
        for row in series_rows:
            selected_windows = tuple(
                tuple(str(path) for path in window)
                for window in row["selected_windows"]
            )
            features = tuple(
                tuple(float(value) for value in feature)
                for feature in row["position_features"]
            )
            if row["series_uid"] is None:
                if selected_windows:
                    raise ValueError("Missing cached series contains selected windows")
            else:
                if len(selected_windows) > window_count:
                    raise ValueError("Cached series has too many selected windows")
            if len(features) != len(selected_windows):
                raise ValueError("Cached position-feature count changed")
            window_paths.extend(
                tuple(image_root / path for path in window)
                for window in selected_windows
            )
            window_paths.extend(
                [(None, None, None)] * (window_count - len(selected_windows))
            )
            position_features.extend(features)
            position_features.extend(
                [(0.0, 0.0, 0.0)] * (window_count - len(features))
            )
            selections.append(
                SeriesSelection(
                    slot=str(row["slot"]),
                    series_rank=int(row["series_rank"]),
                    plane=str(row["plane"]),
                    series_uid=(
                        str(row["series_uid"])
                        if row["series_uid"] is not None
                        else None
                    ),
                    weighting=row["weighting"],
                    fluid_sensitive=row["fluid_sensitive"],
                    fat_suppressed=row["fat_suppressed"],
                    classification_source=str(
                        row.get("classification_source", CLASSIFICATION_SOURCE)
                    ),
                    unassigned_reason=row.get("unassigned_reason"),
                    dicom_fluid_sensitive=row.get("dicom_fluid_sensitive"),
                    dicom_fat_suppressed=row.get("dicom_fat_suppressed"),
                    weighting_source=str(row["weighting_source"]),
                    fat_suppression_source=str(row["fat_suppression_source"]),
                    representative_path=row["representative_path"],
                    ordering_method=str(row["ordering_method"]),
                    total_slices=int(row["total_slices"]),
                    physical_coverage_mm=float(row["physical_coverage_mm"]),
                    median_spacing_mm=float(row["median_spacing_mm"]),
                    duplicate_positions=int(row["duplicate_positions"]),
                    large_gaps=int(row["large_gaps"]),
                    all_positions_mm=tuple(
                        float(value) for value in row["all_positions_mm"]
                    ),
                    target_positions_mm=tuple(
                        float(value) for value in row["target_positions_mm"]
                    ),
                    selected_center_positions_mm=tuple(
                        float(value)
                        for value in row["selected_center_positions_mm"]
                    ),
                    selected_windows=selected_windows,
                    position_features=features,
                )
            )
        classifications = tuple(
            SeriesClassification(
                series_uid=str(row["series_uid"]),
                plane=str(row["plane"]),
                total_slices=int(row["total_slices"]),
                weighting=str(row["weighting"]),
                fluid_sensitive=bool(row["fluid_sensitive"]),
                fat_suppressed=bool(row["fat_suppressed"]),
                classification_source=str(
                    row.get("classification_source", CLASSIFICATION_SOURCE)
                ),
                unassigned_reason=row.get("unassigned_reason"),
                dicom_fluid_sensitive=row.get("dicom_fluid_sensitive"),
                dicom_fat_suppressed=row.get("dicom_fat_suppressed"),
                weighting_source=str(row["weighting_source"]),
                fat_suppression_source=str(row["fat_suppression_source"]),
                representative_path=str(row["representative_path"]),
                matching_slots=tuple(str(value) for value in row["matching_slots"]),
            )
            for row in entry.get("series_classifications", [])
        )
        orientation_audits = tuple(
            SeriesOrientationAudit(
                series_uid=str(row["series_uid"]),
                representative_path=str(row["representative_path"]),
                plane=str(row["plane"]),
                laterality_tag=row.get("laterality_tag"),
                geometry_laterality=row.get("geometry_laterality"),
                resolved_laterality=row.get("resolved_laterality"),
                laterality_source=str(row["laterality_source"]),
                horizontal_flip=bool(row["horizontal_flip"]),
                tag_geometry_disagreement=bool(
                    row["tag_geometry_disagreement"]
                ),
                audit_reason=row.get("audit_reason"),
            )
            for row in entry.get("orientation_audits", [])
        )
        records.append(
            StudyRecord(
                study_uid=study_uid,
                window_paths=tuple(window_paths),
                position_features=tuple(position_features),
                num_slots=len(slot_names),
                max_series_per_slot=max_series_per_slot,
                selections=tuple(selections),
                classifications=classifications,
                orientation_audits=orientation_audits,
            )
        )
    return records, dict(split.get("summary", {}))


def center_crop_physical_fov(
    pixels: np.ndarray,
    pixel_spacing: Sequence[float],
    physical_fov_mm: float,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Center-crop a physical field of view and report required zero padding."""
    if pixels.ndim != 2:
        raise ValueError("Physical FOV cropping requires a two-dimensional image")
    if not math.isfinite(physical_fov_mm) or physical_fov_mm <= 0:
        raise ValueError("Physical FOV must be finite and positive")
    try:
        row_spacing, column_spacing = (
            float(pixel_spacing[0]),
            float(pixel_spacing[1]),
        )
    except (IndexError, TypeError, ValueError) as error:
        raise ValueError("PixelSpacing must contain two numeric values") from error
    if not all(
        math.isfinite(value) and value > 0
        for value in (row_spacing, column_spacing)
    ):
        raise ValueError("PixelSpacing values must be finite and positive")

    target_rows = max(1, math.floor(physical_fov_mm / row_spacing + 0.5))
    target_columns = max(
        1, math.floor(physical_fov_mm / column_spacing + 0.5)
    )
    rows, columns = pixels.shape
    crop_rows = min(rows, target_rows)
    crop_columns = min(columns, target_columns)
    row_start = (rows - crop_rows) // 2
    column_start = (columns - crop_columns) // 2
    cropped = pixels[
        row_start : row_start + crop_rows,
        column_start : column_start + crop_columns,
    ]

    missing_rows = target_rows - crop_rows
    missing_columns = target_columns - crop_columns
    top = missing_rows // 2
    left = missing_columns // 2
    padding = (
        top,
        missing_rows - top,
        left,
        missing_columns - left,
    )
    return cropped, padding


def preprocess_physical_fov(
    pixels: np.ndarray,
    pixel_spacing: Sequence[float],
    physical_fov_mm: float,
    invert: bool,
) -> np.ndarray:
    """Sanitize, crop, window, and pad one image at a fixed physical scale."""
    finite = np.isfinite(pixels)
    if not finite.any():
        raise ValueError("Image contains no finite pixels")
    fill = float(np.median(pixels[finite]))
    sanitized = np.nan_to_num(pixels, nan=fill, posinf=fill, neginf=fill)
    cropped, (top, bottom, left, right) = center_crop_physical_fov(
        sanitized, pixel_spacing, physical_fov_mm
    )
    low, high = np.percentile(cropped, (1.0, 99.0))
    normalized = np.clip(
        (cropped - low) / max(float(high - low), 1e-6), 0.0, 1.0
    )
    if invert:
        normalized = 1.0 - normalized
    if top or bottom or left or right:
        normalized = np.pad(
            normalized,
            ((top, bottom), (left, right)),
            mode="constant",
            constant_values=0.0,
        )
    return normalized.astype(np.float32, copy=False)


def read_dicom(
    path: Path, image_size: int, physical_fov_mm: float
) -> torch.Tensor:
    """Decode, physically crop, robustly window, and resize one DICOM image."""
    dataset = pydicom.dcmread(path, force=True)
    pixels = np.asarray(dataset.pixel_array, dtype=np.float32)
    if pixels.ndim > 2:
        pixels = pixels[pixels.shape[0] // 2]
    slope = float(getattr(dataset, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0) or 0.0)
    pixels = pixels * slope + intercept
    orientation = decide_orientation(
        {
            "ImageOrientationPatient": getattr(
                dataset, "ImageOrientationPatient", None
            ),
            "ImagePositionPatient": getattr(dataset, "ImagePositionPatient", None),
            "PixelSpacing": getattr(dataset, "PixelSpacing", None),
            "Laterality": getattr(dataset, "Laterality", None),
        },
        pixels.shape,
    )
    if orientation.horizontal_flip:
        pixels = np.flip(pixels, axis=1)
    pixels = preprocess_physical_fov(
        pixels,
        getattr(dataset, "PixelSpacing", None),
        physical_fov_mm,
        str(getattr(dataset, "PhotometricInterpretation", "")) == "MONOCHROME1",
    )

    tensor = torch.from_numpy(np.ascontiguousarray(pixels)).view(1, 1, *pixels.shape)
    tensor = functional.interpolate(
        tensor, size=(image_size, image_size), mode="bilinear", align_corners=False
    )
    return tensor.squeeze(0)


class KneeStudyDataset(
    Dataset[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ]
):
    """Load a fixed-size bag of MRI slices for each study."""

    def __init__(
        self,
        records: Sequence[StudyRecord],
        image_size: int,
        physical_fov_mm: float,
        labels: np.ndarray | None = None,
        label_weights: np.ndarray | None = None,
        cache_dir: str | Path | None = None,
        image_cache: "ImageCacheReader | None" = None,
    ) -> None:
        self.records = list(records)
        self.image_size = image_size
        self.physical_fov_mm = physical_fov_mm
        self.labels = labels
        self.label_weights = label_weights
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.image_cache = image_cache
        if self.cache_dir is not None and self.image_cache is not None:
            raise ValueError("Ephemeral and persistent image caches are exclusive")
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        if (labels is None) != (label_weights is None):
            raise ValueError("Labels and label weights must be supplied together")
        if labels is not None and len(labels) != len(self.records):
            raise ValueError("Label count does not match study records")
        if label_weights is not None and label_weights.shape != labels.shape:
            raise ValueError("Label weights must have the same shape as labels")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self, index: int
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        record = self.records[index]
        images, present = self._load_images(record)
        num_slots = record.num_slots
        max_series = record.max_series_per_slot
        series_count = num_slots * max_series
        if (
            num_slots <= 0
            or max_series <= 0
            or len(record.selections) != series_count
            or len(record.window_paths) % series_count
        ):
            raise ValueError("Study record has an invalid window layout")
        window_count = len(record.window_paths) // series_count
        images = images.reshape(
            num_slots,
            max_series,
            window_count,
            3,
            self.image_size,
            self.image_size,
        )
        present = present.reshape(num_slots, max_series, window_count)
        positions = torch.tensor(record.position_features, dtype=torch.float32).reshape(
            num_slots, max_series, window_count, 3
        )

        label = (
            torch.from_numpy(self.labels[index]).float()
            if self.labels is not None
            else torch.empty(0)
        )
        label_weight = (
            torch.from_numpy(self.label_weights[index]).float()
            if self.label_weights is not None
            else torch.empty(0)
        )
        return images, present, positions, label, label_weight

    def _load_images(self, record: StudyRecord) -> tuple[torch.Tensor, torch.Tensor]:
        """Load one study, reusing an ephemeral preprocessed cache when enabled."""
        if self.image_cache is not None:
            return self._load_persistent_images(record)
        flat_paths = [path for window in record.window_paths for path in window]
        unique_paths = tuple(
            dict.fromkeys(path for path in flat_paths if path is not None)
        )
        path_indices = {path: index for index, path in enumerate(unique_paths)}
        inverse = tuple(
            path_indices[path] if path is not None else -1 for path in flat_paths
        )
        cache_path = self._cache_path(record)
        if cache_path is not None and cache_path.is_file():
            try:
                with np.load(cache_path, allow_pickle=False) as cached:
                    unique_present = torch.from_numpy(
                        cached["present"].astype(np.bool_)
                    )
                    present_images = torch.from_numpy(
                        np.asarray(cached["images"], dtype=np.float32)
                    ).div_(255.0)
                image_shape = (1, self.image_size, self.image_size)
                if tuple(unique_present.shape) != (len(unique_paths),) or tuple(
                    present_images.shape
                ) != (int(unique_present.sum()), *image_shape):
                    raise ValueError("cached tensor shape does not match the study")
                unique_images = torch.zeros((len(unique_paths), *image_shape))
                unique_images[unique_present] = present_images
                return self._assemble_windows(unique_images, unique_present, inverse)
            except (OSError, ValueError, KeyError) as error:
                print(f"Warning: ignoring invalid DICOM cache {cache_path}: {error}")

        images = []
        present = []
        for path in unique_paths:
            try:
                image = read_dicom(path, self.image_size, self.physical_fov_mm)
            except Exception as error:
                print(f"Warning: failed to decode {path}: {error}", flush=True)
                image = None
            if image is None:
                image = torch.zeros((1, self.image_size, self.image_size))
                present.append(False)
            else:
                present.append(True)
            images.append(image.float())
        image_tensor = (
            torch.stack(images)
            if images
            else torch.empty((0, 1, self.image_size, self.image_size))
        )
        present_tensor = torch.tensor(present)
        if cache_path is not None:
            self._write_cache(cache_path, image_tensor, present_tensor)
        return self._assemble_windows(image_tensor, present_tensor, inverse)

    def _load_persistent_images(
        self, record: StudyRecord
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load selected paths strictly from one persistent study archive."""
        if self.image_cache is None:
            raise RuntimeError("Persistent image cache is unavailable")

        def cache_key(path: Path) -> str:
            """Normalize absolute competition paths to cache-relative keys."""
            parts = path.parts
            try:
                study_index = parts.index(record.study_uid)
            except ValueError:
                return path.as_posix()
            return "/".join(parts[study_index:])

        flat_paths = [path for window in record.window_paths for path in window]
        unique_paths = tuple(
            dict.fromkeys(cache_key(path) for path in flat_paths if path is not None)
        )
        inverse_by_path = {path: index for index, path in enumerate(unique_paths)}
        inverse = tuple(
            inverse_by_path[cache_key(path)] if path is not None else -1
            for path in flat_paths
        )
        cached_images, cached_present, cached_paths = self.image_cache.load_study(
            record.study_uid
        )
        cached_indices = {path: index for index, path in enumerate(cached_paths)}
        missing = [path for path in unique_paths if path not in cached_indices]
        if missing:
            raise ValueError(
                f"Selected paths are absent from persistent image cache: {missing[:3]}"
            )
        selected_indices = [cached_indices[path] for path in unique_paths]
        if selected_indices:
            image_tensor = torch.from_numpy(
                np.ascontiguousarray(cached_images[selected_indices])
            ).unsqueeze(1).float().div_(255.0)
            present_tensor = torch.from_numpy(
                np.ascontiguousarray(cached_present[selected_indices])
            )
        else:
            image_tensor = torch.empty(
                (0, 1, self.image_size, self.image_size), dtype=torch.float32
            )
            present_tensor = torch.empty(0, dtype=torch.bool)
        return self._assemble_windows(image_tensor, present_tensor, inverse)

    def _assemble_windows(
        self,
        unique_images: torch.Tensor,
        unique_present: torch.Tensor,
        inverse: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images = torch.zeros(
            (len(inverse), 1, self.image_size, self.image_size),
            dtype=torch.float32,
        )
        channels_present = torch.zeros(len(inverse), dtype=torch.bool)
        for output_index, source_index in enumerate(inverse):
            if source_index >= 0 and bool(unique_present[source_index]):
                images[output_index] = unique_images[source_index]
                channels_present[output_index] = True
        return images.reshape(-1, 3, self.image_size, self.image_size), (
            channels_present.reshape(-1, 3).all(dim=1)
        )

    def _cache_path(self, record: StudyRecord) -> Path | None:
        if self.cache_dir is None:
            return None
        identity = "\n".join(
            [
                PREPROCESSING_VERSION,
                str(self.image_size),
                str(self.physical_fov_mm),
                *(
                    str(path) if path is not None else "<missing>"
                    for window in record.window_paths
                    for path in window
                ),
            ]
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / f"{digest}.npz"

    @staticmethod
    def _write_cache(
        cache_path: Path, images: torch.Tensor, present: torch.Tensor
    ) -> None:
        """Atomically save preprocessed images without publishing medical data."""
        temporary = cache_path.with_name(
            f".{cache_path.name}.{os.getpid()}.temporary"
        )
        try:
            with temporary.open("wb") as file:
                np.savez_compressed(
                    file,
                    images=(images[present].mul(255.0).round().byte().numpy()),
                    present=present.numpy().astype(np.bool_),
                )
            os.replace(temporary, cache_path)
        except OSError as error:
            temporary.unlink(missing_ok=True)
            print(f"Warning: could not write DICOM cache {cache_path}: {error}")
