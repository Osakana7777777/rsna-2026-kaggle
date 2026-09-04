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

import numpy as np
import pandas as pd
import pydicom
import torch
from torch.nn import functional
from torch.utils.data import Dataset

from rsna2026.acquisition import (
    CLASSIFICATION_METHOD,
    AcquisitionClass,
    choose_largest_series,
    read_acquisition_header,
    resolve_slots,
)
from rsna2026.slice_selection import central_band_indices, sample_ordered_paths


ORDERING_WORKERS = 16
PREPROCESSING_VERSION = "dicom-physical-fov-window-uint8-v3"


@dataclass(frozen=True)
class StudyRecord:
    """Paths sampled from one MRI study."""

    study_uid: str
    image_paths: tuple[Path | None, ...]
    selections: tuple["SeriesSelection", ...] = ()
    classifications: tuple["SeriesClassification", ...] = ()


@dataclass(frozen=True)
class SeriesClassification:
    """Auditable acquisition classification for one candidate series."""

    series_uid: str
    plane: str
    total_slices: int
    weighting: str
    fluid_sensitive: bool
    fat_suppressed: bool
    weighting_source: str
    fat_suppression_source: str
    representative_path: str
    matching_slots: tuple[str, ...]


@dataclass(frozen=True)
class SeriesSelection:
    """Auditable slice selection metadata for one acquisition slot."""

    slot: str
    plane: str
    series_uid: str | None
    weighting: str | None
    fluid_sensitive: bool | None
    fat_suppressed: bool | None
    weighting_source: str
    fat_suppression_source: str
    representative_path: str | None
    ordering_method: str
    total_slices: int
    selected_indices: tuple[int, ...]
    selected_relative_positions: tuple[float, ...]
    selected_paths: tuple[str, ...]


@dataclass(frozen=True)
class _SeriesCandidate:
    study_uid: str
    series_uid: str
    plane: str
    series_dir: Path
    paths: tuple[Path, ...]
    acquisition: AcquisitionClass
    representative_path: Path
    matching_slots: tuple[str, ...]


def build_study_records(
    series_table: pd.DataFrame,
    image_root: Path,
    study_uids: Sequence[str],
    sampled_slices_per_slot: int,
    slice_band_start: float,
    slice_band_end: float,
    slot_names: Sequence[str],
    classification_method: str,
) -> list[StudyRecord]:
    """Classify series, fill fixed acquisition slots, and sample each stack."""
    central_band_indices(
        0, sampled_slices_per_slot, slice_band_start, slice_band_end
    )
    if classification_method != CLASSIFICATION_METHOD:
        raise ValueError(
            f"Unsupported series classification method: {classification_method}"
        )
    slots = resolve_slots(slot_names)
    required = {"StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane"}
    missing = required.difference(series_table.columns)
    if missing:
        raise ValueError(f"Series table is missing columns: {sorted(missing)}")

    wanted = {str(uid) for uid in study_uids}
    series_inputs: dict[tuple[str, str], tuple[str, Path, tuple[Path, ...]]] = {}
    for row in series_table.itertuples(index=False):
        study_uid = str(row.StudyInstanceUID)
        if study_uid not in wanted:
            continue
        plane = str(row.Anatomical_Plane).strip().lower()
        if plane not in {slot.plane for slot in slots}:
            continue
        series_uid = str(row.SeriesInstanceUID)
        series_dir = image_root / study_uid / series_uid
        paths = (
            tuple(sorted(series_dir.glob("*.dcm")))
            if series_dir.is_dir()
            else ()
        )
        if paths:
            series_inputs[(study_uid, series_uid)] = (plane, series_dir, paths)

    def classify_series(
        item: tuple[tuple[str, str], tuple[str, Path, tuple[Path, ...]]]
    ) -> _SeriesCandidate:
        (study_uid, series_uid), (plane, series_dir, paths) = item
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
        matching = tuple(
            slot.name for slot in slots if slot.matches(plane, acquisition)
        )
        return _SeriesCandidate(
            study_uid=study_uid,
            series_uid=series_uid,
            plane=plane,
            series_dir=series_dir,
            paths=paths,
            acquisition=acquisition,
            representative_path=representative,
            matching_slots=matching,
        )

    ordered_inputs = sorted(series_inputs.items())
    with ThreadPoolExecutor(max_workers=ORDERING_WORKERS) as pool:
        classified = list(pool.map(classify_series, ordered_inputs))

    candidates: dict[str, dict[str, list[_SeriesCandidate]]] = {}
    classifications: dict[str, list[SeriesClassification]] = {}
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
                fluid_sensitive=candidate.acquisition.fluid_sensitive,
                fat_suppressed=candidate.acquisition.fat_suppressed,
                weighting_source=candidate.acquisition.weighting_source,
                fat_suppression_source=(
                    candidate.acquisition.fat_suppression_source
                ),
                representative_path=relative_header,
                matching_slots=candidate.matching_slots,
            )
        )
        for slot_name in candidate.matching_slots:
            candidates.setdefault(candidate.study_uid, {}).setdefault(
                slot_name, []
            ).append(candidate)

    chosen: dict[str, dict[str, _SeriesCandidate]] = {}
    for study_uid, slots_for_study in candidates.items():
        for slot_name, slot_candidates in slots_for_study.items():
            selected_uid = choose_largest_series(
                [
                    (candidate.series_uid, len(candidate.paths))
                    for candidate in slot_candidates
                ]
            )
            chosen.setdefault(study_uid, {})[slot_name] = next(
                candidate
                for candidate in slot_candidates
                if candidate.series_uid == selected_uid
            )

    series_directories = sorted(
        {
            candidate.series_dir
            for slots_for_study in chosen.values()
            for candidate in slots_for_study.values()
        }
    )

    def sample_series(series_dir: Path):
        paths = list(series_dir.glob("*.dcm"))
        return paths, sample_ordered_paths(
            paths,
            sampled_slices_per_slot,
            slice_band_start,
            slice_band_end,
        )

    with ThreadPoolExecutor(max_workers=ORDERING_WORKERS) as pool:
        sampled_series = dict(
            zip(series_directories, pool.map(sample_series, series_directories))
        )

    records = []
    for uid in study_uids:
        study_uid = str(uid)
        selected: list[Path | None] = []
        selections = []
        for slot in slots:
            candidate = chosen.get(study_uid, {}).get(slot.name)
            if candidate is None:
                paths = []
                sampled, method, indices, relative_positions = (
                    [],
                    "missing",
                    (),
                    (),
                )
            else:
                paths, sample = sampled_series[candidate.series_dir]
                sampled, method, indices, relative_positions = sample
            selected.extend(sampled or [None] * sampled_slices_per_slot)
            series_uid = candidate.series_uid if candidate is not None else None
            acquisition = candidate.acquisition if candidate is not None else None
            selections.append(
                SeriesSelection(
                    slot=slot.name,
                    plane=slot.plane,
                    series_uid=series_uid,
                    weighting=(acquisition.weighting if acquisition else None),
                    fluid_sensitive=(
                        acquisition.fluid_sensitive if acquisition else None
                    ),
                    fat_suppressed=(
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
                        f"{study_uid}/{series_uid}/"
                        f"{candidate.representative_path.name}"
                        if candidate is not None
                        else None
                    ),
                    ordering_method=method,
                    total_slices=len(paths),
                    selected_indices=indices,
                    selected_relative_positions=relative_positions,
                    selected_paths=tuple(
                        f"{study_uid}/{series_uid}/{path.name}"
                        for path in sampled
                    ),
                )
            )
        records.append(
            StudyRecord(
                study_uid,
                tuple(selected),
                tuple(selections),
                tuple(
                    sorted(
                        classifications.get(study_uid, []),
                        key=lambda item: item.series_uid,
                    )
                ),
            )
        )
    return records


def load_study_records_cache(
    cache_path: Path,
    image_root: Path,
    study_uids: Sequence[str],
    sampled_slices_per_slot: int,
    slice_band_start: float,
    slice_band_end: float,
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
    if report.get("slots") != list(slot_names):
        raise ValueError("Study-record cache slot order changed")
    if int(report.get("sampled_slices_per_slot", -1)) != sampled_slices_per_slot:
        raise ValueError("Study-record cache sampled-slice count changed")
    if report.get("slice_band") != [slice_band_start, slice_band_end]:
        raise ValueError("Study-record cache slice band changed")

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
        image_paths: list[Path | None] = []
        series_rows = entry.get("series", [])
        if [row.get("slot") for row in series_rows] != list(slot_names):
            raise ValueError(f"Cached slot order changed for study {study_uid}")
        for row in series_rows:
            selected_paths = tuple(str(path) for path in row["selected_paths"])
            if row["series_uid"] is None:
                if selected_paths:
                    raise ValueError("Missing cached series contains selected paths")
                image_paths.extend([None] * sampled_slices_per_slot)
            else:
                if len(selected_paths) != sampled_slices_per_slot:
                    raise ValueError("Cached series has an invalid selected-path count")
                image_paths.extend(image_root / path for path in selected_paths)
            selections.append(
                SeriesSelection(
                    slot=str(row["slot"]),
                    plane=str(row["plane"]),
                    series_uid=(
                        str(row["series_uid"])
                        if row["series_uid"] is not None
                        else None
                    ),
                    weighting=row["weighting"],
                    fluid_sensitive=row["fluid_sensitive"],
                    fat_suppressed=row["fat_suppressed"],
                    weighting_source=str(row["weighting_source"]),
                    fat_suppression_source=str(row["fat_suppression_source"]),
                    representative_path=row["representative_path"],
                    ordering_method=str(row["ordering_method"]),
                    total_slices=int(row["total_slices"]),
                    selected_indices=tuple(
                        int(value) for value in row["selected_indices"]
                    ),
                    selected_relative_positions=tuple(
                        float(value) for value in row["selected_relative_positions"]
                    ),
                    selected_paths=selected_paths,
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
                weighting_source=str(row["weighting_source"]),
                fat_suppression_source=str(row["fat_suppression_source"]),
                representative_path=str(row["representative_path"]),
                matching_slots=tuple(str(value) for value in row["matching_slots"]),
            )
            for row in entry.get("series_classifications", [])
        )
        records.append(
            StudyRecord(
                study_uid=study_uid,
                image_paths=tuple(image_paths),
                selections=tuple(selections),
                classifications=classifications,
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
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
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
    ) -> None:
        self.records = list(records)
        self.image_size = image_size
        self.physical_fov_mm = physical_fov_mm
        self.labels = labels
        self.label_weights = label_weights
        self.cache_dir = Path(cache_dir) if cache_dir else None
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        images, present = self._load_images(self.records[index])

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
        return images, present, label, label_weight

    def _load_images(self, record: StudyRecord) -> tuple[torch.Tensor, torch.Tensor]:
        """Load one study, reusing an ephemeral preprocessed cache when enabled."""
        cache_path = self._cache_path(record)
        if cache_path is not None and cache_path.is_file():
            try:
                with np.load(cache_path, allow_pickle=False) as cached:
                    present = torch.from_numpy(cached["present"].astype(np.bool_))
                    present_images = torch.from_numpy(
                        np.asarray(cached["images"], dtype=np.float32)
                    ).div_(255.0)
                image_shape = (
                    1,
                    self.image_size,
                    self.image_size,
                )
                if tuple(present.shape) != (
                    len(record.image_paths),
                ) or tuple(present_images.shape) != (
                    int(present.sum()),
                    *image_shape,
                ):
                    raise ValueError("cached tensor shape does not match the study")
                images = torch.zeros((len(present), *image_shape))
                images[present] = present_images
                return images, present
            except (OSError, ValueError, KeyError) as error:
                print(f"Warning: ignoring invalid DICOM cache {cache_path}: {error}")

        images = []
        present = []
        for path in record.image_paths:
            try:
                image = (
                    read_dicom(path, self.image_size, self.physical_fov_mm)
                    if path is not None
                    else None
                )
            except Exception as error:
                print(f"Warning: failed to decode {path}: {error}", flush=True)
                image = None
            if image is None:
                image = torch.zeros((1, self.image_size, self.image_size))
                present.append(False)
            else:
                present.append(True)
            images.append(image.float())
        image_tensor = torch.stack(images)
        present_tensor = torch.tensor(present)
        if cache_path is not None:
            self._write_cache(cache_path, image_tensor, present_tensor)
        return image_tensor, present_tensor

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
                    for path in record.image_paths
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
                np.savez(
                    file,
                    images=(images[present].mul(255.0).round().byte().numpy()),
                    present=present.numpy().astype(np.bool_),
                )
            os.replace(temporary, cache_path)
        except OSError as error:
            temporary.unlink(missing_ok=True)
            print(f"Warning: could not write DICOM cache {cache_path}: {error}")
