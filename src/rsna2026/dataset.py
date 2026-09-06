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
from rsna2026.slice_selection import select_physical_windows


ORDERING_WORKERS = 16
PREPROCESSING_VERSION = "dicom-physical-fov-window-uint8-v4"
SLICE_SELECTION_METHOD = "physical-adjacent-three-window-v1"


@dataclass(frozen=True)
class StudyRecord:
    """Paths sampled from one MRI study."""

    study_uid: str
    window_paths: tuple[tuple[Path | None, Path | None, Path | None], ...]
    position_features: tuple[tuple[float, float, float], ...]
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
    representative_path: Path
    matching_slots: tuple[str, ...]


def build_study_records(
    series_table: pd.DataFrame,
    image_root: Path,
    study_uids: Sequence[str],
    window_count: int,
    slot_names: Sequence[str],
    classification_method: str,
) -> list[StudyRecord]:
    """Classify series, fill fixed acquisition slots, and sample each stack."""
    if window_count <= 0:
        raise ValueError("Window count must be positive")
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
            candidate = chosen.get(study_uid, {}).get(slot.name)
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
                study_uid,
                tuple(selected),
                tuple(position_features),
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
    window_count: int,
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
    if report.get("slice_selection_method") != SLICE_SELECTION_METHOD:
        raise ValueError("Study-record cache selection method changed")
    if int(report.get("window_count", -1)) != window_count:
        raise ValueError("Study-record cache window count changed")
    if int(report.get("window_size", -1)) != 3:
        raise ValueError("Study-record cache window size changed")

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
        if [row.get("slot") for row in series_rows] != list(slot_names):
            raise ValueError(f"Cached slot order changed for study {study_uid}")
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
                window_paths=tuple(window_paths),
                position_features=tuple(position_features),
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
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        record = self.records[index]
        images, present = self._load_images(record)
        num_slots = len(record.selections)
        if num_slots <= 0 or len(record.window_paths) % num_slots:
            raise ValueError("Study record has an invalid window layout")
        window_count = len(record.window_paths) // num_slots
        images = images.reshape(
            num_slots, window_count, 3, self.image_size, self.image_size
        )
        present = present.reshape(num_slots, window_count)
        positions = torch.tensor(record.position_features, dtype=torch.float32).reshape(
            num_slots, window_count, 3
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
