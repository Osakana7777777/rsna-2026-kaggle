"""Study-level DICOM loading for the RSNA knee baseline."""

from __future__ import annotations

import hashlib
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
PREPROCESSING_VERSION = "dicom-window-uint8-v2"


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


def read_dicom(path: Path, image_size: int) -> torch.Tensor:
    """Decode, robustly window, and resize one DICOM image."""
    dataset = pydicom.dcmread(path, force=True)
    pixels = np.asarray(dataset.pixel_array, dtype=np.float32)
    if pixels.ndim > 2:
        pixels = pixels[pixels.shape[0] // 2]
    slope = float(getattr(dataset, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0) or 0.0)
    pixels = pixels * slope + intercept

    finite = np.isfinite(pixels)
    if not finite.any():
        raise ValueError(f"No finite pixels in {path}")
    fill = float(np.median(pixels[finite]))
    pixels = np.nan_to_num(pixels, nan=fill, posinf=fill, neginf=fill)
    low, high = np.percentile(pixels, (1.0, 99.0))
    pixels = np.clip((pixels - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    if str(getattr(dataset, "PhotometricInterpretation", "")) == "MONOCHROME1":
        pixels = 1.0 - pixels
    pixels = pixels.astype(np.float32, copy=False)

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
        labels: np.ndarray | None = None,
        label_weights: np.ndarray | None = None,
        augment: bool = False,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.records = list(records)
        self.image_size = image_size
        self.labels = labels
        self.label_weights = label_weights
        self.augment = augment
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
        if self.augment:
            scales = 0.9 + 0.2 * torch.rand((len(images), 1, 1, 1))
            images = (images * scales).clamp(0.0, 1.0)

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
                image = read_dicom(path, self.image_size) if path is not None else None
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
