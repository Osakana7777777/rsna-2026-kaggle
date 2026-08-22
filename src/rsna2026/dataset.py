"""Study-level DICOM loading for the RSNA knee baseline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as functional
from torch.utils.data import Dataset


PLANES = ("sagittal", "coronal", "axial")


@dataclass(frozen=True)
class StudyRecord:
    """Paths sampled from one MRI study."""

    study_uid: str
    image_paths: tuple[Path | None, ...]


def _stable_sample(paths: Sequence[Path], count: int, key: str) -> list[Path]:
    """Choose reproducible files without reading every DICOM header."""
    if not paths:
        return []
    ordered = sorted(paths)
    if len(ordered) <= count:
        chosen = list(ordered)
    else:
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        offset = int.from_bytes(digest[:4], "little") % len(ordered)
        indices = np.linspace(0, len(ordered) - 1, count, dtype=int)
        chosen = [ordered[(int(index) + offset) % len(ordered)] for index in indices]
    while len(chosen) < count:
        chosen.append(chosen[-1])
    return chosen


def build_study_records(
    series_table: pd.DataFrame,
    image_root: Path,
    study_uids: Sequence[str],
    slices_per_plane: int,
) -> list[StudyRecord]:
    """Select the largest available series in each anatomical plane."""
    required = {"StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane"}
    missing = required.difference(series_table.columns)
    if missing:
        raise ValueError(f"Series table is missing columns: {sorted(missing)}")

    wanted = {str(uid) for uid in study_uids}
    candidates: dict[str, dict[str, tuple[int, Path]]] = {}
    for row in series_table.itertuples(index=False):
        study_uid = str(row.StudyInstanceUID)
        if study_uid not in wanted:
            continue
        plane = str(row.Anatomical_Plane).strip().lower()
        if plane not in PLANES:
            continue
        series_uid = str(row.SeriesInstanceUID)
        series_dir = image_root / study_uid / series_uid
        paths = list(series_dir.glob("*.dcm")) if series_dir.is_dir() else []
        previous = candidates.setdefault(study_uid, {}).get(plane)
        if paths and (previous is None or len(paths) > previous[0]):
            candidates[study_uid][plane] = (len(paths), series_dir)

    records = []
    for uid in study_uids:
        study_uid = str(uid)
        selected: list[Path | None] = []
        for plane in PLANES:
            entry = candidates.get(study_uid, {}).get(plane)
            paths = list(entry[1].glob("*.dcm")) if entry is not None else []
            sampled = _stable_sample(paths, slices_per_plane, f"{study_uid}:{plane}")
            selected.extend(sampled or [None] * slices_per_plane)
        records.append(StudyRecord(study_uid, tuple(selected)))
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


class KneeStudyDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Load a fixed-size bag of MRI slices for each study."""

    def __init__(
        self,
        records: Sequence[StudyRecord],
        image_size: int,
        labels: np.ndarray | None = None,
        augment: bool = False,
    ) -> None:
        self.records = list(records)
        self.image_size = image_size
        self.labels = labels
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images = []
        present = []
        for path in self.records[index].image_paths:
            try:
                image = read_dicom(path, self.image_size) if path is not None else None
            except Exception as error:
                print(f"Warning: failed to decode {path}: {error}", flush=True)
                image = None
            if image is None:
                image = torch.zeros((1, self.image_size, self.image_size))
                present.append(False)
            else:
                if self.augment:
                    scale = 0.9 + 0.2 * torch.rand(())
                    image = (image * scale).clamp(0.0, 1.0)
                present.append(True)
            images.append(image.float())

        label = (
            torch.from_numpy(self.labels[index]).float()
            if self.labels is not None
            else torch.empty(0)
        )
        return torch.stack(images), torch.tensor(present), label
