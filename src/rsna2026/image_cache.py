"""Persistent, classification-independent preprocessing cache for training MRI."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

from rsna2026.dataset import PREPROCESSING_VERSION, read_dicom
from rsna2026.orientation import ORIENTATION_NORMALIZATION_METHOD, ORIENTATION_TAGS
from rsna2026.slice_selection import select_physical_windows


IMAGE_CACHE_FORMAT_VERSION = "rsna2026-image-cache-v1"
DEFAULT_CACHE_SHARD_COUNT = 4
MAX_CACHE_SHARD_BYTES = 18 * 1024**3
ACQUISITION_HEADER_TAGS = (
    "SeriesDescription",
    "SequenceName",
    "ProtocolName",
    "ScanOptions",
    "RepetitionTime",
    "EchoTime",
    "EchoTrainLength",
    "InversionTime",
    "FlipAngle",
    "ScanningSequence",
    "SequenceVariant",
    "MRAcquisitionType",
    "SliceThickness",
    "SpacingBetweenSlices",
    "PixelSpacing",
    "MagneticFieldStrength",
    "Manufacturer",
    "ManufacturerModelName",
    *ORIENTATION_TAGS,
)


def shard_for_study(study_uid: str, shard_count: int) -> int:
    """Assign one study to a deterministic cache shard."""
    if shard_count <= 0:
        raise ValueError("Cache shard count must be positive")
    digest = hashlib.sha256(str(study_uid).encode("utf-8")).digest()
    return int.from_bytes(digest, "big") % shard_count


def collection_sha256(values: Sequence[str]) -> str:
    """Hash a sorted identifier collection with an unambiguous delimiter."""
    material = "".join(f"{value}\n" for value in sorted(map(str, values)))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 for a potentially large cache file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: object) -> object:
    """Convert pydicom values to stable JSON-compatible primitives."""
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_value(item) for item in value]
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)
    return number if np.isfinite(number) else None


def read_series_header(path: Path) -> dict[str, object]:
    """Read raw acquisition fields retained for later slot reclassification."""
    dataset = pydicom.dcmread(
        path,
        force=True,
        stop_before_pixels=True,
        specific_tags=list(dict.fromkeys(ACQUISITION_HEADER_TAGS)),
    )
    return {
        name: _json_value(getattr(dataset, name, None))
        for name in dict.fromkeys(ACQUISITION_HEADER_TAGS)
    }


def _write_study_archive(
    archive_path: Path,
    relative_paths: Sequence[str],
    source_paths: Sequence[Path],
    image_size: int,
    physical_fov_mm: float,
) -> tuple[int, int]:
    """Decode every unique source once and atomically write one study archive."""
    images = np.zeros(
        (len(source_paths), image_size, image_size), dtype=np.uint8
    )
    present = np.zeros(len(source_paths), dtype=np.bool_)
    for index, path in enumerate(source_paths):
        try:
            image = read_dicom(path, image_size, physical_fov_mm)
        except Exception as error:
            print(f"Warning: failed to decode {path}: {error}", flush=True)
            continue
        values = image.squeeze(0).clamp(0.0, 1.0).mul(255.0).round().byte()
        images[index] = values.numpy()
        present[index] = True

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive_path.with_name(
        f".{archive_path.name}.{os.getpid()}.temporary"
    )
    try:
        with temporary.open("wb") as file:
            np.savez_compressed(
                file,
                images=images,
                present=present,
                relative_paths=np.asarray(relative_paths, dtype=np.str_),
            )
        os.replace(temporary, archive_path)
    finally:
        temporary.unlink(missing_ok=True)
    return int(present.sum()), int((~present).sum())


def build_image_cache_shard(
    series_table: pd.DataFrame,
    image_root: Path,
    study_uids: Sequence[str],
    output_dir: Path,
    shard_index: int,
    shard_count: int = DEFAULT_CACHE_SHARD_COUNT,
    image_size: int = 224,
    physical_fov_mm: float = 130.0,
    window_count: int = 10,
    maximum_bytes: int = MAX_CACHE_SHARD_BYTES,
) -> dict[str, object]:
    """Build one complete, independently verifiable image-cache shard."""
    if not 0 <= shard_index < shard_count:
        raise ValueError("Cache shard index is outside the configured range")
    if image_size != 224:
        raise ValueError("Persistent image cache is fixed at 224 pixels")
    required = {"StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane"}
    missing = required.difference(series_table.columns)
    if missing:
        raise ValueError(f"Series table is missing columns: {sorted(missing)}")

    all_uids = tuple(str(uid) for uid in study_uids)
    if len(set(all_uids)) != len(all_uids):
        raise ValueError("Training study identifiers must be unique")
    selected_uids = tuple(
        uid for uid in all_uids if shard_for_study(uid, shard_count) == shard_index
    )
    selected_set = set(selected_uids)
    rows_by_study: dict[str, list[tuple[str, str]]] = {
        uid: [] for uid in selected_uids
    }
    for row in series_table.itertuples(index=False):
        study_uid = str(row.StudyInstanceUID)
        if study_uid in selected_set:
            rows_by_study[study_uid].append(
                (str(row.SeriesInstanceUID), str(row.Anatomical_Plane).lower())
            )

    cache_dir = output_dir / "image-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    studies = []
    total_bytes = 0
    total_series = 0
    total_images = 0
    total_present = 0
    total_failed = 0
    for study_number, study_uid in enumerate(selected_uids, start=1):
        series_entries = []
        unique_sources: dict[str, Path] = {}
        for series_uid, plane in sorted(rows_by_study[study_uid]):
            series_dir = image_root / study_uid / series_uid
            paths = tuple(sorted(series_dir.glob("*.dcm")))
            selection = select_physical_windows(paths, window_count)
            for window in selection.windows:
                for path in window:
                    relative = f"{study_uid}/{series_uid}/{path.name}"
                    unique_sources.setdefault(relative, path)
            representative = paths[len(paths) // 2] if paths else None
            try:
                header = read_series_header(representative) if representative else {}
                header_error = None
            except Exception as error:
                header = {}
                header_error = type(error).__name__
            series_entries.append(
                {
                    "series_uid": series_uid,
                    "plane": plane,
                    "total_slices": len(paths),
                    "representative_path": (
                        f"{study_uid}/{series_uid}/{representative.name}"
                        if representative is not None
                        else None
                    ),
                    "acquisition_header": header,
                    "acquisition_header_error": header_error,
                    "ordering_method": selection.method,
                    "physical_coverage_mm": selection.coverage_mm,
                    "median_spacing_mm": selection.median_spacing_mm,
                    "duplicate_positions": selection.duplicate_positions,
                    "large_gaps": selection.large_gaps,
                    "all_positions_mm": selection.all_positions_mm,
                    "target_positions_mm": selection.target_positions_mm,
                    "selected_center_positions_mm": selection.center_positions_mm,
                    "selected_windows": [
                        [f"{study_uid}/{series_uid}/{path.name}" for path in window]
                        for window in selection.windows
                    ],
                    "position_features": selection.position_features,
                }
            )

        relative_paths = tuple(unique_sources)
        archive_relative = f"studies/{study_uid}.npz"
        archive_path = cache_dir / archive_relative
        present_count, failed_count = _write_study_archive(
            archive_path,
            relative_paths,
            tuple(unique_sources.values()),
            image_size,
            physical_fov_mm,
        )
        archive_bytes = archive_path.stat().st_size
        total_bytes += archive_bytes
        if total_bytes > maximum_bytes:
            raise RuntimeError(
                f"Image-cache shard exceeded its byte limit: {total_bytes} > "
                f"{maximum_bytes}; rebuild with eight shards"
            )
        archive_hash = file_sha256(archive_path)
        studies.append(
            {
                "study_uid": study_uid,
                "archive": archive_relative,
                "archive_bytes": archive_bytes,
                "archive_sha256": archive_hash,
                "relative_paths": relative_paths,
                "present_count": present_count,
                "failed_count": failed_count,
                "series": series_entries,
            }
        )
        total_series += len(series_entries)
        total_images += len(relative_paths)
        total_present += present_count
        total_failed += failed_count
        if study_number % 100 == 0 or study_number == len(selected_uids):
            print(
                f"Prepared shard {shard_index}: studies={study_number}/"
                f"{len(selected_uids)}, images={total_images}, "
                f"bytes={total_bytes}",
                flush=True,
            )

    manifest: dict[str, object] = {
        "format_version": IMAGE_CACHE_FORMAT_VERSION,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "all_study_uids_sha256": collection_sha256(all_uids),
        "shard_study_uids_sha256": collection_sha256(selected_uids),
        "preprocessing": {
            "version": PREPROCESSING_VERSION,
            "image_size": image_size,
            "physical_fov_mm": physical_fov_mm,
            "normalization_percentiles": [1.0, 99.0],
            "monochrome1_inverted": True,
            "orientation_normalization": ORIENTATION_NORMALIZATION_METHOD,
            "medial_side": "image-right",
            "resize_interpolation": "bilinear-align-corners-false",
            "storage_dtype": "uint8",
            "compression": "numpy-npz-deflate-lossless",
            "window_count": window_count,
            "window_size": 3,
        },
        "summary": {
            "studies": len(studies),
            "series": total_series,
            "images": total_images,
            "decoded_images": total_present,
            "failed_images": total_failed,
            "archive_bytes": total_bytes,
            "maximum_archive_bytes": maximum_bytes,
        },
        "studies": studies,
    }
    manifest_path = cache_dir / "cache-manifest.json"
    temporary = manifest_path.with_suffix(".json.temporary")
    temporary.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)
    print(
        f"Completed image-cache shard {shard_index}: "
        f"manifest_sha256={file_sha256(manifest_path)}, "
        f"summary={manifest['summary']}",
        flush=True,
    )
    return manifest


class ImageCacheReader:
    """Strict reader for one or more persistent preprocessing-cache shards."""

    def __init__(
        self,
        manifest_paths: Sequence[Path],
        *,
        expected_study_uids: Sequence[str] | None = None,
        expected_manifest_sha256: Sequence[str] | None = None,
        verify_files: bool = True,
    ) -> None:
        if not manifest_paths:
            raise ValueError("At least one image-cache manifest is required")
        if expected_manifest_sha256 is not None and len(
            expected_manifest_sha256
        ) != len(manifest_paths):
            raise ValueError("Image-cache manifest SHA-256 count changed")
        self._studies: dict[str, tuple[Path, Mapping[str, object]]] = {}
        self.summaries: list[Mapping[str, object]] = []
        shard_indices: set[int] = set()
        shard_count: int | None = None
        all_uid_hash: str | None = None
        for manifest_number, manifest_path in enumerate(manifest_paths):
            if expected_manifest_sha256 is not None and file_sha256(
                manifest_path
            ) != str(expected_manifest_sha256[manifest_number]):
                raise ValueError(
                    f"Image-cache manifest SHA-256 changed: {manifest_path}"
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("format_version") != IMAGE_CACHE_FORMAT_VERSION:
                raise ValueError("Image-cache format version changed")
            preprocessing = manifest.get("preprocessing", {})
            if preprocessing.get("version") != PREPROCESSING_VERSION:
                raise ValueError("Image-cache preprocessing version changed")
            if int(preprocessing.get("image_size", -1)) != 224:
                raise ValueError("Image-cache size changed")
            if float(preprocessing.get("physical_fov_mm", -1.0)) != 130.0:
                raise ValueError("Image-cache physical field of view changed")
            if preprocessing.get("normalization_percentiles") != [1.0, 99.0]:
                raise ValueError("Image-cache normalization changed")
            if preprocessing.get("monochrome1_inverted") is not True:
                raise ValueError("Image-cache monochrome inversion changed")
            if (
                preprocessing.get("orientation_normalization")
                != ORIENTATION_NORMALIZATION_METHOD
                or preprocessing.get("medial_side") != "image-right"
            ):
                raise ValueError("Image-cache orientation normalization changed")
            if (
                preprocessing.get("resize_interpolation")
                != "bilinear-align-corners-false"
            ):
                raise ValueError("Image-cache interpolation changed")
            if preprocessing.get("storage_dtype") != "uint8":
                raise ValueError("Image-cache storage type changed")
            current_count = int(manifest["shard_count"])
            current_index = int(manifest["shard_index"])
            if shard_count is None:
                shard_count = current_count
                all_uid_hash = str(manifest["all_study_uids_sha256"])
            if current_count != shard_count or str(
                manifest["all_study_uids_sha256"]
            ) != all_uid_hash:
                raise ValueError("Image-cache shard configuration changed")
            if current_index in shard_indices:
                raise ValueError("Duplicate image-cache shard index")
            shard_indices.add(current_index)
            self.summaries.append(manifest["summary"])
            for study in manifest["studies"]:
                study_uid = str(study["study_uid"])
                if study_uid in self._studies:
                    raise ValueError(f"Duplicate cached study: {study_uid}")
                archive_path = manifest_path.parent / str(study["archive"])
                if not archive_path.is_file():
                    raise ValueError(f"Missing image-cache archive: {archive_path}")
                if archive_path.stat().st_size != int(study["archive_bytes"]):
                    raise ValueError(f"Image-cache archive size changed: {archive_path}")
                if verify_files and file_sha256(archive_path) != str(
                    study["archive_sha256"]
                ):
                    raise ValueError(f"Image-cache archive SHA-256 changed: {archive_path}")
                self._studies[study_uid] = (archive_path, study)
        if shard_count is None or shard_indices != set(range(shard_count)):
            raise ValueError("Image-cache shard set is incomplete")
        if expected_study_uids is not None:
            expected = set(map(str, expected_study_uids))
            if set(self._studies) != expected:
                raise ValueError("Image-cache study UID set changed")
            if collection_sha256(tuple(expected)) != all_uid_hash:
                raise ValueError("Image-cache study UID SHA-256 changed")

    @property
    def study_uids(self) -> tuple[str, ...]:
        """Return every cached study identifier in stable order."""
        return tuple(sorted(self._studies))

    def study_metadata(self, study_uid: str) -> Mapping[str, object]:
        """Return immutable manifest metadata for one cached study."""
        try:
            return self._studies[str(study_uid)][1]
        except KeyError as error:
            raise KeyError(
                f"Study is absent from image cache: {study_uid}"
            ) from error

    def load_study(
        self, study_uid: str
    ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        """Load one study with strict array and path-layout validation."""
        try:
            archive_path, study = self._studies[str(study_uid)]
        except KeyError as error:
            raise KeyError(f"Study is absent from image cache: {study_uid}") from error
        with np.load(archive_path, allow_pickle=False) as archive:
            images = np.asarray(archive["images"], dtype=np.uint8)
            present = np.asarray(archive["present"], dtype=np.bool_)
            relative_paths = tuple(str(value) for value in archive["relative_paths"])
        expected_paths = tuple(str(value) for value in study["relative_paths"])
        if images.shape != (len(expected_paths), 224, 224):
            raise ValueError("Cached image array shape changed")
        if present.shape != (len(expected_paths),):
            raise ValueError("Cached presence array shape changed")
        if relative_paths != expected_paths:
            raise ValueError("Cached relative-path layout changed")
        return images, present, relative_paths

    def load_image(self, relative_path: str) -> tuple[np.ndarray, bool]:
        """Resolve one relative DICOM path without any DICOM fallback."""
        study_uid = str(relative_path).split("/", 1)[0]
        images, present, paths = self.load_study(study_uid)
        try:
            index = paths.index(str(relative_path))
        except ValueError as error:
            raise KeyError(
                f"DICOM path is absent from image cache: {relative_path}"
            ) from error
        return images[index], bool(present[index])
