from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from rsna2026.acquisition import CLASSIFICATION_METHOD, SLOT_NAMES
from rsna2026.dataset import KneeStudyDataset, build_cached_study_records
from rsna2026.image_cache import (
    ImageCacheReader,
    build_image_cache_shard,
    shard_for_study,
)
from rsna2026.slice_selection import PhysicalWindowSelection
from rsna2026.train import open_verified_image_cache, slice_selection_report


def _selection(paths: list[Path]) -> PhysicalWindowSelection:
    windows = (
        (paths[0], paths[1], paths[2]),
        (paths[1], paths[2], paths[3]),
    )
    return PhysicalWindowSelection(
        windows=windows,
        center_positions_mm=(1.0, 2.0),
        position_features=((1 / 3, 1.0, 1.0), (2 / 3, 1.0, 1.0)),
        target_positions_mm=(1.0, 2.0),
        all_positions_mm=(0.0, 1.0, 2.0, 3.0),
        method="physical-coordinate-v1",
        coverage_mm=3.0,
        median_spacing_mm=1.0,
        duplicate_positions=0,
        large_gaps=0,
    )


def test_study_hash_partition_is_deterministic_and_complete() -> None:
    uids = [f"study-{index}" for index in range(100)]
    partitions = [
        {uid for uid in uids if shard_for_study(uid, 4) == shard}
        for shard in range(4)
    ]

    assert set.union(*partitions) == set(uids)
    assert sum(len(partition) for partition in partitions) == len(uids)
    assert shard_for_study("study-1", 4) == shard_for_study("study-1", 4)


def test_verified_image_cache_always_returns_run_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []

    class FakeImageCacheReader:
        summaries = [
            {
                "studies": 2,
                "series": 3,
                "images": 5,
                "decoded_images": 4,
                "failed_images": 1,
                "archive_bytes": 100,
            },
            {
                "studies": 1,
                "series": 4,
                "images": 6,
                "decoded_images": 6,
                "failed_images": 0,
                "archive_bytes": 200,
            },
        ]

        def __init__(self, manifest_paths, **kwargs) -> None:
            calls.append({"manifest_paths": manifest_paths, **kwargs})

    monkeypatch.setattr("rsna2026.train.ImageCacheReader", FakeImageCacheReader)
    manifests = [tmp_path / "shard-0.json", tmp_path / "shard-1.json"]
    reader, summary = open_verified_image_cache(
        manifests,
        ["study-1", "study-2", "study-3"],
        ["a" * 64, "b" * 64],
        {
            "expected_studies": 3,
            "expected_series": 7,
            "expected_images": 11,
            "expected_failed_images": 1,
        },
    )

    assert isinstance(reader, FakeImageCacheReader)
    assert calls == [
        {
            "manifest_paths": manifests,
            "expected_study_uids": ["study-1", "study-2", "study-3"],
            "expected_manifest_sha256": ["a" * 64, "b" * 64],
            "verify_files": True,
        }
    ]
    assert summary == {
        "studies": 3,
        "series": 7,
        "images": 11,
        "decoded_images": 10,
        "failed_images": 1,
        "archive_bytes": 300,
    }


def test_builds_all_series_cache_and_decodes_each_source_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study_uid = "study-1"
    image_root = tmp_path / "train_series"
    rows = []
    for series_index in range(2):
        series_uid = f"series-{series_index}"
        series_dir = image_root / study_uid / series_uid
        series_dir.mkdir(parents=True)
        paths = []
        for slice_index in range(4):
            path = series_dir / f"slice-{slice_index}.dcm"
            path.touch()
            paths.append(path)
        rows.append(
            {
                "StudyInstanceUID": study_uid,
                "SeriesInstanceUID": series_uid,
                "Anatomical_Plane": "Sagittal",
                "Fluid_Sensitive": 1,
                "Fat_Suppression": 1,
            }
        )

    decoded: list[Path] = []

    def fake_read(path: Path, image_size: int, physical_fov_mm: float):
        decoded.append(path)
        value = int(path.stem.split("-")[-1]) / 3
        return torch.full((1, image_size, image_size), value)

    monkeypatch.setattr(
        "rsna2026.image_cache.select_physical_windows",
        lambda paths, _count: _selection(list(paths)),
    )
    monkeypatch.setattr("rsna2026.image_cache.read_dicom", fake_read)
    monkeypatch.setattr(
        "rsna2026.image_cache.read_series_header",
        lambda _path: {
            "SeriesDescription": "SAG PD FS",
            "ImageOrientationPatient": [0, 1, 0, 0, 0, -1],
            "ImagePositionPatient": [100, 0, 0],
            "PixelSpacing": [1, 1],
            "Laterality": "R",
            "Rows": 224,
            "Columns": 224,
        },
    )

    manifest = build_image_cache_shard(
        pd.DataFrame(rows),
        image_root,
        [study_uid],
        tmp_path / "output",
        shard_index=0,
        shard_count=1,
    )

    assert manifest["summary"]["series"] == 2
    assert manifest["summary"]["images"] == 8
    assert len(decoded) == 8
    assert len(set(decoded)) == 8
    study = manifest["studies"][0]
    assert len(study["series"]) == 2
    assert study["series"][0]["acquisition_header"] == {
        "SeriesDescription": "SAG PD FS",
        "ImageOrientationPatient": [0, 1, 0, 0, 0, -1],
        "ImagePositionPatient": [100, 0, 0],
        "PixelSpacing": [1, 1],
        "Laterality": "R",
        "Rows": 224,
        "Columns": 224,
    }

    reader = ImageCacheReader(
        [tmp_path / "output/image-cache/cache-manifest.json"],
        expected_study_uids=[study_uid],
    )
    images, present, relative_paths = reader.load_study(study_uid)
    assert images.shape == (8, 224, 224)
    assert present.all()
    assert len(relative_paths) == 8
    assert images[1, 0, 0] == 85
    image, is_present = reader.load_image(relative_paths[1])
    assert is_present
    np.testing.assert_array_equal(image, images[1])
    with pytest.raises(KeyError, match="absent from image cache"):
        reader.load_image(f"{study_uid}/unknown/missing.dcm")

    records = build_cached_study_records(
        pd.DataFrame(rows),
        reader,
        [study_uid],
        window_count=2,
        max_series_per_slot=1,
        slot_names=SLOT_NAMES,
        classification_method=CLASSIFICATION_METHOD,
    )
    assert len(records[0].orientation_audits) == 2
    assert all(
        audit.tag_geometry_disagreement
        for audit in records[0].orientation_audits
    )
    audit_summary = slice_selection_report(
        {"train": records}, window_count=2, max_series_per_slot=1
    )["splits"]["train"]["summary"]
    assert audit_summary["laterality_disagreement_studies"] == [study_uid]
    assert audit_summary["laterality_disagreement_study_count"] == 1
    monkeypatch.setattr(
        "rsna2026.dataset.read_dicom",
        lambda *_args: pytest.fail("persistent cache must not decode DICOM"),
    )
    batch_images, batch_present, _, _, _ = KneeStudyDataset(
        records,
        image_size=224,
        physical_fov_mm=130.0,
        image_cache=reader,
    )[0]
    assert batch_images.shape == (6, 1, 2, 3, 224, 224)
    assert batch_present[0, 0].tolist() == [True, True]
    torch.testing.assert_close(
        batch_images[0, 0, 0, 1], batch_images[0, 0, 1, 0]
    )

    absolute_records = [
        replace(
            records[0],
            window_paths=tuple(
                tuple(
                    tmp_path / "competition/train_series" / path
                    if path is not None
                    else None
                    for path in window
                )
                for window in records[0].window_paths
            ),
        )
    ]
    absolute_images, absolute_present, _, _, _ = KneeStudyDataset(
        absolute_records,
        image_size=224,
        physical_fov_mm=130.0,
        image_cache=reader,
    )[0]
    torch.testing.assert_close(absolute_images, batch_images)
    torch.testing.assert_close(absolute_present, batch_present)


def test_cache_records_decode_failure_and_rejects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study_uid = "study"
    series_uid = "series"
    series_dir = tmp_path / "images" / study_uid / series_uid
    series_dir.mkdir(parents=True)
    paths = []
    for index in range(4):
        path = series_dir / f"{index}.dcm"
        path.touch()
        paths.append(path)
    table = pd.DataFrame(
        [
            {
                "StudyInstanceUID": study_uid,
                "SeriesInstanceUID": series_uid,
                "Anatomical_Plane": "Axial",
                "Fluid_Sensitive": 1,
                "Fat_Suppression": 1,
            }
        ]
    )
    monkeypatch.setattr(
        "rsna2026.image_cache.select_physical_windows",
        lambda _paths, _count: _selection(paths),
    )
    monkeypatch.setattr(
        "rsna2026.image_cache.read_series_header", lambda _path: {}
    )

    def fake_read(path: Path, image_size: int, _physical_fov_mm: float):
        if path.name == "2.dcm":
            raise ValueError("corrupt")
        return torch.full((1, image_size, image_size), 0.5)

    monkeypatch.setattr("rsna2026.image_cache.read_dicom", fake_read)
    output = tmp_path / "output"
    manifest = build_image_cache_shard(
        table,
        tmp_path / "images",
        [study_uid],
        output,
        shard_index=0,
        shard_count=1,
    )
    assert manifest["summary"]["failed_images"] == 1

    archive = output / "image-cache/studies/study.npz"
    archive.write_bytes(archive.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="size changed"):
        ImageCacheReader([output / "image-cache/cache-manifest.json"])


def test_cache_enforces_fixed_resolution(tmp_path: Path) -> None:
    table = pd.DataFrame(
        columns=["StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane"]
    )
    with pytest.raises(ValueError, match="fixed at 224"):
        build_image_cache_shard(
            table,
            tmp_path,
            [],
            tmp_path / "output",
            shard_index=0,
            shard_count=1,
            image_size=336,
        )
