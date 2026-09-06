from __future__ import annotations

import hashlib
import json
import lzma

import pytest


pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("pydicom")
pytest.importorskip("torch")

from rsna2026.acquisition import CLASSIFICATION_METHOD, SLOT_NAMES  # noqa: E402
from rsna2026.dataset import (  # noqa: E402
    SLICE_SELECTION_METHOD,
    load_study_records_cache,
)
from rsna2026.train import (  # noqa: E402
    apply_study_records_cache_environment,
    parse_args,
)


def test_prepare_selection_cache_mode_is_accepted() -> None:
    args = parse_args(
        [
            "--config",
            "baseline.toml",
            "--mode",
            "prepare-selection-cache",
        ]
    )
    assert args.mode == "prepare-selection-cache"


def test_runtime_selection_cache_overrides_empty_config(monkeypatch) -> None:
    data_config: dict[str, object] = {
        "study_records_cache_path": "",
        "study_records_cache_sha256": "",
    }
    config: dict[str, object] = {"data": data_config}
    monkeypatch.setenv(
        "RSNA2026_STUDY_RECORDS_CACHE_PATH", "/tmp/selection.json"
    )
    monkeypatch.setenv("RSNA2026_STUDY_RECORDS_CACHE_SHA256", "a" * 64)

    apply_study_records_cache_environment(config)

    assert data_config["study_records_cache_path"] == "/tmp/selection.json"
    assert data_config["study_records_cache_sha256"] == "a" * 64


def test_runtime_selection_cache_requires_path_and_hash(monkeypatch) -> None:
    config: dict[str, object] = {"data": {}}
    monkeypatch.setenv(
        "RSNA2026_STUDY_RECORDS_CACHE_PATH", "/tmp/selection.json"
    )

    with pytest.raises(ValueError, match="must be set together"):
        apply_study_records_cache_environment(config)


def _selection(slot: str, present: bool) -> dict[str, object]:
    series_uid = "series-1" if present else None
    selected_windows = (
        [[
            "study-1/series-1/first.dcm",
            "study-1/series-1/second.dcm",
            "study-1/series-1/third.dcm",
        ]]
        if present
        else []
    )
    return {
        "slot": slot,
        "plane": slot.split("-", 1)[0],
        "series_uid": series_uid,
        "weighting": "pd" if present else None,
        "fluid_sensitive": True if present else None,
        "fat_suppressed": True if present else None,
        "weighting_source": "test" if present else "missing",
        "fat_suppression_source": "test" if present else "missing",
        "representative_path": selected_windows[0][1] if present else None,
        "ordering_method": "physical-coordinate-v1" if present else "missing",
        "total_slices": 3 if present else 0,
        "physical_coverage_mm": 6.0 if present else 0.0,
        "median_spacing_mm": 3.0 if present else 0.0,
        "duplicate_positions": 0,
        "large_gaps": 0,
        "all_positions_mm": [0.0, 3.0, 6.0] if present else [],
        "target_positions_mm": [3.0] if present else [],
        "selected_center_positions_mm": [3.0] if present else [],
        "selected_windows": selected_windows,
        "position_features": [[0.5, 1.0, 1.0]] if present else [],
    }


def test_loads_pinned_study_records_without_reading_dicom(tmp_path) -> None:
    report = {
        "series_classification": CLASSIFICATION_METHOD,
        "slice_selection_method": SLICE_SELECTION_METHOD,
        "slots": list(SLOT_NAMES),
        "window_count": 2,
        "window_size": 3,
        "splits": {
            "all-training-studies": {
                "summary": {"studies": 1},
                "studies": [
                    {
                        "study_uid": "study-1",
                        "series": [
                            _selection(slot, index == 0)
                            for index, slot in enumerate(SLOT_NAMES)
                        ],
                        "series_classifications": [],
                    }
                ],
            }
        },
    }
    cache_path = tmp_path / "slice-selection.json.xz"
    raw = (json.dumps(report) + "\n").encode("utf-8")
    cache_path.write_bytes(lzma.compress(raw))
    uid_sha256 = hashlib.sha256(b"study-1\n").hexdigest()

    records, summary = load_study_records_cache(
        cache_path=cache_path,
        image_root=tmp_path / "images",
        study_uids=["study-1"],
        window_count=2,
        slot_names=SLOT_NAMES,
        classification_method=CLASSIFICATION_METHOD,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
        expected_study_uids_sha256=uid_sha256,
    )

    assert summary == {"studies": 1}
    assert len(records) == 1
    assert records[0].study_uid == "study-1"
    assert records[0].window_paths[:2] == (
        (
            tmp_path / "images" / "study-1/series-1/first.dcm",
            tmp_path / "images" / "study-1/series-1/second.dcm",
            tmp_path / "images" / "study-1/series-1/third.dcm",
        ),
        (None, None, None),
    )
    assert records[0].window_paths[2:] == ((None, None, None),) * 10
