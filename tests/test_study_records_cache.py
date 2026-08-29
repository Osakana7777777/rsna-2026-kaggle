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
from rsna2026.dataset import load_study_records_cache  # noqa: E402


def _selection(slot: str, present: bool) -> dict[str, object]:
    series_uid = "series-1" if present else None
    selected_paths = (
        ["study-1/series-1/first.dcm", "study-1/series-1/second.dcm"]
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
        "representative_path": selected_paths[0] if present else None,
        "ordering_method": "geometry" if present else "missing",
        "total_slices": 2 if present else 0,
        "selected_indices": [0, 1] if present else [],
        "selected_relative_positions": [0.0, 1.0] if present else [],
        "selected_paths": selected_paths,
    }


def test_loads_pinned_study_records_without_reading_dicom(tmp_path) -> None:
    report = {
        "series_classification": CLASSIFICATION_METHOD,
        "slots": list(SLOT_NAMES),
        "sampled_slices_per_slot": 2,
        "slice_band": [0.2, 0.8],
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
        sampled_slices_per_slot=2,
        slice_band_start=0.2,
        slice_band_end=0.8,
        slot_names=SLOT_NAMES,
        classification_method=CLASSIFICATION_METHOD,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
        expected_study_uids_sha256=uid_sha256,
    )

    assert summary == {"studies": 1}
    assert len(records) == 1
    assert records[0].study_uid == "study-1"
    assert records[0].image_paths[:2] == (
        tmp_path / "images" / "study-1/series-1/first.dcm",
        tmp_path / "images" / "study-1/series-1/second.dcm",
    )
    assert records[0].image_paths[2:] == (None,) * 10
