from __future__ import annotations

import pytest

from rsna2026.acquisition import (
    SLOT_NAMES,
    choose_largest_series,
    classify_acquisition,
    resolve_slots,
)


@pytest.mark.parametrize(
    ("header", "weighting", "fluid", "fat_suppressed", "source"),
    [
        (
            {"SeriesDescription": "SAG PD FS"},
            "pd",
            True,
            True,
            "series-text:pd",
        ),
        (
            {"SeriesDescription": "COR T1", "RepetitionTime": 2500},
            "t1",
            False,
            False,
            "series-text:t1",
        ),
        (
            {"RepetitionTime": 3000, "EchoTime": 80, "ScanOptions": ["FS"]},
            "t2",
            True,
            True,
            "timing:te>60",
        ),
        (
            {"RepetitionTime": 500, "ScanOptions": "SAT_GEMS"},
            "t1",
            False,
            False,
            "timing:tr<800",
        ),
    ],
)
def test_classifies_acquisition_from_header(
    header, weighting, fluid, fat_suppressed, source
) -> None:
    result = classify_acquisition(header)

    assert result.weighting == weighting
    assert result.fluid_sensitive is fluid
    assert result.fat_suppressed is fat_suppressed
    assert result.weighting_source == source


def test_maps_acquisitions_to_fixed_slots_without_fallback() -> None:
    slots = resolve_slots(SLOT_NAMES)
    t1 = classify_acquisition({"SeriesDescription": "T1 COR"})
    unknown = classify_acquisition({})

    assert [slot.name for slot in slots if slot.matches("coronal", t1)] == [
        "coronal-t1"
    ]
    assert not any(slot.matches("coronal", unknown) for slot in slots)


def test_selects_largest_series_with_uid_tie_break() -> None:
    assert (
        choose_largest_series([("series-b", 20), ("series-a", 20)])
        == "series-a"
    )
    assert (
        choose_largest_series([("series-a", 20), ("series-b", 21)])
        == "series-b"
    )


def test_rejects_changed_slot_order() -> None:
    with pytest.raises(ValueError, match="fixed six-slot order"):
        resolve_slots(tuple(reversed(SLOT_NAMES)))
