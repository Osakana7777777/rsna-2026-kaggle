from pathlib import Path

import pytest

from rsna2026.slice_selection import (
    SliceHeader,
    central_band_indices,
    order_slice_paths,
    sample_ordered_paths,
    select_physical_windows,
)


def header(
    position=None,
    orientation=None,
    instance_number=None,
) -> SliceHeader:
    return SliceHeader(position, orientation, instance_number)


def test_orders_oblique_slices_by_projected_patient_position() -> None:
    paths = [Path("slice-c.dcm"), Path("slice-a.dcm"), Path("slice-b.dcm")]
    orientation = (1.0, 0.0, 0.0, 0.0, 2**-0.5, 2**-0.5)
    normal = (0.0, -(2**-0.5), 2**-0.5)
    positions = {
        "slice-a.dcm": tuple(value * 2.0 for value in normal),
        "slice-b.dcm": tuple(value * 1.0 for value in normal),
        "slice-c.dcm": tuple(value * 3.0 for value in normal),
    }

    ordered = order_slice_paths(
        paths,
        lambda path: header(positions[path.name], orientation, 99),
    )

    assert ordered.method == "geometry"
    assert [path.name for path in ordered.paths] == [
        "slice-b.dcm",
        "slice-a.dcm",
        "slice-c.dcm",
    ]


def test_falls_back_to_instance_number_for_the_whole_series() -> None:
    paths = [Path("a.dcm"), Path("b.dcm"), Path("c.dcm")]
    headers = {
        "a.dcm": header((0.0, 0.0, 0.0), None, 30),
        "b.dcm": header((0.0, 0.0, 1.0), (1, 0, 0, 0, 1, 0), 10),
        "c.dcm": header((0.0, 0.0, 2.0), (1, 0, 0, 0, 1, 0), 20),
    }

    ordered = order_slice_paths(paths, lambda path: headers[path.name])

    assert ordered.method == "instance_number"
    assert [path.name for path in ordered.paths] == ["b.dcm", "c.dcm", "a.dcm"]


def test_falls_back_to_file_name_if_instance_number_is_incomplete() -> None:
    paths = [Path("slice-10.dcm"), Path("slice-02.dcm"), Path("slice-01.dcm")]
    headers = {
        "slice-01.dcm": header(instance_number=1),
        "slice-02.dcm": header(instance_number=None),
        "slice-10.dcm": header(instance_number=10),
    }

    ordered = order_slice_paths(paths, lambda path: headers[path.name])

    assert ordered.method == "file_name"
    assert [path.name for path in ordered.paths] == [
        "slice-01.dcm",
        "slice-02.dcm",
        "slice-10.dcm",
    ]


def test_header_read_failure_uses_the_same_fallback_rules() -> None:
    def fail(_path: Path) -> SliceHeader:
        raise ValueError("unreadable header")

    ordered = order_slice_paths([Path("b.dcm"), Path("a.dcm")], fail)

    assert ordered.method == "file_name"
    assert [path.name for path in ordered.paths] == ["a.dcm", "b.dcm"]


def test_inconsistent_slice_normals_fall_back_for_the_whole_series() -> None:
    paths = [Path("a.dcm"), Path("b.dcm")]
    headers = {
        "a.dcm": header((0, 0, 0), (1, 0, 0, 0, 1, 0), 2),
        "b.dcm": header((0, 0, 1), (1, 0, 0, 0, -1, 0), 1),
    }

    ordered = order_slice_paths(paths, lambda path: headers[path.name])

    assert ordered.method == "instance_number"
    assert [path.name for path in ordered.paths] == ["b.dcm", "a.dcm"]


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        (11, (2, 5, 8)),
        (10, (2, 5, 7)),
        (5, (1, 2, 3)),
        (4, (0, 1, 2)),
        (3, (0, 1, 2)),
        (2, (0, 1, 1)),
        (1, (0, 0, 0)),
        (0, ()),
    ],
)
def test_selects_three_slices_from_the_central_band(length, expected) -> None:
    assert central_band_indices(length, 3, 0.20, 0.80) == expected


@pytest.mark.parametrize(
    ("start", "end"),
    [(-0.1, 0.8), (0.2, 1.1), (0.9, 0.1), (0.5, 0.5)],
)
def test_rejects_invalid_slice_bands(start, end) -> None:
    with pytest.raises(ValueError, match="Slice band"):
        central_band_indices(10, 3, start, end)


def test_rejects_nonpositive_slice_count() -> None:
    with pytest.raises(ValueError, match="Slice count"):
        central_band_indices(10, 0, 0.20, 0.80)


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        (20, (4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)),
        (12, tuple(range(12))),
        (3, (0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)),
        (1, (0,) * 12),
        (0, ()),
    ],
)
def test_selects_twelve_slices_from_the_central_band(length, expected) -> None:
    assert central_band_indices(length, 12, 0.20, 0.80) == expected


def test_sampling_reports_method_indices_and_relative_positions() -> None:
    paths = [Path(f"{index:02d}.dcm") for index in reversed(range(10))]

    selected, method, indices, relative = sample_ordered_paths(
        paths,
        3,
        0.20,
        0.80,
        lambda path: header(instance_number=int(path.stem)),
    )

    assert method == "instance_number"
    assert indices == (2, 5, 7)
    assert [path.name for path in selected] == ["02.dcm", "05.dcm", "07.dcm"]
    assert relative == pytest.approx((2 / 9, 5 / 9, 7 / 9))


def test_physical_windows_are_adjacent_unique_and_span_irregular_positions() -> None:
    paths = [Path(f"slice-{index}.dcm") for index in range(14)]
    coordinates = [
        0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 9.0,
        10.0, 14.0, 15.0, 18.0, 22.0, 23.0, 27.0,
    ]
    headers = {
        path.name: header(
            (0.0, 0.0, coordinate), (1, 0, 0, 0, 1, 0), 0
        )
        for path, coordinate in zip(paths, coordinates)
    }

    selected = select_physical_windows(
        paths[::-1], 10, lambda path: headers[path.name]
    )

    assert selected.method == "physical-coordinate-v1"
    assert len(selected.windows) == 10
    assert len(set(selected.windows)) == 10
    for window in selected.windows:
        indices = [paths.index(path) for path in window]
        assert indices == list(range(indices[0], indices[0] + 3))
    assert selected.center_positions_mm == tuple(
        sorted(selected.center_positions_mm)
    )
    assert selected.position_features[0][0] == pytest.approx(0.0)
    assert selected.position_features[-1][0] == pytest.approx(1.0)


def test_physical_windows_keep_short_series_without_repeating_windows() -> None:
    paths = [Path(f"slice-{index}.dcm") for index in range(5)]

    selected = select_physical_windows(
        paths,
        10,
        lambda path: header(
            (0.0, 0.0, float(paths.index(path) * 3)),
            (1, 0, 0, 0, 1, 0),
            0,
        ),
    )

    assert len(selected.windows) == 3
    assert len(set(selected.windows)) == 3


def test_physical_windows_reject_missing_or_duplicate_geometry() -> None:
    paths = [Path(f"slice-{index}.dcm") for index in range(3)]
    missing = select_physical_windows(
        paths, 10, lambda path: header(None, None, 1)
    )
    duplicate = select_physical_windows(
        paths,
        10,
        lambda path: header((0, 0, 0), (1, 0, 0, 0, 1, 0), 1),
    )

    assert missing.windows == ()
    assert missing.method == "invalid-geometry:orientation"
    assert duplicate.windows == ()
    assert duplicate.method == "invalid-geometry:duplicate-position"
