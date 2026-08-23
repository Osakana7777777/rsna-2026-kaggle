"""Deterministic DICOM slice ordering and central-band sampling."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SliceHeader:
    """Spatial metadata needed to order one DICOM slice."""

    image_position: tuple[float, float, float] | None
    image_orientation: tuple[float, float, float, float, float, float] | None
    instance_number: float | None


@dataclass(frozen=True)
class OrderedSlices:
    """A complete series order and the metadata source used for it."""

    paths: tuple[Path, ...]
    method: str


HeaderReader = Callable[[Path], SliceHeader]


def read_slice_header(path: Path) -> SliceHeader:
    """Read only the DICOM attributes needed for ordering."""
    import pydicom

    dataset = pydicom.dcmread(
        path,
        force=True,
        stop_before_pixels=True,
        specific_tags=[
            "ImagePositionPatient",
            "ImageOrientationPatient",
            "InstanceNumber",
        ],
    )
    return SliceHeader(
        image_position=_finite_tuple(
            getattr(dataset, "ImagePositionPatient", None), 3
        ),
        image_orientation=_finite_tuple(
            getattr(dataset, "ImageOrientationPatient", None), 6
        ),
        instance_number=_finite_number(getattr(dataset, "InstanceNumber", None)),
    )


def _finite_tuple(value: object, length: int) -> tuple[float, ...] | None:
    try:
        numbers = tuple(float(item) for item in value)  # type: ignore[union-attr]
    except (TypeError, ValueError):
        return None
    if len(numbers) < length or not all(math.isfinite(item) for item in numbers[:length]):
        return None
    return numbers[:length]


def _finite_number(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _slice_normal(
    orientation: tuple[float, float, float, float, float, float] | None,
) -> tuple[float, float, float] | None:
    if orientation is None:
        return None
    row = orientation[:3]
    column = orientation[3:]
    normal = (
        row[1] * column[2] - row[2] * column[1],
        row[2] * column[0] - row[0] * column[2],
        row[0] * column[1] - row[1] * column[0],
    )
    normal_length = math.sqrt(sum(value * value for value in normal))
    if not math.isfinite(normal_length) or normal_length <= 1e-6:
        return None
    return tuple(value / normal_length for value in normal)


def projected_slice_position(header: SliceHeader) -> float | None:
    """Project image position onto the normal of its image plane."""
    position = header.image_position
    normal = _slice_normal(header.image_orientation)
    if position is None or normal is None:
        return None
    coordinate = sum(position[index] * normal[index] for index in range(3))
    return coordinate if math.isfinite(coordinate) else None


def order_slice_paths(
    paths: Sequence[Path], header_reader: HeaderReader = read_slice_header
) -> OrderedSlices:
    """Order a whole series by geometry, instance number, or file name."""
    lexical_paths = tuple(sorted(paths))
    if not lexical_paths:
        return OrderedSlices((), "missing")

    headers = []
    for path in lexical_paths:
        try:
            headers.append(header_reader(path))
        except Exception:
            headers.append(SliceHeader(None, None, None))

    normals = [_slice_normal(header.image_orientation) for header in headers]
    positions = [header.image_position for header in headers]
    reference_normal = normals[0]
    aligned = (
        reference_normal is not None
        and all(
            normal is not None
            and sum(
                reference_normal[index] * normal[index] for index in range(3)
            )
            > 0.99
            for normal in normals
        )
    )
    if aligned and all(position is not None for position in positions):
        coordinates = [
            sum(position[index] * reference_normal[index] for index in range(3))
            for position in positions
        ]
        ordered = sorted(
            zip(lexical_paths, coordinates),
            key=lambda item: (float(item[1]), item[0].as_posix()),
        )
        return OrderedSlices(tuple(path for path, _ in ordered), "geometry")

    instance_numbers = [header.instance_number for header in headers]
    if all(number is not None for number in instance_numbers):
        ordered = sorted(
            zip(lexical_paths, instance_numbers),
            key=lambda item: (float(item[1]), item[0].as_posix()),
        )
        return OrderedSlices(tuple(path for path, _ in ordered), "instance_number")

    return OrderedSlices(lexical_paths, "file_name")


def central_band_indices(
    length: int,
    count: int,
    band_start: float,
    band_end: float,
) -> tuple[int, ...]:
    """Choose fixed-count indices spread across a central relative band."""
    if length < 0:
        raise ValueError("Series length must not be negative")
    if count <= 0:
        raise ValueError("Slice count must be positive")
    if not 0.0 <= band_start < band_end <= 1.0:
        raise ValueError("Slice band must satisfy 0 <= start < end <= 1")
    if length == 0:
        return ()
    if length <= count:
        indices = list(range(length))
        indices.extend([indices[-1]] * (count - len(indices)))
        return tuple(indices)

    lower_position = band_start * (length - 1)
    upper_position = band_end * (length - 1)
    lower = math.ceil(lower_position)
    upper = math.floor(upper_position)
    while upper - lower + 1 < count:
        left_distance = (
            lower_position - (lower - 1) if lower > 0 else math.inf
        )
        right_distance = (
            (upper + 1) - upper_position if upper < length - 1 else math.inf
        )
        if left_distance <= right_distance + 1e-12:
            lower -= 1
        else:
            upper += 1
    if count == 1:
        return ((lower + upper) // 2,)
    return tuple(
        math.floor(
            lower + index * (upper - lower) / (count - 1) + 0.5
        )
        for index in range(count)
    )


def sample_ordered_paths(
    paths: Sequence[Path],
    count: int,
    band_start: float,
    band_end: float,
    header_reader: HeaderReader = read_slice_header,
) -> tuple[list[Path], str, tuple[int, ...], tuple[float, ...]]:
    """Order a series and return selected paths with relative positions."""
    ordered = order_slice_paths(paths, header_reader)
    indices = central_band_indices(
        len(ordered.paths), count, band_start, band_end
    )
    selected = [ordered.paths[index] for index in indices]
    if len(ordered.paths) == 1:
        relative_positions = tuple(0.5 for _ in indices)
    else:
        denominator = max(len(ordered.paths) - 1, 1)
        relative_positions = tuple(index / denominator for index in indices)
    return selected, ordered.method, indices, relative_positions
