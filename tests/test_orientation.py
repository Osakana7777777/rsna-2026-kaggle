from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rsna2026.dataset import read_dicom
from rsna2026.orientation import decide_orientation


@pytest.mark.parametrize(
    ("orientation", "position_x", "laterality", "plane", "flip"),
    [
        ((1, 0, 0, 0, 0, -1), -100, "R", "coronal", False),
        ((1, 0, 0, 0, 0, -1), 100, "L", "coronal", True),
        ((-1, 0, 0, 0, 0, -1), 100, "L", "coronal", False),
        ((1, 0, 0, 0, 1, 0), 100, "L", "axial", True),
        ((1, 0, 0, 0, 1, 0), -100, "R", "axial", False),
        ((0, 1, 0, 0, 0, -1), 100, "L", "sagittal", False),
    ],
)
def test_patient_orientation_normalizes_only_coronal_and_axial_planes(
    orientation, position_x, laterality, plane, flip
) -> None:
    decision = decide_orientation(
        {
            "ImageOrientationPatient": orientation,
            "ImagePositionPatient": (position_x, 0, 0),
            "PixelSpacing": (1, 1),
            "Laterality": laterality,
        },
        (4, 4),
    )

    assert decision.plane == plane
    assert decision.geometry_laterality == laterality
    assert decision.resolved_laterality == laterality
    assert decision.horizontal_flip is flip
    assert decision.tag_geometry_disagreement is False


def test_geometry_controls_and_audits_laterality_tag_disagreement() -> None:
    decision = decide_orientation(
        {
            "ImageOrientationPatient": (1, 0, 0, 0, 0, -1),
            "ImagePositionPatient": (100, 0, 0),
            "PixelSpacing": (1, 1),
            "Laterality": "R",
        },
        (4, 4),
    )

    assert decision.geometry_laterality == "L"
    assert decision.resolved_laterality == "L"
    assert decision.laterality_source == "geometry"
    assert decision.horizontal_flip is True
    assert decision.tag_geometry_disagreement is True
    assert decision.audit_reason == "laterality-tag-geometry-disagreement"


def test_geometry_is_unresolved_when_field_of_view_crosses_midline() -> None:
    decision = decide_orientation(
        {
            "ImageOrientationPatient": (1, 0, 0, 0, 1, 0),
            "ImagePositionPatient": (-1, 0, 0),
            "PixelSpacing": (1, 1),
            "Laterality": "L",
        },
        (4, 4),
    )

    assert decision.geometry_laterality is None
    assert decision.resolved_laterality == "L"
    assert decision.laterality_source == "dicom-tag"


def test_read_dicom_applies_orientation_flip_before_preprocessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pixels = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
    dataset = SimpleNamespace(
        pixel_array=pixels,
        PixelSpacing=(1.0, 1.0),
        ImageOrientationPatient=(1, 0, 0, 0, 0, -1),
        ImagePositionPatient=(100, 0, 0),
        Laterality="L",
        RescaleSlope=1.0,
        RescaleIntercept=0.0,
        PhotometricInterpretation="MONOCHROME2",
    )
    monkeypatch.setattr(
        "rsna2026.dataset.pydicom.dcmread", lambda *args, **kwargs: dataset
    )

    image = read_dicom(Path("left-coronal.dcm"), 3, 2.0)

    assert tuple(image.shape) == (1, 3, 3)
    assert image[0, 1, 0] > image[0, 1, 2]
