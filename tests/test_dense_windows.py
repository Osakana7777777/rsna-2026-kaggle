from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from rsna2026.acquisition import (  # noqa: E402
    CLASSIFICATION_METHOD,
    SLOT_NAMES,
    AcquisitionClass,
)
from rsna2026.augmentation import AugmentationConfig  # noqa: E402
from rsna2026.dataset import (  # noqa: E402
    KneeStudyDataset,
    StudyRecord,
    build_study_records,
)
from rsna2026.train import (  # noqa: E402
    ExponentialMovingAverage,
    slice_selection_report,
    train_epoch,
)


def disabled_augmentation() -> AugmentationConfig:
    return AugmentationConfig.from_mapping(
        {
            "enabled": False,
            "rotation_degrees": 0.0,
            "rotation_probability": 0.0,
            "translation_fraction": 0.0,
            "scale_min": 1.0,
            "scale_max": 1.0,
            "affine_probability": 0.0,
            "brightness_limit": 0.0,
            "contrast_limit": 0.0,
            "brightness_contrast_probability": 0.0,
            "gamma_min": 1.0,
            "gamma_max": 1.0,
            "gamma_probability": 0.0,
            "noise_std_max": 0.0,
            "noise_probability": 0.0,
            "blur_kernel_sizes": [3],
            "blur_sigma_min": 1.0,
            "blur_sigma_max": 1.0,
            "blur_probability": 0.0,
        }
    )


def test_dataset_decodes_repeated_source_once_and_masks_incomplete_window(
    monkeypatch,
) -> None:
    first, second, third, fourth = (Path(f"{name}.dcm") for name in "abcd")
    record = StudyRecord(
        study_uid="study",
        window_paths=(
            (first, second, third),
            (second, third, fourth),
            (first, None, third),
            (None, None, None),
        ),
        position_features=((0.0, 1.0, 1.0),) * 4,
        num_slots=1,
        max_series_per_slot=2,
        selections=(object(), object()),
    )
    calls = []

    def fake_read(path, image_size, physical_fov_mm):
        del physical_fov_mm
        calls.append(path)
        return torch.full((1, image_size, image_size), float(len(calls)))

    monkeypatch.setattr("rsna2026.dataset.read_dicom", fake_read)
    images, present, positions, _, _ = KneeStudyDataset(
        [record], 4, 130.0
    )[0]

    assert images.shape == (1, 2, 2, 3, 4, 4)
    assert present.tolist() == [[[True, True], [False, False]]]
    assert positions.shape == (1, 2, 2, 3)
    assert calls == [first, second, third, fourth]
    assert torch.equal(images[0, 0, 0, 1], images[0, 0, 1, 0])


def test_build_records_retains_two_largest_series_and_pads_slots(
    tmp_path, monkeypatch
) -> None:
    rows = []
    for series_uid, count in (("series-c", 2), ("series-a", 3), ("series-b", 3)):
        series_dir = tmp_path / "study" / series_uid
        series_dir.mkdir(parents=True)
        for index in range(count):
            (series_dir / f"{index}.dcm").touch()
        rows.append(
            {
                "StudyInstanceUID": "study",
                "SeriesInstanceUID": series_uid,
                "Anatomical_Plane": "Sagittal",
                "Fluid_Sensitive": 1,
                "Fat_Suppression": 1,
            }
        )

    monkeypatch.setattr(
        "rsna2026.dataset.read_acquisition_header",
        lambda _: AcquisitionClass(
            weighting="pd",
            fluid_sensitive=True,
            fat_suppressed=True,
            weighting_source="test",
            fat_suppression_source="test",
        ),
    )

    def fake_select(paths, window_count):
        paths = tuple(sorted(paths))
        windows = (paths[:3],) if len(paths) >= 3 else ()
        return SimpleNamespace(
            windows=windows,
            position_features=((0.5, 1.0, 1.0),) if windows else (),
            method="physical-coordinate-v1" if windows else "missing",
            coverage_mm=6.0 if windows else 0.0,
            median_spacing_mm=3.0 if windows else 0.0,
            duplicate_positions=0,
            large_gaps=0,
            all_positions_mm=(0.0, 3.0, 6.0) if windows else (),
            target_positions_mm=(3.0,) if windows else (),
            center_positions_mm=(3.0,) if windows else (),
        )

    monkeypatch.setattr("rsna2026.dataset.select_physical_windows", fake_select)
    records = build_study_records(
        pd.DataFrame(rows),
        tmp_path,
        ["study"],
        window_count=2,
        max_series_per_slot=2,
        slot_names=SLOT_NAMES,
        classification_method=CLASSIFICATION_METHOD,
    )

    assert len(records) == 1
    record = records[0]
    assert [selection.series_uid for selection in record.selections[:2]] == [
        "series-a",
        "series-b",
    ]
    assert [selection.series_rank for selection in record.selections[:2]] == [1, 2]
    assert all(selection.series_uid is None for selection in record.selections[2:])
    assert len(record.window_paths) == len(SLOT_NAMES) * 2 * 2
    summary = slice_selection_report({"all": records}, 2, 2)["splits"]["all"][
        "summary"
    ]
    assert summary["selected_series"] == 2
    assert summary["selected_series_by_slot_and_rank"][SLOT_NAMES[0]] == {
        "1": 1,
        "2": 1,
    }


class TinyDenseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, images, present, positions):
        del images, present, positions
        return self.weight.expand(1, 12)


class CountingScheduler:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1


def test_train_epoch_steps_optimizer_after_gradient_accumulation(monkeypatch) -> None:
    samples = []
    for _ in range(5):
        samples.append(
            (
                torch.zeros(1, 1, 3, 2, 2),
                torch.ones(1, 1, dtype=torch.bool),
                torch.zeros(1, 1, 3),
                torch.ones(12),
                torch.ones(12),
            )
        )
    model = TinyDenseModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    original_step = optimizer.step
    optimizer_steps = []

    def counted_step(*args, **kwargs):
        optimizer_steps.append(1)
        return original_step(*args, **kwargs)

    monkeypatch.setattr(optimizer, "step", counted_step)
    scheduler = CountingScheduler()
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    loss = train_epoch(
        model,
        DataLoader(samples, batch_size=1),
        optimizer,
        scheduler,  # type: ignore[arg-type]
        scaler,
        torch.device("cpu"),
        torch.ones(12),
        disabled_augmentation(),
        gradient_accumulation_steps=2,
    )

    assert np.isfinite(loss)
    assert len(optimizer_steps) == 3
    assert scheduler.steps == 3


def test_ema_updates_after_each_optimizer_step() -> None:
    samples = [
        (
            torch.zeros(1, 1, 3, 2, 2),
            torch.ones(1, 1, dtype=torch.bool),
            torch.zeros(1, 1, 3),
            torch.ones(12),
            torch.ones(12),
        )
        for _ in range(2)
    ]
    model = TinyDenseModel()
    ema = ExponentialMovingAverage(model, decay=0.5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    train_epoch(
        model,
        DataLoader(samples, batch_size=1),
        optimizer,
        CountingScheduler(),  # type: ignore[arg-type]
        torch.amp.GradScaler("cuda", enabled=False),
        torch.device("cpu"),
        torch.ones(12),
        disabled_augmentation(),
        gradient_accumulation_steps=1,
        ema=ema,
    )

    assert model.weight.item() > ema.model.weight.item() > 0.0


@pytest.mark.parametrize("decay", [-0.1, 1.0, float("nan")])
def test_ema_rejects_invalid_decay(decay) -> None:
    with pytest.raises(ValueError, match="EMA decay"):
        ExponentialMovingAverage(TinyDenseModel(), decay)
