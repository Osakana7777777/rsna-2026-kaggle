from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from rsna2026.augmentation import AugmentationConfig  # noqa: E402
from rsna2026.dataset import KneeStudyDataset, StudyRecord  # noqa: E402
from rsna2026.train import train_epoch  # noqa: E402


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

    assert images.shape == (2, 2, 3, 4, 4)
    assert present.tolist() == [[True, True], [False, False]]
    assert positions.shape == (2, 2, 3)
    assert calls == [first, second, third, fourth]
    assert torch.equal(images[0, 0, 1], images[0, 1, 0])


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
