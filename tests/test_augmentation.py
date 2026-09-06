from __future__ import annotations

from dataclasses import replace

import pytest


torch = pytest.importorskip("torch")

from rsna2026.augmentation import (  # noqa: E402
    AugmentationConfig,
    augment_2_5d_slots,
    augment_dense_windows,
)


def augmentation_config(**overrides) -> AugmentationConfig:
    values = {
        "enabled": True,
        "rotation_degrees": 8.0,
        "rotation_probability": 0.5,
        "translation_fraction": 0.05,
        "scale_min": 0.92,
        "scale_max": 1.08,
        "affine_probability": 0.5,
        "brightness_limit": 0.10,
        "contrast_limit": 0.10,
        "brightness_contrast_probability": 0.4,
        "gamma_min": 0.85,
        "gamma_max": 1.15,
        "gamma_probability": 0.3,
        "noise_std_max": 0.02,
        "noise_probability": 0.2,
        "blur_kernel_sizes": [3, 5],
        "blur_sigma_min": 0.1,
        "blur_sigma_max": 1.0,
        "blur_probability": 0.1,
    }
    values.update(overrides)
    return AugmentationConfig.from_mapping(values)


def zero_probability_config() -> AugmentationConfig:
    return augmentation_config(
        rotation_probability=0.0,
        affine_probability=0.0,
        brightness_contrast_probability=0.0,
        gamma_probability=0.0,
        noise_probability=0.0,
        blur_probability=0.0,
    )


def test_disabled_and_zero_probability_augmentation_are_identity() -> None:
    images = torch.rand((2, 6, 1, 12, 12))
    present = torch.ones((2, 6), dtype=torch.bool)

    disabled = augment_2_5d_slots(
        images, present, 2, 3, replace(zero_probability_config(), enabled=False)
    )
    zero_probability = augment_2_5d_slots(
        images, present, 2, 3, zero_probability_config()
    )

    assert disabled.data_ptr() == images.data_ptr()
    assert zero_probability.data_ptr() == images.data_ptr()
    assert torch.equal(disabled, images)
    assert torch.equal(zero_probability, images)


def test_geometry_is_shared_within_slots_and_independent_between_slots() -> None:
    pattern = torch.zeros((12, 12))
    pattern[2:7, 4:9] = 1.0
    images = pattern.repeat(1, 6, 1, 1, 1)
    present = torch.ones((1, 6), dtype=torch.bool)
    config = replace(
        zero_probability_config(),
        rotation_probability=1.0,
        affine_probability=1.0,
    )

    torch.manual_seed(7)
    augmented = augment_2_5d_slots(images, present, 2, 3, config)
    slots = augmented.reshape(1, 2, 3, 12, 12)

    assert torch.allclose(slots[0, 0, 0], slots[0, 0, 1])
    assert torch.allclose(slots[0, 0, 1], slots[0, 0, 2])
    assert torch.allclose(slots[0, 1, 0], slots[0, 1, 1])
    assert not torch.allclose(slots[0, 0, 0], slots[0, 1, 0])


def test_missing_slices_remain_zero_and_output_is_bounded() -> None:
    images = torch.full((1, 6, 1, 12, 12), 0.5)
    present = torch.tensor([[True, False, True, False, False, False]])
    config = replace(
        zero_probability_config(),
        brightness_contrast_probability=1.0,
        gamma_probability=1.0,
        noise_probability=1.0,
        blur_probability=1.0,
    )

    torch.manual_seed(11)
    augmented = augment_2_5d_slots(images, present, 2, 3, config)

    assert augmented.shape == images.shape
    assert torch.isfinite(augmented).all()
    assert augmented.min() >= 0.0
    assert augmented.max() <= 1.0
    assert torch.count_nonzero(augmented[:, ~present[0]]) == 0


def test_augmentation_is_reproducible_for_a_fixed_seed() -> None:
    images = torch.rand((2, 6, 1, 12, 12))
    present = torch.ones((2, 6), dtype=torch.bool)
    config = augmentation_config()

    torch.manual_seed(23)
    first = augment_2_5d_slots(images, present, 2, 3, config)
    torch.manual_seed(23)
    second = augment_2_5d_slots(images, present, 2, 3, config)

    assert torch.equal(first, second)


def test_dense_windows_share_slot_transform_and_mask_missing_windows() -> None:
    pattern = torch.zeros((12, 12))
    pattern[2:7, 4:9] = 1.0
    images = pattern.repeat(1, 2, 4, 3, 1, 1)
    present = torch.tensor([[[True, True, False, True], [True] * 4]])
    config = replace(
        zero_probability_config(),
        rotation_probability=1.0,
        affine_probability=1.0,
        noise_probability=1.0,
        noise_std_max=0.02,
    )

    torch.manual_seed(31)
    augmented = augment_dense_windows(images, present, config)

    assert augmented.shape == images.shape
    assert torch.count_nonzero(augmented[0, 0, 2]) == 0
    assert torch.equal(augmented[0, 0, 0, 0], augmented[0, 0, 1, 0])
    assert not torch.equal(augmented[0, 0, 0, 0], augmented[0, 1, 0, 0])


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"rotation_probability": 1.1}, "within"),
        ({"translation_fraction": -0.1}, "nonnegative"),
        ({"scale_min": 0.0}, "Scale"),
        ({"gamma_min": 1.2, "gamma_max": 0.8}, "Gamma"),
        ({"blur_kernel_sizes": [3, 4]}, "odd"),
        ({"blur_sigma_min": 0.0}, "Blur sigma"),
    ],
)
def test_rejects_invalid_configuration(override, message) -> None:
    with pytest.raises(ValueError, match=message):
        augmentation_config(**override)


def test_metadata_is_json_compatible() -> None:
    metadata = augmentation_config().metadata()

    assert metadata["enabled"] is True
    assert metadata["blur_kernel_sizes"] == [3, 5]
