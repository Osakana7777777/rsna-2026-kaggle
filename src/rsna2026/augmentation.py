"""Synchronized training augmentation for three-slice MRI inputs."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch.nn import functional


@dataclass(frozen=True)
class AugmentationConfig:
    """Validated parameters for conservative MRI image augmentation."""

    enabled: bool
    rotation_degrees: float
    rotation_probability: float
    translation_fraction: float
    scale_min: float
    scale_max: float
    affine_probability: float
    brightness_limit: float
    contrast_limit: float
    brightness_contrast_probability: float
    gamma_min: float
    gamma_max: float
    gamma_probability: float
    noise_std_max: float
    noise_probability: float
    blur_kernel_sizes: tuple[int, ...]
    blur_sigma_min: float
    blur_sigma_max: float
    blur_probability: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "AugmentationConfig":
        """Build and validate an augmentation configuration table."""
        required = {
            "enabled",
            "rotation_degrees",
            "rotation_probability",
            "translation_fraction",
            "scale_min",
            "scale_max",
            "affine_probability",
            "brightness_limit",
            "contrast_limit",
            "brightness_contrast_probability",
            "gamma_min",
            "gamma_max",
            "gamma_probability",
            "noise_std_max",
            "noise_probability",
            "blur_kernel_sizes",
            "blur_sigma_min",
            "blur_sigma_max",
            "blur_probability",
        }
        missing = sorted(required.difference(values))
        if missing:
            raise ValueError(f"Augmentation configuration is missing: {missing}")
        if not isinstance(values["enabled"], bool):
            raise ValueError("Augmentation enabled must be a boolean")
        config = cls(
            enabled=values["enabled"],
            rotation_degrees=float(values["rotation_degrees"]),
            rotation_probability=float(values["rotation_probability"]),
            translation_fraction=float(values["translation_fraction"]),
            scale_min=float(values["scale_min"]),
            scale_max=float(values["scale_max"]),
            affine_probability=float(values["affine_probability"]),
            brightness_limit=float(values["brightness_limit"]),
            contrast_limit=float(values["contrast_limit"]),
            brightness_contrast_probability=float(
                values["brightness_contrast_probability"]
            ),
            gamma_min=float(values["gamma_min"]),
            gamma_max=float(values["gamma_max"]),
            gamma_probability=float(values["gamma_probability"]),
            noise_std_max=float(values["noise_std_max"]),
            noise_probability=float(values["noise_probability"]),
            blur_kernel_sizes=tuple(
                int(value) for value in values["blur_kernel_sizes"]
            ),
            blur_sigma_min=float(values["blur_sigma_min"]),
            blur_sigma_max=float(values["blur_sigma_max"]),
            blur_probability=float(values["blur_probability"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Reject parameters that cannot define a safe augmentation."""
        numeric = {
            name: value
            for name, value in asdict(self).items()
            if name not in {"enabled", "blur_kernel_sizes"}
        }
        if not all(math.isfinite(value) for value in numeric.values()):
            raise ValueError("Augmentation numeric values must be finite")
        probabilities = {
            "rotation_probability": self.rotation_probability,
            "affine_probability": self.affine_probability,
            "brightness_contrast_probability": (
                self.brightness_contrast_probability
            ),
            "gamma_probability": self.gamma_probability,
            "noise_probability": self.noise_probability,
            "blur_probability": self.blur_probability,
        }
        for name, probability in probabilities.items():
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        nonnegative = {
            "rotation_degrees": self.rotation_degrees,
            "translation_fraction": self.translation_fraction,
            "brightness_limit": self.brightness_limit,
            "contrast_limit": self.contrast_limit,
            "noise_std_max": self.noise_std_max,
        }
        for name, value in nonnegative.items():
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if self.translation_fraction > 1.0:
            raise ValueError("translation_fraction must not exceed 1")
        if self.contrast_limit > 1.0:
            raise ValueError("contrast_limit must not exceed 1")
        if not 0.0 < self.scale_min <= self.scale_max:
            raise ValueError("Scale range must be positive and ordered")
        if not 0.0 < self.gamma_min <= self.gamma_max:
            raise ValueError("Gamma range must be positive and ordered")
        if not 0.0 < self.blur_sigma_min <= self.blur_sigma_max:
            raise ValueError("Blur sigma range must be positive and ordered")
        if not self.blur_kernel_sizes or any(
            size <= 0 or size % 2 == 0 for size in self.blur_kernel_sizes
        ):
            raise ValueError("Blur kernel sizes must be positive odd integers")

    def metadata(self) -> dict[str, object]:
        """Return JSON-compatible experiment provenance."""
        values = asdict(self)
        values["blur_kernel_sizes"] = list(self.blur_kernel_sizes)
        return values

    def is_active(self) -> bool:
        """Return whether any stochastic transform can change an image."""
        return self.enabled and any(
            probability > 0.0
            for probability in (
                self.rotation_probability,
                self.affine_probability,
                self.brightness_contrast_probability,
                self.gamma_probability,
                self.noise_probability,
                self.blur_probability,
            )
        )


def _uniform(
    count: int,
    lower: float,
    upper: float,
    reference: torch.Tensor,
) -> torch.Tensor:
    return lower + (upper - lower) * torch.rand(
        count, device=reference.device, dtype=reference.dtype
    )


def _apply_affine(
    images: torch.Tensor, config: AugmentationConfig
) -> torch.Tensor:
    count = len(images)
    rotation_mask = torch.rand(count, device=images.device) < (
        config.rotation_probability
    )
    angles = _uniform(
        count, -config.rotation_degrees, config.rotation_degrees, images
    )
    angles = torch.where(rotation_mask, angles, torch.zeros_like(angles))

    affine_mask = torch.rand(count, device=images.device) < config.affine_probability
    scales = _uniform(count, config.scale_min, config.scale_max, images)
    scales = torch.where(affine_mask, scales, torch.ones_like(scales))
    translate_x = _uniform(
        count, -config.translation_fraction, config.translation_fraction, images
    )
    translate_y = _uniform(
        count, -config.translation_fraction, config.translation_fraction, images
    )
    translate_x = torch.where(
        affine_mask, translate_x, torch.zeros_like(translate_x)
    )
    translate_y = torch.where(
        affine_mask, translate_y, torch.zeros_like(translate_y)
    )

    radians = torch.deg2rad(angles)
    cosine = torch.cos(radians)
    sine = torch.sin(radians)
    normalized_x = 2.0 * translate_x
    normalized_y = 2.0 * translate_y
    theta = images.new_zeros((count, 2, 3))
    theta[:, 0, 0] = cosine / scales
    theta[:, 0, 1] = sine / scales
    theta[:, 1, 0] = -sine / scales
    theta[:, 1, 1] = cosine / scales
    theta[:, 0, 2] = -(cosine * normalized_x + sine * normalized_y) / scales
    theta[:, 1, 2] = (sine * normalized_x - cosine * normalized_y) / scales
    grid = functional.affine_grid(theta, images.shape, align_corners=False)
    return functional.grid_sample(
        images,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )


def _apply_blur(
    images: torch.Tensor, config: AugmentationConfig
) -> torch.Tensor:
    apply_mask = torch.rand(len(images), device=images.device) < config.blur_probability
    for index in apply_mask.nonzero(as_tuple=False).flatten().tolist():
        kernel_index = int(
            torch.randint(
                len(config.blur_kernel_sizes), (1,), device=images.device
            ).item()
        )
        kernel_size = config.blur_kernel_sizes[kernel_index]
        sigma = float(
            _uniform(
                1, config.blur_sigma_min, config.blur_sigma_max, images
            ).item()
        )
        coordinates = torch.arange(
            kernel_size, device=images.device, dtype=images.dtype
        ) - kernel_size // 2
        kernel_1d = torch.exp(-(coordinates.square()) / (2.0 * sigma * sigma))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        channels = images.shape[1]
        weight = kernel_2d.expand(channels, 1, kernel_size, kernel_size)
        padding = kernel_size // 2
        sample = functional.pad(
            images[index : index + 1],
            (padding, padding, padding, padding),
            mode="reflect",
        )
        images[index : index + 1] = functional.conv2d(
            sample, weight, groups=channels
        )
    return images


def augment_2_5d_slots(
    images: torch.Tensor,
    present: torch.Tensor,
    num_slots: int,
    slices_per_slot: int,
    config: AugmentationConfig,
) -> torch.Tensor:
    """Augment each slot while sharing parameters across its slice channels."""
    if not config.is_active():
        return images
    if images.ndim != 5 or images.shape[2] != 1:
        raise ValueError("Expected [batch, views, 1, height, width] images")
    batch_size, num_views, _, height, width = images.shape
    if num_views != num_slots * slices_per_slot:
        raise ValueError("Image view count does not match slots and slices")
    if present.shape != (batch_size, num_views):
        raise ValueError("Presence mask shape does not match images")
    if height <= max(config.blur_kernel_sizes) // 2 or width <= max(
        config.blur_kernel_sizes
    ) // 2:
        raise ValueError("Images are too small for configured reflection padding")

    slot_images = images.reshape(
        batch_size, num_slots, slices_per_slot, height, width
    ).reshape(batch_size * num_slots, slices_per_slot, height, width)
    slot_present = present.reshape(
        batch_size * num_slots, slices_per_slot, 1, 1
    ).to(dtype=slot_images.dtype)

    augmented = _apply_affine(slot_images, config)
    count = len(augmented)
    intensity_mask = torch.rand(count, device=images.device) < (
        config.brightness_contrast_probability
    )
    brightness = _uniform(
        count, -config.brightness_limit, config.brightness_limit, images
    )
    contrast = _uniform(
        count, 1.0 - config.contrast_limit, 1.0 + config.contrast_limit, images
    )
    brightness = torch.where(intensity_mask, brightness, torch.zeros_like(brightness))
    contrast = torch.where(intensity_mask, contrast, torch.ones_like(contrast))
    augmented = (
        (augmented - 0.5) * contrast[:, None, None, None]
        + 0.5
        + brightness[:, None, None, None]
    ).clamp(0.0, 1.0)

    gamma_mask = torch.rand(count, device=images.device) < config.gamma_probability
    gamma = _uniform(count, config.gamma_min, config.gamma_max, images)
    gamma = torch.where(gamma_mask, gamma, torch.ones_like(gamma))
    augmented = augmented.pow(gamma[:, None, None, None])

    noise_mask = torch.rand(count, device=images.device) < config.noise_probability
    noise_std = _uniform(count, 0.0, config.noise_std_max, images)
    noise_std = torch.where(noise_mask, noise_std, torch.zeros_like(noise_std))
    augmented = augmented + torch.randn_like(augmented) * noise_std[
        :, None, None, None
    ]
    augmented = _apply_blur(augmented, config).clamp(0.0, 1.0)
    augmented = augmented * slot_present
    return augmented.reshape(batch_size, num_views, 1, height, width)


def augment_dense_windows(
    images: torch.Tensor,
    present: torch.Tensor,
    config: AugmentationConfig,
) -> torch.Tensor:
    """Apply one synchronized transform to every window in an acquisition slot."""
    if not config.is_active():
        return images
    if images.ndim != 6 or images.shape[3] != 3:
        raise ValueError("Expected [batch, slots, windows, 3, height, width]")
    batch_size, num_slots, window_count, channels, height, width = images.shape
    if present.shape != (batch_size, num_slots, window_count):
        raise ValueError("Window mask shape does not match dense windows")
    if height <= max(config.blur_kernel_sizes) // 2 or width <= max(
        config.blur_kernel_sizes
    ) // 2:
        raise ValueError("Images are too small for configured reflection padding")

    slot_images = images.reshape(
        batch_size * num_slots, window_count * channels, height, width
    )
    augmented = _apply_affine(slot_images, config)
    count = len(augmented)
    intensity_mask = torch.rand(count, device=images.device) < (
        config.brightness_contrast_probability
    )
    brightness = _uniform(
        count, -config.brightness_limit, config.brightness_limit, images
    )
    contrast = _uniform(
        count, 1.0 - config.contrast_limit, 1.0 + config.contrast_limit, images
    )
    brightness = torch.where(intensity_mask, brightness, torch.zeros_like(brightness))
    contrast = torch.where(intensity_mask, contrast, torch.ones_like(contrast))
    augmented = (
        (augmented - 0.5) * contrast[:, None, None, None]
        + 0.5
        + brightness[:, None, None, None]
    ).clamp(0.0, 1.0)
    gamma_mask = torch.rand(count, device=images.device) < config.gamma_probability
    gamma = _uniform(count, config.gamma_min, config.gamma_max, images)
    gamma = torch.where(gamma_mask, gamma, torch.ones_like(gamma))
    augmented = augmented.pow(gamma[:, None, None, None])
    noise_mask = torch.rand(count, device=images.device) < config.noise_probability
    noise_std = _uniform(count, 0.0, config.noise_std_max, images)
    noise_std = torch.where(noise_mask, noise_std, torch.zeros_like(noise_std))
    shared_noise = torch.randn(
        (count, 1, height, width), device=images.device, dtype=images.dtype
    )
    augmented = augmented + shared_noise * noise_std[:, None, None, None]
    augmented = _apply_blur(augmented, config).clamp(0.0, 1.0)
    window_mask = present.reshape(
        batch_size * num_slots, window_count, 1, 1, 1
    ).to(augmented.dtype)
    augmented = augmented.reshape(
        batch_size * num_slots, window_count, channels, height, width
    )
    return (augmented * window_mask).reshape_as(images)
