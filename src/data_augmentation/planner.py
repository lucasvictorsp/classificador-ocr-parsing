"""Augmentation scenario catalog and round-robin scenario selection."""

from __future__ import annotations

import os
import random
from collections.abc import Callable
from dataclasses import dataclass

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A

from data_augmentation.utils.transforms import (
    affine,
    colored_background,
    make_replay_compose,
    perspective,
)


ScenarioBuilder = Callable[[random.Random, int], A.ReplayCompose]


@dataclass(frozen=True)
class AugmentationScenario:
    """Named augmentation scenario that builds an Albumentations replay pipeline.

    Attributes:
        name: Stable scenario identifier used in logs and manifests.
        build: Callable that returns a seeded ``ReplayCompose`` pipeline.
    """

    name: str
    build: ScenarioBuilder


def build_scenarios() -> tuple[AugmentationScenario, ...]:
    """Create the balanced scenario catalog.

    Returns:
        Tuple of available augmentation scenarios in round-robin order.
    """
    return (
        AugmentationScenario("phone_tilt_color_background", _phone_tilt_color_background),
        AugmentationScenario("perspective_shadow", _perspective_shadow),
        AugmentationScenario("low_light_blur", _low_light_blur),
        AugmentationScenario("noise_jpeg", _noise_jpeg),
        AugmentationScenario("strong_skew_color_background", _strong_skew_color_background),
        AugmentationScenario("multi_degradation", _multi_degradation),
    )


def scenario_for_index(index: int) -> AugmentationScenario:
    """Select a scenario by cycling through the catalog.

    Args:
        index: Zero-based scenario index for a class-specific round-robin counter.

    Returns:
        Augmentation scenario assigned to the index.
    """
    scenarios = build_scenarios()
    return scenarios[index % len(scenarios)]


def _phone_tilt_color_background(rng: random.Random, min_bbox_size: int) -> A.ReplayCompose:
    """Build a mild phone tilt scenario with colored background and lighting noise.

    Args:
        rng: Random generator used to sample deterministic fill colors.
        min_bbox_size: Minimum bounding-box size retained by Albumentations.

    Returns:
        Replayable Albumentations pipeline.
    """
    fill = colored_background(rng)
    return make_replay_compose(
        [
            affine(fill=fill, rotate=(-10, 10), scale=(0.90, 1.03), translate=(-0.055, 0.055)),
            A.RandomBrightnessContrast(brightness_limit=(-0.18, 0.18), contrast_limit=(-0.18, 0.22), p=1),
            A.HueSaturationValue(hue_shift_limit=(-4, 4), sat_shift_limit=(-16, 16), val_shift_limit=(-8, 8), p=0.65),
            A.RandomShadow(
                shadow_roi=(0, 0, 1, 1),
                num_shadows_limit=(1, 2),
                shadow_dimension=5,
                shadow_intensity_range=(0.30, 0.58),
                p=0.8,
            ),
        ],
        min_bbox_size,
    )


def _perspective_shadow(rng: random.Random, min_bbox_size: int) -> A.ReplayCompose:
    """Build a perspective and shadow scenario.

    Args:
        rng: Random generator used to sample deterministic fill colors.
        min_bbox_size: Minimum bounding-box size retained by Albumentations.

    Returns:
        Replayable Albumentations pipeline.
    """
    fill = colored_background(rng)
    return make_replay_compose(
        [
            perspective(fill=fill, scale=(0.035, 0.10)),
            affine(fill=fill, rotate=(-4, 4), scale=(0.92, 1.02), translate=(-0.035, 0.035)),
            A.RandomShadow(
                shadow_roi=(0, 0, 1, 1),
                num_shadows_limit=(1, 2),
                shadow_dimension=6,
                shadow_intensity_range=(0.38, 0.66),
                p=1,
            ),
            A.RandomBrightnessContrast(brightness_limit=(-0.14, 0.08), contrast_limit=(-0.10, 0.28), p=1),
        ],
        min_bbox_size,
    )


def _low_light_blur(rng: random.Random, min_bbox_size: int) -> A.ReplayCompose:
    """Build a low-light capture scenario with blur.

    Args:
        rng: Random generator used to sample deterministic fill colors.
        min_bbox_size: Minimum bounding-box size retained by Albumentations.

    Returns:
        Replayable Albumentations pipeline.
    """
    fill = colored_background(rng)
    return make_replay_compose(
        [
            affine(fill=fill, rotate=(-7, 7), scale=(0.91, 1.01), translate=(-0.04, 0.04), shear=(-2, 2)),
            A.RandomBrightnessContrast(brightness_limit=(-0.42, -0.18), contrast_limit=(-0.28, 0.05), p=1),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 7), p=1),
                    A.MotionBlur(blur_limit=(3, 9), allow_shifted=True, p=1),
                    A.Defocus(radius=(2, 5), alias_blur=(0.1, 0.35), p=1),
                ],
                p=1,
            ),
            A.RandomShadow(
                shadow_roi=(0, 0, 1, 1),
                num_shadows_limit=(1, 2),
                shadow_dimension=5,
                shadow_intensity_range=(0.35, 0.68),
                p=0.85,
            ),
        ],
        min_bbox_size,
    )


def _noise_jpeg(rng: random.Random, min_bbox_size: int) -> A.ReplayCompose:
    """Build a noisy compressed-photo scenario.

    Args:
        rng: Random generator used to sample deterministic fill colors.
        min_bbox_size: Minimum bounding-box size retained by Albumentations.

    Returns:
        Replayable Albumentations pipeline.
    """
    fill = colored_background(rng)
    return make_replay_compose(
        [
            affine(fill=fill, rotate=(-6, 6), scale=(0.92, 1.04), translate=(-0.045, 0.045)),
            A.GaussNoise(std_range=(0.015, 0.075), mean_range=(0.0, 0.0), per_channel=True, p=1),
            A.RandomBrightnessContrast(brightness_limit=(-0.24, 0.16), contrast_limit=(-0.22, 0.20), p=1),
            A.ImageCompression(compression_type="jpeg", quality_range=(48, 78), p=1),
        ],
        min_bbox_size,
    )


def _strong_skew_color_background(rng: random.Random, min_bbox_size: int) -> A.ReplayCompose:
    """Build a stronger skew and perspective scenario.

    Args:
        rng: Random generator used to sample deterministic fill colors.
        min_bbox_size: Minimum bounding-box size retained by Albumentations.

    Returns:
        Replayable Albumentations pipeline.
    """
    fill = colored_background(rng)
    return make_replay_compose(
        [
            perspective(fill=fill, scale=(0.07, 0.14)),
            affine(fill=fill, rotate=(-12, 12), scale=(0.88, 1.00), translate=(-0.065, 0.065), shear=(-4, 4)),
            A.RandomBrightnessContrast(brightness_limit=(-0.22, 0.22), contrast_limit=(-0.24, 0.34), p=1),
            A.HueSaturationValue(hue_shift_limit=(-6, 6), sat_shift_limit=(-22, 22), val_shift_limit=(-12, 12), p=0.75),
        ],
        min_bbox_size,
    )


def _multi_degradation(rng: random.Random, min_bbox_size: int) -> A.ReplayCompose:
    """Build a compound degradation scenario for harder production-like samples.

    Args:
        rng: Random generator used to sample deterministic fill colors.
        min_bbox_size: Minimum bounding-box size retained by Albumentations.

    Returns:
        Replayable Albumentations pipeline.
    """
    fill = colored_background(rng)
    return make_replay_compose(
        [
            perspective(fill=fill, scale=(0.04, 0.11)),
            affine(fill=fill, rotate=(-9, 9), scale=(0.89, 1.02), translate=(-0.055, 0.055), shear=(-2.5, 2.5)),
            A.RandomShadow(
                shadow_roi=(0, 0, 1, 1),
                num_shadows_limit=(1, 3),
                shadow_dimension=6,
                shadow_intensity_range=(0.32, 0.70),
                p=0.9,
            ),
            A.RandomBrightnessContrast(brightness_limit=(-0.32, 0.12), contrast_limit=(-0.30, 0.30), p=1),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1),
                    A.MotionBlur(blur_limit=(3, 7), allow_shifted=True, p=1),
                    A.NoOp(p=1),
                ],
                p=0.6,
            ),
            A.GaussNoise(std_range=(0.012, 0.055), mean_range=(0.0, 0.0), per_channel=True, p=0.75),
            A.ImageCompression(compression_type="jpeg", quality_range=(52, 86), p=1),
        ],
        min_bbox_size,
    )
