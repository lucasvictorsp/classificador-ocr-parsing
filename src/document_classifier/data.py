"""Dataset discovery, leakage-safe splitting, and PyTorch datasets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from document_classifier.constants import IMAGENET_MEAN, IMAGENET_STD, KNOWN_CLASSES
ORIGIN_PATTERN = re.compile(r"^(?P<origin>.+?)__(?:orig|aug\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class ImageSample:
    """Single image sample used by the classifier.

    Attributes:
        image_path: Path to the input image.
        label: Class folder name.
        label_index: Integer target used by PyTorch.
        origin_id: Identifier shared by an original image and its augmentations.
        group_id: Leakage-safe split key, combining class and origin ID.
    """

    image_path: Path
    label: str
    label_index: int
    origin_id: str
    group_id: str


class DocumentImageDataset(Dataset[tuple[torch.Tensor, int]]):
    """PyTorch dataset for Brazilian document images.

    Attributes:
        samples: Samples assigned to this split.
        transform: Image transform applied before returning tensors.
    """

    def __init__(self, samples: list[ImageSample], transform: transforms.Compose) -> None:
        """Initialize the dataset.

        Args:
            samples: Samples assigned to the dataset split.
            transform: Torchvision transform pipeline.
        """
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of samples in this dataset.

        Returns:
            Number of image samples.
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Load and transform one image sample.

        Args:
            index: Sample position.

        Returns:
            Tuple with image tensor and integer label.
        """
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, sample.label_index


def extract_origin_id(image_path: Path) -> str:
    """Extract the original document ID from an image filename.

    The expected project naming convention is ``<id>__orig.jpg`` for original
    copies and ``<id>__augNN.jpg`` for augmented variants. If a file does not
    match the convention, the filename stem before the first ``__`` is used.

    Args:
        image_path: Path to an image file.

    Returns:
        Origin identifier shared by all variants of the same source document.
    """
    stem = image_path.stem
    match = ORIGIN_PATTERN.match(stem)
    if match:
        return match.group("origin")
    return stem.split("__", maxsplit=1)[0]


def is_training_image(path: Path) -> bool:
    """Check whether a path is a classifier input image.

    Args:
        path: Candidate file path.

    Returns:
        ``True`` when the file is a ``.jpg`` image and is not a mask image.
    """
    lower_name = path.name.lower()
    return path.is_file() and path.suffix.lower() == ".jpg" and not lower_name.endswith("_mask.jpg")


def discover_samples(
    dataset_dir: Path,
    class_names: Iterable[str] = KNOWN_CLASSES,
) -> list[ImageSample]:
    """Discover valid classifier samples from the augmented dataset directory.

    Args:
        dataset_dir: Root directory with one folder per known class.
        class_names: Expected class folder names.

    Returns:
        List of discovered image samples.

    Raises:
        FileNotFoundError: If the dataset root or a class folder is missing.
        ValueError: If no valid training images are found.
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    samples: list[ImageSample] = []
    class_to_index = {class_name: index for index, class_name in enumerate(class_names)}
    for class_name in class_names:
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Class directory not found: {class_dir}")
        for image_path in sorted(class_dir.iterdir()):
            if not is_training_image(image_path):
                continue
            origin_id = extract_origin_id(image_path)
            samples.append(
                ImageSample(
                    image_path=image_path,
                    label=class_name,
                    label_index=class_to_index[class_name],
                    origin_id=origin_id,
                    group_id=f"{class_name}:{origin_id}",
                )
            )

    if not samples:
        raise ValueError(f"No valid .jpg images found under {dataset_dir}")
    return samples


def samples_to_frame(samples: list[ImageSample]) -> pd.DataFrame:
    """Convert samples to a DataFrame for splitting and audit artifacts.

    Args:
        samples: Discovered image samples.

    Returns:
        DataFrame with paths, labels, and leakage-safe group identifiers.
    """
    return pd.DataFrame(
        [
            {
                "image_path": str(sample.image_path),
                "label": sample.label,
                "label_index": sample.label_index,
                "origin_id": sample.origin_id,
                "group_id": sample.group_id,
            }
            for sample in samples
        ]
    )


def _can_stratify(labels: pd.Series, test_fraction: float) -> bool:
    """Decide whether a stratified split is feasible for the given labels.

    Args:
        labels: Class labels at group level.
        test_fraction: Fraction assigned to the second split.

    Returns:
        ``True`` when each class has enough examples for both split sides.
    """
    counts = labels.value_counts()
    if counts.empty or counts.min() < 2:
        return False
    test_count = int(round(len(labels) * test_fraction))
    train_count = len(labels) - test_count
    return test_count >= labels.nunique() and train_count >= labels.nunique()


def split_samples(
    samples: list[ImageSample],
    seed: int,
    train_fraction: float = 0.70,
    val_fraction: float = 0.20,
    test_fraction: float = 0.10,
) -> dict[str, list[ImageSample]]:
    """Split samples by source-document group without augmentation leakage.

    Args:
        samples: Discovered image samples.
        seed: Fixed random seed.
        train_fraction: Fraction of groups assigned to training.
        val_fraction: Fraction of groups assigned to validation.
        test_fraction: Fraction of groups assigned to testing.

    Returns:
        Mapping from split name to samples.

    Raises:
        ValueError: If split fractions do not sum to one.
    """
    total_fraction = train_fraction + val_fraction + test_fraction
    if abs(total_fraction - 1.0) > 1e-6:
        raise ValueError("Split fractions must sum to 1.0")

    frame = samples_to_frame(samples)
    group_frame = frame[["group_id", "label"]].drop_duplicates().reset_index(drop=True)

    temp_fraction = val_fraction + test_fraction
    stratify = group_frame["label"] if _can_stratify(group_frame["label"], temp_fraction) else None
    train_groups, temp_groups = train_test_split(
        group_frame,
        test_size=temp_fraction,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )

    relative_test_fraction = test_fraction / temp_fraction
    temp_stratify = (
        temp_groups["label"]
        if _can_stratify(temp_groups["label"], relative_test_fraction)
        else None
    )
    val_groups, test_groups = train_test_split(
        temp_groups,
        test_size=relative_test_fraction,
        random_state=seed,
        shuffle=True,
        stratify=temp_stratify,
    )

    group_to_split = {
        **{group_id: "train" for group_id in train_groups["group_id"]},
        **{group_id: "val" for group_id in val_groups["group_id"]},
        **{group_id: "test" for group_id in test_groups["group_id"]},
    }

    splits: dict[str, list[ImageSample]] = {"train": [], "val": [], "test": []}
    for sample in samples:
        splits[group_to_split[sample.group_id]].append(sample)
    return splits


def build_transforms(input_size: int) -> dict[str, transforms.Compose]:
    """Create lightweight training and evaluation transforms.

    Args:
        input_size: Model input size in pixels.

    Returns:
        Transform pipelines for ``train`` and ``eval``.
    """
    train_transform = transforms.Compose(
        [
            transforms.Resize(int(input_size * 1.14)),
            transforms.RandomResizedCrop(input_size, scale=(0.86, 1.0), ratio=(0.90, 1.10)),
            transforms.RandomRotation(degrees=4),
            transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.04),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return {"train": train_transform, "eval": eval_transform}


def write_split_csv(splits: dict[str, list[ImageSample]], output_path: Path) -> None:
    """Write split assignment details to a CSV artifact.

    Args:
        splits: Mapping from split name to samples.
        output_path: Destination CSV path.
    """
    rows = []
    for split_name, split_samples_list in splits.items():
        for sample in split_samples_list:
            rows.append(
                {
                    "split": split_name,
                    "image_path": str(sample.image_path),
                    "label": sample.label,
                    "label_index": sample.label_index,
                    "origin_id": sample.origin_id,
                    "group_id": sample.group_id,
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
