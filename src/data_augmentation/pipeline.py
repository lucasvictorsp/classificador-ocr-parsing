"""End-to-end orchestration for synchronized image, mask, and OCR augmentation."""

from __future__ import annotations

import random
import shutil
from pathlib import Path

from data_augmentation.config import AugmentationConfig
from data_augmentation.models import AugmentationResult, DocumentTriplet
from data_augmentation.planner import scenario_for_index
from data_augmentation.utils.discovery import discover_triplets
from data_augmentation.utils.io import (
    copy_file,
    parse_annotations,
    read_image,
    read_mask,
    save_image,
    save_mask,
    write_annotations,
    write_jsonl,
    write_manifest_csv,
    write_params_json,
    write_sidecar_json,
    write_summary_csv,
)
from data_augmentation.utils.transforms import apply_replay_compose


def run_augmentation(config: AugmentationConfig) -> list[AugmentationResult]:
    """Run the complete augmentation workflow.

    Args:
        config: Validated pipeline configuration.

    Returns:
        Manifest records for copied originals and augmented variants.

    Raises:
        ValueError: If no valid document triplets are discovered.
    """
    if config.clean_output:
        _clean_output_dir(config)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.log_dir.mkdir(parents=True, exist_ok=True)

    triplets = discover_triplets(config)
    if not triplets:
        raise ValueError(f"Nenhuma trinca encontrada em {config.dataset_dir}")

    records: list[AugmentationResult] = []
    scenario_index_by_class = {class_name: 0 for class_name in config.class_names}

    for triplet in triplets:
        records.extend(_copy_original_if_needed(triplet, config))

        for variant_index in range(1, config.variants_per_document + 1):
            scenario_index = scenario_index_by_class[triplet.class_name]
            scenario = scenario_for_index(scenario_index)
            scenario_index_by_class[triplet.class_name] = scenario_index + 1

            variant_seed = _variant_seed(config.seed, triplet.class_name, triplet.document_id, variant_index)
            rng = random.Random(variant_seed)
            records.append(_augment_triplet(triplet, config, variant_index, scenario, rng, variant_seed))

    write_jsonl(config.log_dir / "augmentation_log.jsonl", records)
    write_manifest_csv(config.log_dir / "mlflow_manifest.csv", records)
    write_summary_csv(config.log_dir / "summary.csv", records)
    write_params_json(
        config.log_dir / "mlflow_params.json",
        {
            "dataset_dir": str(config.dataset_dir),
            "output_dir": str(config.output_dir),
            "seed": config.seed,
            "total_factor": config.total_factor,
            "copy_originals": config.copy_originals,
            "clean_output": config.clean_output,
            "classes": list(config.class_names),
            "documents_found": len(triplets),
            "records_written": len(records),
            "variants_per_document": config.variants_per_document,
        },
    )
    return records


def _copy_original_if_needed(
    triplet: DocumentTriplet,
    config: AugmentationConfig,
) -> list[AugmentationResult]:
    """Copy a source triplet into the output dataset when configured.

    Args:
        triplet: Source document triplet.
        config: Runtime configuration.

    Returns:
        A single manifest record when originals are copied, otherwise an empty list.
    """
    if not config.copy_originals:
        return []

    output_image, output_mask, output_annotation, output_json = _output_paths(
        triplet,
        config.output_dir,
        "orig",
    )
    _ensure_can_write(output_image, config.overwrite)
    _ensure_can_write(output_mask, config.overwrite)
    _ensure_can_write(output_annotation, config.overwrite)
    _ensure_can_write(output_json, config.overwrite)

    copy_file(triplet.image_path, output_image)
    copy_file(triplet.mask_path, output_mask)
    copy_file(triplet.annotation_path, output_annotation)

    image = read_image(triplet.image_path)
    height, width = image.shape[:2]
    boxes = parse_annotations(triplet.annotation_path)

    record = AugmentationResult(
        class_name=triplet.class_name,
        document_id=triplet.document_id,
        variant_index=0,
        source_image=str(triplet.image_path),
        source_mask=str(triplet.mask_path),
        source_annotation=str(triplet.annotation_path),
        output_image=str(output_image),
        output_mask=str(output_mask),
        output_annotation=str(output_annotation),
        is_augmented=False,
        scenario="original",
        seed=config.seed,
        original_width=width,
        original_height=height,
        output_width=width,
        output_height=height,
        source_boxes=len(boxes),
        output_boxes=len(boxes),
        transforms=[],
    )
    write_sidecar_json(output_json, record)
    return [record]


def _clean_output_dir(config: AugmentationConfig) -> None:
    """Remove the output directory before a fresh generation run.

    Args:
        config: Runtime configuration containing dataset and output paths.

    Raises:
        ValueError: If the output directory points to the dataset, project root, or
            a filesystem root.
    """
    output_dir = config.output_dir.resolve()
    dataset_dir = config.dataset_dir.resolve()
    project_dir = Path.cwd().resolve()

    if output_dir == dataset_dir:
        raise ValueError("--clean-output recusado: output_dir e dataset_dir sao a mesma pasta.")
    if output_dir == project_dir:
        raise ValueError("--clean-output recusado: output_dir aponta para a raiz do projeto.")
    if output_dir == Path(output_dir.anchor):
        raise ValueError("--clean-output recusado: output_dir aponta para a raiz do disco.")
    if not output_dir.is_relative_to(project_dir):
        raise ValueError("--clean-output recusado: output_dir esta fora da raiz do projeto.")
    if output_dir.exists():
        shutil.rmtree(output_dir)


def _augment_triplet(
    triplet: DocumentTriplet,
    config: AugmentationConfig,
    variant_index: int,
    scenario,
    rng: random.Random,
    variant_seed: int,
) -> AugmentationResult:
    """Generate one augmented variant for a source triplet.

    Args:
        triplet: Source document triplet.
        config: Runtime configuration.
        variant_index: One-based augmented variant number.
        scenario: Scenario object that builds the Albumentations pipeline.
        rng: Random generator used for deterministic scenario construction.
        variant_seed: Seed used by Albumentations and OpenCV.

    Returns:
        Manifest record describing the generated variant.
    """
    output_image, output_mask, output_annotation, output_json = _output_paths(
        triplet,
        config.output_dir,
        f"aug{variant_index:02d}",
    )
    _ensure_can_write(output_image, config.overwrite)
    _ensure_can_write(output_mask, config.overwrite)
    _ensure_can_write(output_annotation, config.overwrite)
    _ensure_can_write(output_json, config.overwrite)

    image = read_image(triplet.image_path)
    mask = read_mask(triplet.mask_path)
    boxes = parse_annotations(triplet.annotation_path)
    original_height, original_width = image.shape[:2]

    transform = scenario.build(rng, config.min_bbox_size)
    augmented = apply_replay_compose(
        image=image,
        mask=mask,
        boxes=boxes,
        transform=transform,
        seed=variant_seed,
    )

    save_image(augmented.image, output_image, quality=config.jpeg_quality)
    save_mask(augmented.mask, output_mask, quality=config.jpeg_quality)
    write_annotations(output_annotation, augmented.boxes)

    output_height, output_width = augmented.image.shape[:2]
    record = AugmentationResult(
        class_name=triplet.class_name,
        document_id=triplet.document_id,
        variant_index=variant_index,
        source_image=str(triplet.image_path),
        source_mask=str(triplet.mask_path),
        source_annotation=str(triplet.annotation_path),
        output_image=str(output_image),
        output_mask=str(output_mask),
        output_annotation=str(output_annotation),
        is_augmented=True,
        scenario=scenario.name,
        seed=variant_seed,
        original_width=original_width,
        original_height=original_height,
        output_width=output_width,
        output_height=output_height,
        source_boxes=len(boxes),
        output_boxes=len(augmented.boxes),
        transforms=augmented.records,
    )
    write_sidecar_json(output_json, record)
    return record


def _output_paths(
    triplet: DocumentTriplet,
    output_root: Path,
    suffix: str,
) -> tuple[Path, Path, Path, Path]:
    """Build output paths for a generated triplet.

    Args:
        triplet: Source document triplet.
        output_root: Root output directory.
        suffix: Variant suffix, such as ``orig`` or ``aug01``.

    Returns:
        Tuple with image, mask, annotation, and sidecar JSON output paths.
    """
    class_dir = output_root / triplet.class_name
    stem = f"{triplet.document_id}__{suffix}"
    image_path = class_dir / f"{stem}{triplet.image_path.suffix.lower()}"
    mask_path = class_dir / f"{stem}_mask{triplet.mask_path.suffix.lower()}"
    annotation_path = class_dir / f"{stem}.txt"
    json_path = class_dir / f"{stem}.json"
    return image_path, mask_path, annotation_path, json_path


def _ensure_can_write(path: Path, overwrite: bool) -> None:
    """Ensure an output path can be written safely.

    Args:
        path: Target file path.
        overwrite: Whether existing files may be replaced.

    Raises:
        FileExistsError: If the target exists and overwrite is disabled.
    """
    if path.exists() and not overwrite:
        raise FileExistsError(f"Arquivo ja existe. Use --overwrite para substituir: {path}")


def _variant_seed(base_seed: int, class_name: str, document_id: str, variant_index: int) -> int:
    """Derive a deterministic seed for a specific augmented variant.

    Args:
        base_seed: User-provided base seed.
        class_name: Dataset class name.
        document_id: Source document identifier.
        variant_index: One-based augmented variant index.

    Returns:
        Unsigned 32-bit seed value.
    """
    text = f"{base_seed}:{class_name}:{document_id}:{variant_index}"
    value = 2166136261
    for char in text:
        value ^= ord(char)
        value = (value * 16777619) & 0xFFFFFFFF
    return value
