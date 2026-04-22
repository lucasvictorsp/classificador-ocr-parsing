"""Batch inference and final evaluation for a held-out real inference dataset."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from document_classifier.utils import ensure_dir, select_device, write_json
from document_classifier.utils.inference import load_checkpoint_model, predict_image

LOGGER = logging.getLogger(__name__)
IMAGE_SUFFIXES: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
TABULAR_SUFFIXES: tuple[str, ...] = (".csv", ".tsv", ".parquet", ".xlsx", ".xls")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run final batch inference on a real held-out dataset."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/document_classifier/best_model.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/document_classifier/real_inference"),
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mlflow-tracking-uri", default="file:./mlruns")
    parser.add_argument("--mlflow-experiment", default="brazilian_document_classifier")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def configure_logging() -> None:
    """Configure concise structured logs for local and Docker execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def is_inference_image(path: Path) -> bool:
    """Check whether a file should be used as an inference image.

    Args:
        path: Candidate file path.

    Returns:
        ``True`` when the file has a known image suffix and is not a mask file.
    """
    lower_name = path.name.lower()
    return (
        path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and not lower_name.endswith("_mask.jpg")
    )


def normalize_true_label(folder_name: str, class_names: list[str]) -> str | None:
    """Convert a folder name into an evaluation label.

    Args:
        folder_name: Immediate subfolder name.
        class_names: Known model classes.

    Returns:
        Known class name, ``outros`` for the external ``Outro`` folder, or
        ``None`` when the folder is not a class folder.
    """
    if folder_name in class_names:
        return folder_name
    if folder_name.lower() in {"outro", "outros", "desconhecida", "desconhecido"}:
        return "outros"
    return None


def discover_dataset_images(
    dataset_dir: Path,
    class_names: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Discover images from all immediate subfolders, tolerating empty folders.

    Args:
        dataset_dir: External inference dataset root.
        class_names: Known class names loaded from the checkpoint.

    Returns:
        Tuple with image records and warning messages.

    Raises:
        FileNotFoundError: If the dataset root does not exist.
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Inference dataset directory not found: {dataset_dir}")

    warnings: list[str] = []
    records: list[dict[str, Any]] = []
    subdirectories = sorted(path for path in dataset_dir.iterdir() if path.is_dir())
    if not subdirectories:
        warning = f"No subfolders found under inference dataset: {dataset_dir}"
        LOGGER.warning(warning)
        return records, [warning]

    for folder in subdirectories:
        images = sorted(path for path in folder.rglob("*") if is_inference_image(path))
        if not images:
            warning = f"Empty inference folder or no valid images: {folder}"
            warnings.append(warning)
            LOGGER.warning(warning)
            continue

        true_label = normalize_true_label(folder.name, class_names)
        if true_label is None:
            LOGGER.warning("Folder %s is not a known label; true_label will be empty.", folder)

        for image_path in images:
            records.append(
                {
                    "image_path": image_path,
                    "source_folder": folder.name,
                    "true_label": true_label,
                }
            )
    return records, warnings


def discover_tabular_files(dataset_dir: Path) -> list[Path]:
    """Find tabular files placed directly in the inference dataset root.

    Args:
        dataset_dir: External inference dataset root.

    Returns:
        Sorted tabular file paths. Exact filenames are preserved in MLflow.
    """
    return sorted(
        path
        for path in dataset_dir.iterdir()
        if path.is_file() and path.suffix.lower() in TABULAR_SUFFIXES
    )


def build_prediction_rows(
    image_records: list[dict[str, Any]],
    model: Any,
    metadata: dict[str, Any],
    threshold: float,
    device: Any,
) -> list[dict[str, Any]]:
    """Run inference and flatten probabilities into CSV-friendly rows.

    Args:
        image_records: Discovered image records.
        model: Loaded PyTorch model.
        metadata: Checkpoint metadata.
        threshold: Confidence threshold used for rejection.
        device: Torch device.

    Returns:
        Prediction rows ready to be saved as a DataFrame.
    """
    rows: list[dict[str, Any]] = []
    class_names = list(metadata["class_names"])
    for record in image_records:
        prediction = predict_image(
            model=model,
            image_path=record["image_path"],
            class_names=class_names,
            input_size=int(metadata["input_size"]),
            threshold=threshold,
            device=device,
        )
        rejected_by_threshold = prediction["final_label"] == "outros"
        row = {
            "image_path": prediction["image_path"],
            "source_folder": record["source_folder"],
            "true_label": record["true_label"],
            "predicted_label": prediction["predicted_label"],
            "final_label": prediction["final_label"],
            "max_probability": prediction["confidence"],
            "threshold": prediction["threshold"],
            "rejected_by_threshold": rejected_by_threshold,
        }
        for class_name, probability in prediction["probabilities"].items():
            row[f"prob_{class_name}"] = probability
        rows.append(row)
    return rows


def save_external_confusion_matrix(
    frame: pd.DataFrame,
    labels: list[str],
    output_png: Path,
    output_csv: Path,
) -> None:
    """Save a confusion matrix for final thresholded labels.

    Args:
        frame: Prediction DataFrame with ``true_label`` and ``final_label``.
        labels: Labels included in matrix order.
        output_png: Destination PNG path.
        output_csv: Destination CSV path.
    """
    import matplotlib.pyplot as plt

    matrix = confusion_matrix(frame["true_label"], frame["final_label"], labels=labels)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(output_csv)

    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel("Predito final")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusao - inferencia real")
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            ax.text(col_index, row_index, matrix[row_index, col_index], ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def compute_external_metrics(frame: pd.DataFrame, labels: list[str]) -> dict[str, float]:
    """Compute aggregate metrics when ground-truth labels are available.

    Args:
        frame: Prediction DataFrame.
        labels: Evaluation labels.

    Returns:
        Metrics dictionary for MLflow.
    """
    metrics: dict[str, float] = {
        "external_total_images": float(len(frame)),
        "external_rejected_count": (
            float(frame["rejected_by_threshold"].sum()) if len(frame) else 0.0
        ),
        "external_rejection_rate": (
            float(frame["rejected_by_threshold"].mean()) if len(frame) else 0.0
        ),
    }
    labeled_frame = frame.dropna(subset=["true_label"])
    if labeled_frame.empty:
        return metrics

    metrics["external_labeled_images"] = float(len(labeled_frame))
    metrics["external_accuracy_final"] = float(
        accuracy_score(labeled_frame["true_label"], labeled_frame["final_label"])
    )
    metrics["external_macro_f1_final"] = float(
        f1_score(
            labeled_frame["true_label"],
            labeled_frame["final_label"],
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )

    other_frame = labeled_frame[labeled_frame["true_label"] == "outros"]
    known_frame = labeled_frame[labeled_frame["true_label"] != "outros"]
    if not other_frame.empty:
        metrics["external_outros_recall"] = float(
            (other_frame["final_label"] == "outros").mean()
        )
    if not known_frame.empty:
        metrics["external_known_accuracy_final"] = float(
            accuracy_score(known_frame["true_label"], known_frame["final_label"])
        )
    return metrics


def save_external_reports(
    frame: pd.DataFrame,
    labels: list[str],
    output_dir: Path,
) -> dict[str, float]:
    """Save CSV reports, classification report, and confusion matrix.

    Args:
        frame: Prediction DataFrame.
        labels: Evaluation labels.
        output_dir: Directory for generated artifacts.

    Returns:
        Aggregate metrics.
    """
    ensure_dir(output_dir)
    predictions_csv = output_dir / "real_inference_predictions.csv"
    frame.to_csv(predictions_csv, index=False)

    metrics = compute_external_metrics(frame, labels)
    write_json(output_dir / "real_inference_metrics.json", metrics)

    labeled_frame = frame.dropna(subset=["true_label"])
    if not labeled_frame.empty:
        report = classification_report(
            labeled_frame["true_label"],
            labeled_frame["final_label"],
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
        report_frame = pd.DataFrame(report).transpose()
        report_frame.to_csv(output_dir / "real_inference_classification_report.csv")
        (output_dir / "real_inference_classification_report.txt").write_text(
            report_frame.to_string(),
            encoding="utf-8",
        )
        save_external_confusion_matrix(
            labeled_frame,
            labels,
            output_dir / "real_inference_confusion_matrix.png",
            output_dir / "real_inference_confusion_matrix.csv",
        )
    return metrics


def main() -> None:
    """Run final batch inference and log results to MLflow."""
    configure_logging()
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    device = select_device(args.device)
    model, metadata = load_checkpoint_model(args.checkpoint, device)
    threshold = args.threshold if args.threshold is not None else float(metadata["threshold"])
    class_names = list(metadata["class_names"])
    evaluation_labels = [*class_names, "outros"]

    image_records, warnings = discover_dataset_images(args.dataset_dir, class_names)
    tabular_files = discover_tabular_files(args.dataset_dir)
    rows = build_prediction_rows(image_records, model, metadata, threshold, device)
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "image_path",
                "source_folder",
                "true_label",
                "predicted_label",
                "final_label",
                "max_probability",
                "threshold",
                "rejected_by_threshold",
            ]
            + [f"prob_{class_name}" for class_name in class_names]
        )

    warnings_path = output_dir / "real_inference_warnings.txt"
    warnings_path.write_text("\n".join(warnings), encoding="utf-8")
    metrics = save_external_reports(frame, evaluation_labels, output_dir)
    metrics["external_empty_or_invalid_folders"] = float(len(warnings))
    metrics["external_tabular_files_found"] = float(len(tabular_files))
    write_json(output_dir / "real_inference_metrics.json", metrics)

    run_name = args.run_name or f"real_inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "mode": "real_inference_batch",
                "dataset_dir": str(args.dataset_dir),
                "checkpoint": str(args.checkpoint),
                "model_name": metadata["model_name"],
                "input_size": metadata["input_size"],
                "threshold": threshold,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(output_dir), artifact_path="real_inference_artifacts")
        for tabular_file in tabular_files:
            mlflow.log_artifact(str(tabular_file), artifact_path="real_inference_tabular_files")

    LOGGER.info("Processed %s images from %s.", len(frame), args.dataset_dir)
    LOGGER.info("Results saved to %s.", output_dir)


if __name__ == "__main__":
    main()
