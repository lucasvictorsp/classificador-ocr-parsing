"""Inference entry point with confidence-based unknown rejection."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from document_classifier.data import is_training_image
from document_classifier.utils import select_device, write_json
from document_classifier.utils.inference import load_checkpoint_model, predict_image


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run document classifier inference.")
    parser.add_argument("--input", type=Path, required=True, help="Image file or directory.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/document_classifier/best_model.pt"),
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def iter_input_images(input_path: Path) -> list[Path]:
    """Collect input images for inference.

    Args:
        input_path: File or directory path.

    Returns:
        List of image paths to classify.

    Raises:
        FileNotFoundError: If the input path does not exist.
        ValueError: If no valid image is found.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    if input_path.is_file():
        return [input_path]
    images = sorted(path for path in input_path.rglob("*.jpg") if is_training_image(path))
    if not images:
        raise ValueError(f"No .jpg images found for inference under {input_path}")
    return images


def main() -> None:
    """Run inference for one image or a directory of images."""
    args = parse_args()
    device = select_device(args.device)
    model, metadata = load_checkpoint_model(args.checkpoint, device)
    threshold = args.threshold if args.threshold is not None else float(metadata["threshold"])
    image_paths = iter_input_images(args.input)
    predictions = [
        predict_image(
            model=model,
            image_path=image_path,
            class_names=metadata["class_names"],
            input_size=int(metadata["input_size"]),
            threshold=threshold,
            device=device,
        )
        for image_path in image_paths
    ]

    if args.output_json:
        write_json(args.output_json, {"predictions": predictions})
    if args.output_csv:
        rows = [
            {
                "image_path": prediction["image_path"],
                "predicted_label": prediction["predicted_label"],
                "final_label": prediction["final_label"],
                "confidence": prediction["confidence"],
                "threshold": prediction["threshold"],
            }
            for prediction in predictions
        ]
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.output_csv, index=False)

    for prediction in predictions:
        print(
            f"{prediction['image_path']} -> {prediction['final_label']} "
            f"(pred={prediction['predicted_label']}, "
            f"conf={prediction['confidence']:.4f}, threshold={threshold:.4f})"
        )


if __name__ == "__main__":
    main()
