"""Training entry point for the Brazilian document classifier."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from document_classifier.data import (
    KNOWN_CLASSES,
    DocumentImageDataset,
    build_transforms,
    discover_samples,
    split_samples,
    write_split_csv,
)
from document_classifier.losses import build_loss
from document_classifier.metrics import (
    choose_rejection_threshold,
    classification_report_frame,
    evaluate_model,
    save_confidence_analysis,
    save_confusion_matrix,
)
from document_classifier.models import build_model
from document_classifier.utils import ensure_dir, select_device, set_seed, write_json


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train a lightweight document classifier.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset_augmented"))
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("dataset_augmented/logs/mlflow_manifest.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/document_classifier"))
    parser.add_argument("--model-name", default="efficientnet_b0")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--loss-name", default="cross_entropy")
    parser.add_argument("--threshold-percentile", type=float, default=5.0)
    parser.add_argument("--train-last-blocks", type=int, default=1)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-freeze-backbone", action="store_true")
    parser.add_argument("--mlflow-tracking-uri", default="file:./mlruns")
    parser.add_argument("--mlflow-experiment", default="brazilian_document_classifier")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def make_loader(
    dataset: DocumentImageDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    """Create a memory-conscious DataLoader.

    Args:
        dataset: Dataset used by the loader.
        batch_size: Batch size.
        shuffle: Whether to shuffle samples.
        num_workers: Number of worker processes.

    Returns:
        Configured PyTorch DataLoader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train the model for one epoch.

    Args:
        model: Classifier being trained.
        loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device used for tensors.

    Returns:
        Mean training loss and accuracy.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, targets in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += batch_size
    return running_loss / max(1, total), correct / max(1, total)


def save_checkpoint(
    output_path: Path,
    model: torch.nn.Module,
    metadata: dict[str, Any],
) -> None:
    """Save model weights and inference metadata.

    Args:
        output_path: Destination checkpoint path.
        model: Trained model.
        metadata: Metadata needed by inference.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, output_path)


def main() -> None:
    """Run training, validation, threshold selection, testing, and MLflow logging."""
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)
    output_dir = ensure_dir(args.output_dir)
    reports_dir = ensure_dir(output_dir / "reports")

    samples = discover_samples(args.dataset_dir, KNOWN_CLASSES)
    splits = split_samples(samples, seed=args.seed)
    write_split_csv(splits, output_dir / "splits.csv")

    transform_map = build_transforms(args.input_size)
    train_dataset = DocumentImageDataset(splits["train"], transform_map["train"])
    val_dataset = DocumentImageDataset(splits["val"], transform_map["eval"])
    test_dataset = DocumentImageDataset(splits["test"], transform_map["eval"])

    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_dataset, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test_dataset, args.batch_size, False, args.num_workers)

    model_info = build_model(
        num_classes=len(KNOWN_CLASSES),
        model_name=args.model_name,
        pretrained=not args.no_pretrained,
        freeze_backbone=not args.no_freeze_backbone,
        train_last_blocks=args.train_last_blocks,
    )
    model = model_info.model.to(device)
    criterion = build_loss(args.loss_name)
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    run_name = args.run_name or (
        f"{args.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{args.seed}"
    )
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    checkpoint_path = output_dir / "best_model.pt"
    class_names = list(KNOWN_CLASSES)
    label_map = {str(index): class_name for index, class_name in enumerate(class_names)}
    write_json(output_dir / "label_map.json", label_map)

    metadata: dict[str, Any] = {
        "class_names": class_names,
        "model_name": args.model_name,
        "input_size": args.input_size,
        "threshold": None,
        "threshold_percentile": args.threshold_percentile,
        "pretrained": not args.no_pretrained,
        "freeze_backbone": not args.no_freeze_backbone,
        "train_last_blocks": args.train_last_blocks,
    }

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model_name": args.model_name,
                "input_size": args.input_size,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "patience": args.patience,
                "seed": args.seed,
                "loss_name": args.loss_name,
                "threshold_percentile": args.threshold_percentile,
                "pretrained": not args.no_pretrained,
                "freeze_backbone": not args.no_freeze_backbone,
                "train_last_blocks": args.train_last_blocks,
                "frozen_parameters": model_info.frozen_parameters,
                "trainable_parameters": model_info.trainable_parameters,
                "total_parameters": model_info.total_parameters,
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "test_samples": len(test_dataset),
                "device": str(device),
            }
        )

        for epoch in range(1, args.epochs + 1):
            train_loss, train_accuracy = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_result = evaluate_model(model, val_loader, criterion, device)
            scheduler.step(val_result.loss)

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_result.loss,
                    "val_accuracy": val_result.accuracy,
                    "val_macro_f1": val_result.macro_f1,
                },
                step=epoch,
            )

            if val_result.loss < best_val_loss:
                best_val_loss = val_result.loss
                best_epoch = epoch
                epochs_without_improvement = 0
                save_checkpoint(checkpoint_path, model, metadata)
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= args.patience:
                break

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        val_result = evaluate_model(model, val_loader, criterion, device)
        test_result = evaluate_model(model, test_loader, criterion, device)
        threshold = choose_rejection_threshold(val_result.max_probs, args.threshold_percentile)

        metadata["threshold"] = threshold
        metadata["best_epoch"] = best_epoch
        save_checkpoint(checkpoint_path, model, metadata)
        write_json(output_dir / "threshold.json", metadata)

        test_report = classification_report_frame(
            test_result.y_true,
            test_result.y_pred,
            class_names,
        )
        test_report.to_csv(reports_dir / "classification_report_test.csv")
        (reports_dir / "classification_report_test.txt").write_text(
            test_report.to_string(), encoding="utf-8"
        )
        save_confusion_matrix(
            test_result.y_true,
            test_result.y_pred,
            class_names,
            reports_dir / "confusion_matrix_test.png",
            reports_dir / "confusion_matrix_test.csv",
        )
        val_threshold_summary = save_confidence_analysis(
            val_result,
            threshold,
            class_names,
            reports_dir / "validation_confidence.csv",
        )
        test_threshold_summary = save_confidence_analysis(
            test_result,
            threshold,
            class_names,
            reports_dir / "test_confidence.csv",
        )

        metrics_payload = {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "val_loss": val_result.loss,
            "val_accuracy": val_result.accuracy,
            "val_macro_f1": val_result.macro_f1,
            "test_loss": test_result.loss,
            "test_accuracy": test_result.accuracy,
            "test_macro_f1": test_result.macro_f1,
            "threshold": threshold,
            "validation_rejection_rate": val_threshold_summary["rejection_rate"],
            "test_rejection_rate": test_threshold_summary["rejection_rate"],
            "test_accepted_accuracy": test_threshold_summary["accepted_accuracy"],
        }
        write_json(reports_dir / "metrics_test.json", metrics_payload)
        mlflow.log_metrics(metrics_payload)

        pd.DataFrame(
            [
                {"class_index": index, "class_name": class_name}
                for index, class_name in enumerate(class_names)
            ]
        ).to_csv(output_dir / "classes.csv", index=False)

        if args.manifest_path.exists():
            mlflow.log_artifact(str(args.manifest_path), artifact_path="dataset_manifest")
        mlflow.log_artifacts(str(output_dir), artifact_path="training_artifacts")
        mlflow.pytorch.log_model(model, artifact_path="model")


if __name__ == "__main__":
    main()
