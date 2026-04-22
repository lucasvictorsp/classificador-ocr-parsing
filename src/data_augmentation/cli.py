"""Command-line interface for synchronized document dataset augmentation."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_augmentation.config import DEFAULT_CLASSES, AugmentationConfig
from data_augmentation.pipeline import run_augmentation


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for dataset augmentation.

    Returns:
        Configured ``argparse.ArgumentParser`` instance.
    """
    parser = argparse.ArgumentParser(
        description="Data augmentation sincronizado para imagem, mascara e anotacoes de documentos."
    )
    parser.add_argument("--dataset", type=Path, default=Path("dataset"), help="Pasta raiz do dataset.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset_augmented"),
        help="Pasta onde o dataset aumentado sera criado.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatoria base.")
    parser.add_argument(
        "--total-factor",
        type=int,
        default=1,
        help="Quantidade de augmentations sinteticas por documento real.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=list(DEFAULT_CLASSES),
        help="Classes/pastas a processar.",
    )
    parser.add_argument(
        "--max-documents-per-class",
        type=int,
        default=None,
        help="Limita documentos por classe para testes rapidos.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=92,
        help="Qualidade JPEG para os arquivos gerados.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Substitui arquivos existentes na pasta de saida.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove a pasta de saida antes de gerar o novo dataset.",
    )
    parser.add_argument(
        "--no-copy-originals",
        action="store_true",
        help="Gera apenas arquivos augmentados, sem copiar originais.",
    )
    return parser


def main() -> None:
    """Run the augmentation pipeline from command-line arguments."""
    args = build_parser().parse_args()
    config = AugmentationConfig(
        dataset_dir=args.dataset,
        output_dir=args.output,
        seed=args.seed,
        total_factor=args.total_factor,
        class_names=tuple(args.classes),
        copy_originals=not args.no_copy_originals,
        overwrite=args.overwrite,
        clean_output=args.clean_output,
        jpeg_quality=args.jpeg_quality,
        max_documents_per_class=args.max_documents_per_class,
    )
    records = run_augmentation(config)
    augmented = sum(record.is_augmented for record in records)
    originals = len(records) - augmented
    print(f"Concluido: {originals} originais, {augmented} augmentados, {len(records)} registros.")
    print(f"Manifest MLflow: {config.log_dir / 'mlflow_manifest.csv'}")
