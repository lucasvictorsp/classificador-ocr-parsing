"""Batch command-line runner for OCR parsing and ground truth evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from ocr_parsing.config import (  # type: ignore[no-redef]
        DEFAULT_OCR_LANGUAGE,
        DEFAULT_TESSERACT_CONFIG,
        PipelineConfig,
    )
    from ocr_parsing.evaluation import (  # type: ignore[no-redef]
        aggregate_batch_metrics,
        compare_result_with_ground_truth,
        discover_batch_images,
        ground_truth_path_for_image,
        infer_document_type_from_path,
        save_batch_outputs,
    )
    from ocr_parsing.pipeline import run_pipeline  # type: ignore[no-redef]
else:
    from .config import DEFAULT_OCR_LANGUAGE, DEFAULT_TESSERACT_CONFIG, PipelineConfig
    from .evaluation import (
        aggregate_batch_metrics,
        compare_result_with_ground_truth,
        discover_batch_images,
        ground_truth_path_for_image,
        infer_document_type_from_path,
        save_batch_outputs,
    )
    from .pipeline import run_pipeline


DEFAULT_DATASET_DIR = Path("/mnt/d/Lucas/sample_dataset_data_augmented_12")


def build_parser() -> argparse.ArgumentParser:
    """Create the batch CLI parser.

    Returns:
        Configured ``argparse.ArgumentParser``.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Executa OCR + parsing em lote, arquivo a arquivo, e avalia contra "
            "os ground truths .txt pareados."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Diretório raiz do dataset com subpastas por tipo de documento.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "ocr_parsing_batch",
        help="Diretório onde artefatos por imagem e consolidados serão salvos.",
    )
    parser.add_argument(
        "--ocr-language",
        default=DEFAULT_OCR_LANGUAGE,
        help="Idiomas Tesseract, por exemplo 'por+eng'.",
    )
    parser.add_argument(
        "--tesseract-config",
        default=DEFAULT_TESSERACT_CONFIG,
        help="Flags extras do Tesseract.",
    )
    parser.add_argument(
        "--min-field-confidence",
        type=float,
        default=55.0,
        help="Confiança mínima antes de gerar avisos de OCR/parsing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limita o número de imagens processadas para testes rápidos.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continua o lote mesmo se uma imagem falhar.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse batch command-line arguments.

    Args:
        argv: Optional argument list.

    Returns:
        Parsed namespace.
    """

    return build_parser().parse_args(argv)


def process_one_image(
    image_path: Path,
    output_dir: Path,
    ocr_language: str,
    tesseract_config: str,
    min_field_confidence: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the single-file pipeline and evaluate one image.

    Args:
        image_path: Input image path.
        output_dir: Batch output directory.
        ocr_language: Tesseract language expression.
        tesseract_config: Extra Tesseract flags.
        min_field_confidence: Minimum confidence threshold.

    Returns:
        Tuple with pipeline payload and ground truth comparison bundle.

    Raises:
        ValueError: If document type cannot be inferred from the parent folder.
    """

    document_type = infer_document_type_from_path(image_path)
    if document_type is None:
        raise ValueError(f"Tipo de documento não inferido pela pasta: {image_path}")
    config = PipelineConfig(
        input_image=image_path,
        document_type=document_type,
        output_dir=output_dir / "por_imagem",
        ocr_language=ocr_language,
        tesseract_config=tesseract_config,
        min_field_confidence=min_field_confidence,
    )
    payload = run_pipeline(config)
    ground_truth_path = ground_truth_path_for_image(image_path)
    comparison = compare_result_with_ground_truth(
        payload=payload,
        image_path=image_path,
        ground_truth_path=ground_truth_path,
    )
    return payload, comparison


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    """Execute batch OCR parsing and evaluation.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Summary with counts, metrics, and artifact paths.

    Raises:
        FileNotFoundError: If the dataset directory does not exist.
        RuntimeError: If no valid images are found.
    """

    if not args.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {args.dataset_dir}")
    images = discover_batch_images(args.dataset_dir)
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        raise RuntimeError(f"Nenhuma imagem válida encontrada em: {args.dataset_dir}")

    results: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index, image_path in enumerate(images, start=1):
        print(f"[{index}/{len(images)}] Processando {image_path}")
        try:
            payload, comparison = process_one_image(
                image_path=image_path,
                output_dir=args.output_dir,
                ocr_language=args.ocr_language,
                tesseract_config=args.tesseract_config,
                min_field_confidence=args.min_field_confidence,
            )
        except Exception as exc:
            error = {
                "image_path": image_path.as_posix(),
                "document_type": infer_document_type_from_path(image_path),
                "error": str(exc),
            }
            errors.append(error)
            print(f"Erro em {image_path}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                raise
            continue
        results.append(payload)
        comparisons.append(comparison)

    metrics = aggregate_batch_metrics(results, comparisons, errors)
    artifacts = save_batch_outputs(args.output_dir, results, comparisons, metrics)
    return {
        "total_imagens_descobertas": len(images),
        "total_sucesso": len(results),
        "total_erros": len(errors),
        "artefatos": artifacts,
        "metricas": metrics,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the batch command.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit code.
    """

    args = parse_args(argv)
    try:
        summary = run_batch(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1
    print("Lote concluído.")
    print(f"Imagens descobertas: {summary['total_imagens_descobertas']}")
    print(f"Sucesso: {summary['total_sucesso']}")
    print(f"Erros: {summary['total_erros']}")
    print(f"Relatório: {summary['artefatos']['relatorio_metricas_json']}")
    print(f"Resumo executivo: {summary['artefatos']['resumo_executivo_metricas_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
