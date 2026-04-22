"""Command-line interface for the OCR parsing pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from ocr_parsing.config import (  # type: ignore[no-redef]
        DEFAULT_OCR_LANGUAGE,
        DEFAULT_TESSERACT_CONFIG,
        VALID_DOCUMENT_TYPES,
        PipelineConfig,
    )
    from ocr_parsing.pipeline import run_pipeline  # type: ignore[no-redef]
else:
    from .config import (
        DEFAULT_OCR_LANGUAGE,
        DEFAULT_TESSERACT_CONFIG,
        VALID_DOCUMENT_TYPES,
        PipelineConfig,
    )
    from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser.

    Returns:
        Configured ``argparse.ArgumentParser`` instance.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Executa retificação, pré-processamento, OCR e parsing de um documento "
            "brasileiro cujo tipo já é conhecido."
        )
    )
    parser.add_argument("--image", required=True, type=Path, help="Caminho da imagem do documento.")
    parser.add_argument(
        "--document-type",
        required=True,
        choices=VALID_DOCUMENT_TYPES,
        help="Tipo conhecido do documento; esta etapa não faz classificação.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "ocr_parsing",
        help="Diretório onde os artefatos da execução serão salvos.",
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
        help="Confiança mínima por campo antes de gerar aviso.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Imprime o JSON final completo no stdout.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list. When ``None``, argparse reads ``sys.argv``.

    Returns:
        Parsed argument namespace.
    """

    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the OCR parsing command.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit code.
    """

    args = parse_args(argv)
    config = PipelineConfig(
        input_image=args.image,
        document_type=args.document_type,
        output_dir=args.output_dir,
        ocr_language=args.ocr_language,
        tesseract_config=args.tesseract_config,
        min_field_confidence=args.min_field_confidence,
    )
    try:
        payload = run_pipeline(config)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1
    if args.print_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"JSON salvo em: {payload['artefatos']['json_resultado']}")
        print(f"Imagem retificada: {payload['artefatos']['imagem_retificada']}")
        print(f"Imagem pré-OCR: {payload['artefatos']['imagem_pre_processada_ocr']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
