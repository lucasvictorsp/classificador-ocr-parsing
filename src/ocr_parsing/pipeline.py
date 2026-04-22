"""End-to-end OCR parsing pipeline for one known document image."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .ocr_engine import OCRResult, run_tesseract_ocr
from .parsing import ParsedDocument, parse_document
from .preprocessing import preprocess_for_ocr
from .rectification import rectify_document
from .utils.io import build_output_paths, read_image, write_image, write_json, write_text


def path_to_string(path: Path) -> str:
    """Convert a path to a normalized string for JSON output.

    Args:
        path: Filesystem path.

    Returns:
        Path converted to POSIX-style text.
    """

    return path.as_posix()


def build_result_payload(
    config: PipelineConfig,
    paths: dict[str, Path],
    rectification_metadata: dict[str, object],
    ocr_result: OCRResult,
    ocr_selection: dict[str, Any],
    timings: dict[str, float],
    parsed: ParsedDocument,
) -> dict[str, Any]:
    """Build the final JSON payload returned and saved by the pipeline.

    Args:
        config: Pipeline execution configuration.
        paths: Output artifact paths.
        rectification_metadata: Metadata from document rectification.
        ocr_result: Structured OCR result.
        ocr_selection: Metadata for OCR candidates and selected image version.
        timings: In-memory processing timings in seconds.
        parsed: Parsed document fields and warnings.

    Returns:
        JSON-serializable result payload.
    """

    field_quality = {
        field_name: {
            "ocr_confidence": (
                round(parsed.confidence_by_field[field_name], 2)
                if parsed.confidence_by_field.get(field_name) is not None
                else None
            ),
            "parsing_confidence": round(
                parsed.parsing_confidence_by_field.get(field_name, 0.0),
                2,
            ),
            "parsing_signals": parsed.parsing_signals_by_field.get(field_name, {}),
        }
        for field_name in parsed.fields
    }

    return {
        "tipo_documento": parsed.document_type,
        "imagem_entrada": path_to_string(config.input_image),
        "artefatos": {
            "diretorio_execucao": path_to_string(paths["run_dir"]),
            "imagem_retificada": path_to_string(paths["rectified_image"]),
            "imagem_pre_processada_ocr": path_to_string(paths["ocr_image"]),
            "texto_ocr": path_to_string(paths["ocr_text"]),
            "ocr_detalhado": path_to_string(paths["ocr_data"]),
            "json_resultado": path_to_string(paths["parsed_json"]),
        },
        "campos_extraidos": parsed.fields,
        "confianca_por_campo": {
            key: round(value, 2) if value is not None else None
            for key, value in parsed.confidence_by_field.items()
        },
        "qualidade_por_campo": field_quality,
        "confianca_parsing_por_campo": {
            key: round(value, 2)
            for key, value in parsed.parsing_confidence_by_field.items()
        },
        "validacao_parsing": parsed.parsing_signals_by_field,
        "avisos": parsed.warnings,
        "ocr": {
            "texto_completo": ocr_result.text,
            "confianca_media": round(ocr_result.mean_confidence, 2),
            "quantidade_linhas": len(ocr_result.lines),
            "quantidade_palavras": len(ocr_result.words),
            "versao_imagem_escolhida": ocr_selection["selected_version"],
            "candidatos": ocr_selection["candidates"],
        },
        "tempos": {
            "tempo_ocr_retificada": round(timings.get("tempo_ocr_retificada", 0.0), 6),
            "tempo_ocr_pre_ocr": round(timings.get("tempo_ocr_pre_ocr", 0.0), 6),
            "tempo_parsing_total": round(timings.get("tempo_parsing_total", 0.0), 6),
            "tempo_total_pipeline": round(timings.get("tempo_total_pipeline", 0.0), 6),
            "unidade": "segundos",
            "observacao": "Nao inclui escrita de artefatos em disco.",
        },
        "retificacao": rectification_metadata,
        "metadados": {
            "executado_em_utc": datetime.now(timezone.utc).isoformat(),
            "ocr_engine": "tesseract",
            "ocr_language": config.ocr_language,
            "tesseract_config": config.tesseract_config,
            "estrategia": "opencv_retificacao_preprocessamento_leve_tesseract_multi_ocr_regras",
        },
    }


def choose_best_ocr_result(candidates: dict[str, OCRResult]) -> tuple[str, OCRResult, dict[str, Any]]:
    """Choose the OCR candidate with the best mean confidence.

    Args:
        candidates: Mapping from image-version name to OCR result.

    Returns:
        Tuple with selected version, selected OCR result, and auditable selection metadata.
    """

    selected_version, selected_result = max(
        candidates.items(),
        key=lambda item: (
            item[1].mean_confidence,
            len(item[1].words),
            1 if item[0] == "pre_ocr" else 0,
        ),
    )
    metadata = {
        "selected_version": selected_version,
        "selection_rule": "maior_confianca_media_tesseract_desempate_por_palavras_e_pre_ocr",
        "candidates": {
            version: {
                "confianca_media": round(result.mean_confidence, 2),
                "quantidade_linhas": len(result.lines),
                "quantidade_palavras": len(result.words),
                "texto_vazio": not bool(result.text.strip()),
            }
            for version, result in candidates.items()
        },
    }
    return selected_version, selected_result, metadata


def run_timed_ocr(image: Any, language: str, config: str) -> tuple[OCRResult, float]:
    """Run OCR and measure only the OCR execution time.

    Args:
        image: Image passed to Tesseract.
        language: Tesseract language expression.
        config: Extra Tesseract command-line flags.

    Returns:
        Tuple with OCR result and elapsed time in seconds.
    """

    start = time.perf_counter()
    result = run_tesseract_ocr(image, language=language, config=config)
    elapsed = time.perf_counter() - start
    return result, elapsed


def run_multi_version_ocr(
    rectified_image: Any,
    ocr_image: Any,
    language: str,
    config: str,
) -> tuple[OCRResult, dict[str, Any], dict[str, float]]:
    """Run OCR on rectified and OCR-preprocessed images, then select the best result.

    Args:
        rectified_image: Geometrically rectified document image.
        ocr_image: Final preprocessed OCR image.
        language: Tesseract language expression.
        config: Extra Tesseract command-line flags.

    Returns:
        Tuple with selected OCR result, selection metadata, and OCR timings.
    """

    retified_result, retified_time = run_timed_ocr(
        rectified_image,
        language=language,
        config=config,
    )
    pre_ocr_result, pre_ocr_time = run_timed_ocr(
        ocr_image,
        language=language,
        config=config,
    )
    candidates = {
        "retificada": retified_result,
        "pre_ocr": pre_ocr_result,
    }
    _, selected_result, metadata = choose_best_ocr_result(candidates)
    timings = {
        "tempo_ocr_retificada": retified_time,
        "tempo_ocr_pre_ocr": pre_ocr_time,
    }
    metadata["candidates"]["retificada"]["tempo_ocr"] = round(retified_time, 6)
    metadata["candidates"]["pre_ocr"]["tempo_ocr"] = round(pre_ocr_time, 6)
    metadata["timings"] = {key: round(value, 6) for key, value in timings.items()}
    return selected_result, metadata, timings


def save_pipeline_artifacts(
    paths: dict[str, Path],
    rectified_image: Any,
    ocr_image: Any,
    ocr_result: OCRResult,
    payload: dict[str, Any],
) -> None:
    """Persist pipeline images, OCR artifacts, and final JSON.

    Args:
        paths: Output artifact paths.
        rectified_image: Rectified document image.
        ocr_image: Preprocessed OCR image.
        ocr_result: Structured OCR result.
        payload: Final JSON payload.
    """

    write_image(paths["rectified_image"], rectified_image)
    write_image(paths["ocr_image"], ocr_image)
    write_text(paths["ocr_text"], ocr_result.text)
    write_json(paths["ocr_data"], ocr_result.to_dict())
    write_json(paths["parsed_json"], payload)


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    """Execute rectification, preprocessing, OCR, parsing, and artifact saving.

    Args:
        config: Pipeline execution configuration.

    Returns:
        Final JSON payload with extracted fields and artifact paths.
    """

    config.validate()
    paths = build_output_paths(config.input_image, config.document_type, config.output_dir)
    original_image = read_image(config.input_image)
    in_memory_start = time.perf_counter()
    rectified_image, rectification_metadata = rectify_document(original_image, config.document_type)
    ocr_image = preprocess_for_ocr(rectified_image)
    ocr_result, ocr_selection, timings = run_multi_version_ocr(
        rectified_image,
        ocr_image,
        language=config.ocr_language,
        config=config.tesseract_config,
    )
    parsing_start = time.perf_counter()
    parsed = parse_document(config.document_type, ocr_result, config.min_field_confidence)
    timings["tempo_parsing_total"] = time.perf_counter() - parsing_start
    timings["tempo_total_pipeline"] = time.perf_counter() - in_memory_start
    payload = build_result_payload(
        config=config,
        paths=paths,
        rectification_metadata=rectification_metadata,
        ocr_result=ocr_result,
        ocr_selection=ocr_selection,
        timings=timings,
        parsed=parsed,
    )
    save_pipeline_artifacts(paths, rectified_image, ocr_image, ocr_result, payload)
    return payload
