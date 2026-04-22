"""Batch evaluation helpers for OCR parsing outputs and ground truth text files."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import VALID_DOCUMENT_TYPES
from .ocr_engine import OCRLine, OCRResult
from .parsing import expected_fields, normalize_for_comparison, parse_document
from .utils.io import ensure_directory, write_json
from .utils.metrics import aggregate_timing_metrics, boolean_rate, mean


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
TEXT_FIELD_NAME = "__texto_completo__"
FIELD_ACCEPTANCE_SIMILARITY = 0.8
DOCUMENT_RELAXED_THRESHOLD = 0.8
GOOD_OCR_TEXT_SIMILARITY = 0.8


def is_valid_input_image(path: Path) -> bool:
    """Check whether a file is a valid document image input for batch inference.

    Args:
        path: Candidate file path.

    Returns:
        ``True`` for image files that are not masks.
    """

    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and not path.stem.endswith("_mask")


def infer_document_type_from_path(image_path: Path) -> str | None:
    """Infer the known document type from the image parent folder.

    Args:
        image_path: Dataset image path.

    Returns:
        Document type when the parent folder is one of the accepted types.
    """

    parent_name = image_path.parent.name
    return parent_name if parent_name in VALID_DOCUMENT_TYPES else None


def discover_batch_images(dataset_dir: Path) -> list[Path]:
    """Find valid images recursively in the batch dataset.

    Args:
        dataset_dir: Root dataset directory.

    Returns:
        Sorted list of image paths with valid document-type parent folders.
    """

    images = [
        path
        for path in dataset_dir.rglob("*")
        if is_valid_input_image(path) and infer_document_type_from_path(path) is not None
    ]
    return sorted(images, key=lambda item: item.as_posix())


def ground_truth_path_for_image(image_path: Path) -> Path | None:
    """Locate the ground truth ``.txt`` associated with an image.

    Args:
        image_path: Input image path.

    Returns:
        Matching text path, or ``None`` when missing.
    """

    candidate = image_path.with_suffix(".txt")
    return candidate if candidate.exists() else None


def extract_transcription_from_row(row: str) -> str | None:
    """Extract the transcription column from a dataset text row.

    Args:
        row: One row from the ground truth text file.

    Returns:
        Transcription value, or ``None`` for malformed rows.
    """

    stripped = row.strip()
    compact_header = stripped.lower().replace(" ", "")
    if not stripped or compact_header.startswith("x,y,width,height"):
        return None
    polygon_match = re.match(r"^\[.*\]\s*,\s*\[.*\]\s*,\s*-?\d+\s*,\s*-?\d+\s*,\s*(.+)$", stripped)
    if polygon_match:
        return polygon_match.group(1).strip()
    parts = stripped.split(",", 4)
    if len(parts) == 5:
        return parts[4].strip()
    return None


def read_ground_truth_transcriptions(text_path: Path) -> list[str]:
    """Read transcriptions from a dataset ground truth text file.

    Args:
        text_path: Ground truth ``.txt`` path.

    Returns:
        Ordered transcription strings.
    """

    content = text_path.read_text(encoding="utf-8-sig", errors="replace")
    transcriptions = []
    for row in content.splitlines():
        transcription = extract_transcription_from_row(row)
        if transcription:
            transcriptions.append(transcription)
    return transcriptions


def build_ground_truth_ocr_result(transcriptions: list[str]) -> OCRResult:
    """Build a synthetic OCR result from ground truth transcriptions.

    Args:
        transcriptions: Ground truth text lines.

    Returns:
        OCR-like result with perfect line confidence.
    """

    lines = [
        OCRLine(
            text=text,
            confidence=100.0,
            left=0,
            top=index * 20,
            width=max(len(text), 1) * 10,
            height=12,
        )
        for index, text in enumerate(transcriptions)
    ]
    return OCRResult(
        text="\n".join(transcriptions),
        mean_confidence=100.0 if transcriptions else 0.0,
        words=[],
        lines=lines,
    )


def parse_ground_truth_fields(document_type: str, transcriptions: list[str]) -> dict[str, Any]:
    """Parse structured fields from ground truth transcriptions.

    Args:
        document_type: Known document type.
        transcriptions: Ground truth text lines.

    Returns:
        Parsed ground truth fields when possible.
    """

    result = build_ground_truth_ocr_result(transcriptions)
    parsed = parse_document(document_type, result, min_confidence=0)
    return parsed.fields


def value_to_text(value: Any) -> str:
    """Convert scalar or list field values to comparable text.

    Args:
        value: Parsed field value.

    Returns:
        Comparable string representation.
    """

    if value is None:
        return ""
    if isinstance(value, list):
        return " | ".join(str(item) for item in value)
    return str(value)


def standardize_date_text(value: str) -> str:
    """Standardize a date-like string to ``dd/mm/yyyy`` when possible.

    Args:
        value: Raw date value.

    Returns:
        Standardized date text, or the original value when parsing is not possible.
    """

    match = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b", value)
    if not match:
        return value
    day, month, year = match.groups()
    year_int = int(year)
    if year_int < 100:
        year_int += 2000 if year_int <= 30 else 1900
    return f"{int(day):02d}/{int(month):02d}/{year_int:04d}"


def standardize_cpf_text(value: str) -> str:
    """Standardize CPF-like values to digits-only text.

    Args:
        value: Raw CPF candidate.

    Returns:
        Digits-only CPF when it has 11 digits, otherwise the original value.
    """

    digits = re.sub(r"\D", "", value)
    return digits if len(digits) == 11 else value


def normalize_value_for_evaluation(field_name: str, value: Any) -> str:
    """Normalize a field value before Levenshtein comparison.

    Args:
        field_name: Field name being compared.
        value: Field value.

    Returns:
        Normalized comparison string.
    """

    text = re.sub(r"\s+", " ", value_to_text(value)).strip()
    if field_name in {"cpf", "numero_cpf"}:
        text = standardize_cpf_text(text)
    elif "data" in field_name or field_name in {"validade", "primeira_habilitacao"}:
        text = standardize_date_text(text)
    return normalize_for_comparison(text)


def levenshtein_distance(left: str, right: str) -> int:
    """Compute Levenshtein edit distance between two strings.

    Args:
        left: First string.
        right: Second string.

    Returns:
        Minimum number of insertions, deletions, and substitutions.
    """

    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            insertion = current[right_index - 1] + 1
            deletion = previous[right_index] + 1
            substitution = previous[right_index - 1] + (left_char != right_char)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def normalized_similarity(left: str, right: str) -> float:
    """Calculate normalized string similarity from Levenshtein distance.

    Args:
        left: First string.
        right: Second string.

    Returns:
        Similarity from 0 to 1.
    """

    max_length = max(len(left), len(right))
    if max_length == 0:
        return 1.0
    distance = levenshtein_distance(left, right)
    return max(0.0, 1.0 - (distance / max_length))


def compare_field(
    image_path: Path,
    document_type: str,
    field_name: str,
    predicted_value: Any,
    ground_truth_value: Any,
    field_quality: dict[str, Any],
) -> dict[str, Any]:
    """Compare one parsed field against ground truth.

    Args:
        image_path: Input image path.
        document_type: Known document type.
        field_name: Field being compared.
        predicted_value: Parsed pipeline value.
        ground_truth_value: Parsed ground truth value.
        field_quality: Per-field confidence and parsing signals.

    Returns:
        Comparison record.
    """

    predicted_text = value_to_text(predicted_value)
    ground_truth_text = value_to_text(ground_truth_value)
    predicted_norm = normalize_value_for_evaluation(field_name, predicted_text)
    ground_truth_norm = normalize_value_for_evaluation(field_name, ground_truth_text)
    distance = levenshtein_distance(predicted_norm, ground_truth_norm)
    similarity = normalized_similarity(predicted_norm, ground_truth_norm)
    signals = field_quality.get("parsing_signals", {})
    return {
        "image_path": image_path.as_posix(),
        "document_type": document_type,
        "field_name": field_name,
        "predicted_value": predicted_value,
        "ground_truth_value": ground_truth_value,
        "field_found": bool(predicted_text),
        "ground_truth_available": bool(ground_truth_text),
        "exact_match": bool(ground_truth_norm) and predicted_norm == ground_truth_norm,
        "accepted_match": bool(ground_truth_norm) and similarity >= FIELD_ACCEPTANCE_SIMILARITY,
        "similarity": round(similarity, 4),
        "levenshtein_distance": distance,
        "ocr_confidence": field_quality.get("ocr_confidence"),
        "parsing_confidence": field_quality.get("parsing_confidence"),
        "regex_full_match": signals.get("regex_full_match"),
        "format_valid": signals.get("format_valid"),
        "cpf_has_11_digits": signals.get("cpf_has_11_digits"),
        "cpf_check_digits_valid": signals.get("cpf_check_digits_valid"),
        "date_plausible": signals.get("date_plausible"),
        "minimum_expected_field_found": signals.get("minimum_expected_field_found"),
        "conflict_with_other_candidates": signals.get("conflict_with_other_candidates"),
        "distance_to_label": signals.get("distance_to_label"),
    }


def compare_result_with_ground_truth(
    payload: dict[str, Any],
    image_path: Path,
    ground_truth_path: Path | None,
) -> dict[str, Any]:
    """Compare a pipeline result with its associated ground truth text.

    Args:
        payload: Pipeline output payload.
        image_path: Input image path.
        ground_truth_path: Ground truth text path, when available.

    Returns:
        Comparison bundle with ground truth fields and per-field records.
    """

    document_type = str(payload["tipo_documento"])
    transcriptions = (
        read_ground_truth_transcriptions(ground_truth_path)
        if ground_truth_path is not None
        else []
    )
    ground_truth_fields = parse_ground_truth_fields(document_type, transcriptions) if transcriptions else {}
    predicted_fields = payload.get("campos_extraidos", {})
    field_quality = payload.get("qualidade_por_campo", {})
    fields_to_compare = sorted(
        set(expected_fields(document_type)) | set(predicted_fields) | set(ground_truth_fields)
    )
    comparisons = [
        compare_field(
            image_path=image_path,
            document_type=document_type,
            field_name=field_name,
            predicted_value=predicted_fields.get(field_name),
            ground_truth_value=ground_truth_fields.get(field_name),
            field_quality=field_quality.get(field_name, {}),
        )
        for field_name in fields_to_compare
    ]
    ocr_text = payload.get("ocr", {}).get("texto_completo", "")
    gt_text = "\n".join(transcriptions)
    text_distance = levenshtein_distance(
        normalize_for_comparison(ocr_text),
        normalize_for_comparison(gt_text),
    )
    comparisons.append(
        {
            "image_path": image_path.as_posix(),
            "document_type": document_type,
            "field_name": TEXT_FIELD_NAME,
            "predicted_value": ocr_text,
            "ground_truth_value": gt_text,
            "field_found": bool(ocr_text),
            "ground_truth_available": bool(gt_text),
            "exact_match": bool(gt_text) and normalize_for_comparison(ocr_text) == normalize_for_comparison(gt_text),
            "accepted_match": bool(gt_text)
            and normalized_similarity(
                normalize_for_comparison(ocr_text),
                normalize_for_comparison(gt_text),
            )
            >= GOOD_OCR_TEXT_SIMILARITY,
            "similarity": round(
                normalized_similarity(
                    normalize_for_comparison(ocr_text),
                    normalize_for_comparison(gt_text),
                ),
                4,
            ),
            "levenshtein_distance": text_distance,
            "ocr_confidence": payload.get("ocr", {}).get("confianca_media"),
            "parsing_confidence": None,
            "regex_full_match": None,
            "format_valid": None,
            "cpf_has_11_digits": None,
            "cpf_check_digits_valid": None,
            "date_plausible": None,
            "minimum_expected_field_found": None,
            "conflict_with_other_candidates": None,
            "distance_to_label": None,
        }
    )
    return {
        "image_path": image_path.as_posix(),
        "ground_truth_path": ground_truth_path.as_posix() if ground_truth_path else None,
        "document_type": document_type,
        "ground_truth_fields": ground_truth_fields,
        "comparisons": comparisons,
    }


def group_by_field(comparisons: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group comparison records by field name.

    Args:
        comparisons: Flat comparison records.

    Returns:
        Mapping from field name to comparison records.
    """

    grouped: dict[str, list[dict[str, Any]]] = {}
    for comparison in comparisons:
        grouped.setdefault(comparison["field_name"], []).append(comparison)
    return grouped


def aggregate_batch_metrics(
    results: list[dict[str, Any]],
    comparison_bundles: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate final OCR and parsing quality metrics for a batch run.

    Args:
        results: Successful pipeline payloads.
        comparison_bundles: Ground truth comparison bundles.
        errors: Failed image processing records.

    Returns:
        Final metrics report.
    """

    comparisons = [
        comparison
        for bundle in comparison_bundles
        for comparison in bundle.get("comparisons", [])
    ]
    comparable_fields = [
        record
        for record in comparisons
        if record["field_name"] != TEXT_FIELD_NAME and record["ground_truth_available"]
    ]
    grouped = group_by_field(comparable_fields)
    metrics_by_field = {}
    for field_name, records in grouped.items():
        metrics_by_field[field_name] = {
            "total_comparacoes": len(records),
            "taxa_campos_encontrados": boolean_rate(records, "field_found"),
            "acuracia_exata": boolean_rate(records, "exact_match"),
            "acuracia_aceitavel": boolean_rate(records, "accepted_match"),
            "similaridade_media": mean([float(record["similarity"]) for record in records]),
            "distancia_media": mean([float(record["levenshtein_distance"]) for record in records]),
            "taxa_regex_valida": boolean_rate(records, "regex_full_match"),
            "taxa_formato_valido": boolean_rate(records, "format_valid"),
            "taxa_cpf_11_digitos": boolean_rate(records, "cpf_has_11_digits"),
            "taxa_cpf_valido": boolean_rate(records, "cpf_check_digits_valid"),
            "taxa_data_plausivel": boolean_rate(records, "date_plausible"),
            "confianca_ocr_media": mean(
                [float(record["ocr_confidence"]) for record in records if record.get("ocr_confidence") is not None]
            ),
            "confianca_parsing_media": mean(
                [
                    float(record["parsing_confidence"])
                    for record in records
                    if record.get("parsing_confidence") is not None
                ]
            ),
        }

    field_quality_records = [
        quality
        for result in results
        for quality in result.get("qualidade_por_campo", {}).values()
    ]
    parsing_signal_records = [
        quality.get("parsing_signals", {})
        for quality in field_quality_records
        if quality.get("parsing_signals")
    ]
    ocr_records = [result.get("ocr", {}) for result in results]
    text_records = [record for record in comparisons if record["field_name"] == TEXT_FIELD_NAME]
    timing_metrics = aggregate_timing_metrics(results)

    return {
        "total_imagens_processadas_com_sucesso": len(results),
        "total_imagens_com_erro": len(errors),
        "total_comparacoes_por_campo": len(comparable_fields),
        "acuracia_geral_exata": boolean_rate(comparable_fields, "exact_match"),
        "acuracia_geral_aceitavel": boolean_rate(comparable_fields, "accepted_match"),
        "similaridade_media_geral": mean([float(record["similarity"]) for record in comparable_fields]),
        "distancia_media_geral": mean([float(record["levenshtein_distance"]) for record in comparable_fields]),
        "taxa_campos_encontrados_geral": boolean_rate(comparable_fields, "field_found"),
        "taxa_regex_valida_geral": boolean_rate(comparable_fields, "regex_full_match"),
        "taxa_formato_valido_geral": boolean_rate(comparable_fields, "format_valid"),
        "taxa_cpf_11_digitos": boolean_rate(comparable_fields, "cpf_has_11_digits"),
        "taxa_cpf_valido": boolean_rate(comparable_fields, "cpf_check_digits_valid"),
        "taxa_data_plausivel": boolean_rate(comparable_fields, "date_plausible"),
        "metricas_por_campo": metrics_by_field,
        "metricas_agregadas_parsing": {
            "confianca_parsing_media": mean(
                [
                    float(quality["parsing_confidence"])
                    for quality in field_quality_records
                    if quality.get("parsing_confidence") is not None
                ]
            ),
            "taxa_conflito_candidatos": boolean_rate(
                parsing_signal_records,
                "conflict_with_other_candidates",
            ),
            "taxa_campo_minimo_encontrado": boolean_rate(
                parsing_signal_records,
                "minimum_expected_field_found",
            ),
            "distancia_media_ate_rotulo": mean(
                [
                    abs(float(signals["distance_to_label"]))
                    for signals in parsing_signal_records
                    if signals.get("distance_to_label") is not None
                ]
            ),
        },
        "metricas_agregadas_ocr": {
            "confianca_ocr_media": mean(
                [float(record["confianca_media"]) for record in ocr_records if record.get("confianca_media") is not None]
            ),
            "media_linhas_ocr": mean([float(record.get("quantidade_linhas", 0)) for record in ocr_records]),
            "media_palavras_ocr": mean([float(record.get("quantidade_palavras", 0)) for record in ocr_records]),
            "similaridade_media_texto_completo": mean(
                [float(record["similarity"]) for record in text_records if record["ground_truth_available"]]
            ),
        },
        "metricas_tempos": timing_metrics,
        "erros": errors,
    }


def comparison_fields_for_document(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    """Return comparable field records for one document bundle.

    Args:
        bundle: Ground truth comparison bundle.

    Returns:
        Field-level comparison records, excluding full-text OCR rows.
    """

    return [
        record
        for record in bundle.get("comparisons", [])
        if record["field_name"] != TEXT_FIELD_NAME and record.get("ground_truth_available")
    ]


def text_comparison_for_document(bundle: dict[str, Any]) -> dict[str, Any] | None:
    """Return the full-text OCR comparison record for one document.

    Args:
        bundle: Ground truth comparison bundle.

    Returns:
        Full-text comparison record, or ``None`` when absent.
    """

    for record in bundle.get("comparisons", []):
        if record["field_name"] == TEXT_FIELD_NAME:
            return record
    return None


def summarize_document_comparison(bundle: dict[str, Any]) -> dict[str, Any]:
    """Summarize quality metrics for one processed document.

    Args:
        bundle: Ground truth comparison bundle.

    Returns:
        Document-level metric record.
    """

    fields = comparison_fields_for_document(bundle)
    text_record = text_comparison_for_document(bundle)
    total_fields = len(fields)
    exact_fields = sum(1 for record in fields if record.get("exact_match"))
    accepted_fields = sum(1 for record in fields if record.get("accepted_match"))
    found_fields = sum(1 for record in fields if record.get("field_found"))
    field_accuracy_strict = exact_fields / total_fields if total_fields else 0.0
    field_accuracy_relaxed = accepted_fields / total_fields if total_fields else 0.0
    strict = total_fields > 0 and exact_fields == total_fields
    relaxed = total_fields > 0 and field_accuracy_relaxed >= DOCUMENT_RELAXED_THRESHOLD
    ocr_good = bool(text_record and text_record.get("accepted_match"))
    parsing_error = ocr_good and not relaxed
    severe_failure = total_fields == 0 or field_accuracy_relaxed < 0.5 or found_fields == 0
    return {
        "image_path": bundle["image_path"],
        "document_type": bundle["document_type"],
        "ground_truth_path": bundle.get("ground_truth_path"),
        "total_fields": total_fields,
        "fields_exact": exact_fields,
        "fields_accepted": accepted_fields,
        "fields_found": found_fields,
        "document_accuracy_strict": strict,
        "document_accuracy_relaxed": relaxed,
        "field_accuracy_strict": round(field_accuracy_strict, 4),
        "field_accuracy_relaxed": round(field_accuracy_relaxed, 4),
        "field_found_rate": round(found_fields / total_fields, 4) if total_fields else None,
        "mean_similarity": mean([float(record["similarity"]) for record in fields]),
        "mean_distance": mean([float(record["levenshtein_distance"]) for record in fields]),
        "ocr_text_similarity": text_record.get("similarity") if text_record else None,
        "ocr_is_good": ocr_good,
        "parsing_error_given_good_ocr": parsing_error,
        "end_to_end_error": not relaxed,
        "severe_failure": severe_failure,
    }


def rate_from_document_records(records: list[dict[str, Any]], key: str) -> float | None:
    """Calculate a true-rate over document metric records.

    Args:
        records: Document metric records.
        key: Boolean key.

    Returns:
        True rate, or ``None`` when no records exist.
    """

    if not records:
        return None
    return round(sum(1 for record in records if record.get(key)) / len(records), 4)


def summarize_document_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize document-level metrics for a group.

    Args:
        records: Document metric records.

    Returns:
        Aggregated document metrics.
    """

    return {
        "documents": len(records),
        "document_accuracy_strict": rate_from_document_records(records, "document_accuracy_strict"),
        "document_accuracy_relaxed": rate_from_document_records(records, "document_accuracy_relaxed"),
        "average_document_field_accuracy": mean(
            [float(record["field_accuracy_relaxed"]) for record in records]
        ),
        "mean_similarity": mean(
            [float(record["mean_similarity"]) for record in records if record.get("mean_similarity") is not None]
        ),
        "mean_distance": mean(
            [float(record["mean_distance"]) for record in records if record.get("mean_distance") is not None]
        ),
        "perfect_document_rate": rate_from_document_records(records, "document_accuracy_strict"),
        "acceptable_document_rate": rate_from_document_records(records, "document_accuracy_relaxed"),
        "severe_failure_rate": rate_from_document_records(records, "severe_failure"),
        "field_found_rate": mean(
            [float(record["field_found_rate"]) for record in records if record.get("field_found_rate") is not None]
        ),
        "ocr_error_rate": 1.0 - rate_from_document_records(records, "ocr_is_good")
        if rate_from_document_records(records, "ocr_is_good") is not None
        else None,
        "parsing_error_rate_given_good_ocr": rate_from_document_records(
            [record for record in records if record.get("ocr_is_good")],
            "parsing_error_given_good_ocr",
        ),
        "end_to_end_error_rate": rate_from_document_records(records, "end_to_end_error"),
    }


def build_executive_summary(
    comparison_bundles: list[dict[str, Any]],
    metrics: dict[str, Any],
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build executive-level quality metrics for the OCR parsing pipeline.

    Args:
        comparison_bundles: Ground truth comparison bundles.
        metrics: Existing aggregated batch metrics.
        errors: Failed image records.

    Returns:
        Executive summary dictionary.
    """

    document_records = [summarize_document_comparison(bundle) for bundle in comparison_bundles]
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in document_records:
        by_type[record["document_type"]].append(record)

    return {
        "thresholds": {
            "field_acceptance_similarity": FIELD_ACCEPTANCE_SIMILARITY,
            "document_relaxed_threshold": DOCUMENT_RELAXED_THRESHOLD,
            "good_ocr_text_similarity": GOOD_OCR_TEXT_SIMILARITY,
        },
        "overall": {
            "overall_document_accuracy_strict": rate_from_document_records(
                document_records,
                "document_accuracy_strict",
            ),
            "overall_document_accuracy_relaxed": rate_from_document_records(
                document_records,
                "document_accuracy_relaxed",
            ),
            "overall_field_accuracy": metrics.get("acuracia_geral_aceitavel"),
            "overall_exact_field_accuracy": metrics.get("acuracia_geral_exata"),
            "mean_similarity": metrics.get("similaridade_media_geral"),
            "mean_distance": metrics.get("distancia_media_geral"),
            "processed_documents": len(document_records),
            "failed_documents": len(errors),
        },
        "timing": metrics.get("metricas_tempos", {}),
        "rates": {
            "perfect_document_rate": rate_from_document_records(
                document_records,
                "document_accuracy_strict",
            ),
            "acceptable_document_rate": rate_from_document_records(
                document_records,
                "document_accuracy_relaxed",
            ),
            "severe_failure_rate": rate_from_document_records(document_records, "severe_failure"),
            "field_found_rate": metrics.get("taxa_campos_encontrados_geral"),
            "valid_parsing_rate": metrics.get("taxa_formato_valido_geral"),
        },
        "error_split": {
            "ocr_error_rate": summarize_document_group(document_records).get("ocr_error_rate"),
            "parsing_error_rate_given_good_ocr": summarize_document_group(document_records).get(
                "parsing_error_rate_given_good_ocr"
            ),
            "end_to_end_error_rate": summarize_document_group(document_records).get(
                "end_to_end_error_rate"
            ),
        },
        "by_document_type": {
            document_type: summarize_document_group(by_type.get(document_type, []))
            for document_type in VALID_DOCUMENT_TYPES
        },
        "documents": document_records,
    }


def find_result_for_image(results: list[dict[str, Any]], image_path: str) -> dict[str, Any] | None:
    """Find a pipeline payload by image path.

    Args:
        results: Pipeline payloads.
        image_path: Image path string from comparison bundle.

    Returns:
        Matching payload, or ``None``.
    """

    normalized = Path(image_path).as_posix()
    for result in results:
        if Path(str(result.get("imagem_entrada", ""))).as_posix() == normalized:
            return result
    return None


def classify_predominant_error(rectification_error: bool, ocr_error: bool, parsing_error: bool) -> str:
    """Classify the predominant error source for one document.

    Args:
        rectification_error: Whether rectification fell back or failed to find a contour.
        ocr_error: Whether OCR text similarity is below threshold.
        parsing_error: Whether fields failed despite acceptable OCR.

    Returns:
        Predominant error label.
    """

    if rectification_error and ocr_error:
        return "retificacao_imagem"
    if ocr_error:
        return "ocr"
    if parsing_error:
        return "parsing"
    if rectification_error:
        return "retificacao_sem_impacto_final_claro"
    return "sem_erro_relevante"


def build_detailed_error_report(
    results: list[dict[str, Any]],
    comparison_bundles: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a detailed error analysis report per document.

    Args:
        results: Successful pipeline payloads.
        comparison_bundles: Ground truth comparison bundles.
        errors: Failed image records.

    Returns:
        Detailed error report.
    """

    documents = []
    for bundle in comparison_bundles:
        summary = summarize_document_comparison(bundle)
        result = find_result_for_image(results, bundle["image_path"])
        rectification = result.get("retificacao", {}) if result else {}
        rectification_error = (
            not bool(rectification.get("document_contour_found", True))
            or bool(rectification.get("fallback"))
        )
        text_record = text_comparison_for_document(bundle)
        ocr_error = not bool(text_record and text_record.get("accepted_match"))
        parsing_error = bool(summary["parsing_error_given_good_ocr"])
        problem_fields = [
            {
                "field_name": record["field_name"],
                "predicted_value": record.get("predicted_value"),
                "ground_truth_value": record.get("ground_truth_value"),
                "similarity": record.get("similarity"),
                "levenshtein_distance": record.get("levenshtein_distance"),
                "field_found": record.get("field_found"),
                "ocr_confidence": record.get("ocr_confidence"),
                "parsing_confidence": record.get("parsing_confidence"),
            }
            for record in comparison_fields_for_document(bundle)
            if not record.get("accepted_match")
        ]
        documents.append(
            {
                "image_path": bundle["image_path"],
                "document_type": bundle["document_type"],
                "rectification_error": rectification_error,
                "ocr_error": ocr_error,
                "parsing_error": parsing_error,
                "problematic_fields": problem_fields,
                "predominant_error_type": classify_predominant_error(
                    rectification_error,
                    ocr_error,
                    parsing_error,
                ),
                "document_metrics": summary,
                "ocr_selection": result.get("ocr", {}).get("candidatos") if result else None,
                "selected_ocr_version": result.get("ocr", {}).get("versao_imagem_escolhida")
                if result
                else None,
                "tempos": result.get("tempos") if result else None,
            }
        )

    failed_documents = [
        {
            "image_path": error.get("image_path"),
            "document_type": error.get("document_type"),
            "rectification_error": None,
            "ocr_error": True,
            "parsing_error": None,
            "problematic_fields": [],
            "predominant_error_type": "falha_execucao",
            "error": error.get("error"),
        }
        for error in errors
    ]
    return {
        "thresholds": {
            "field_acceptance_similarity": FIELD_ACCEPTANCE_SIMILARITY,
            "good_ocr_text_similarity": GOOD_OCR_TEXT_SIMILARITY,
        },
        "documents": documents,
        "failed_documents": failed_documents,
    }


def write_executive_markdown(report_path: Path, summary: dict[str, Any]) -> Path:
    """Write a compact executive metrics report in Markdown.

    Args:
        report_path: Destination Markdown path.
        summary: Executive summary metrics.

    Returns:
        Destination path.
    """

    ensure_directory(report_path.parent)
    overall = summary["overall"]
    rates = summary["rates"]
    errors = summary["error_split"]
    timing = summary.get("timing", {})
    lines = [
        "# Resumo executivo OCR + Parsing",
        "",
        f"- Document accuracy strict: {overall['overall_document_accuracy_strict']}",
        f"- Document accuracy relaxed: {overall['overall_document_accuracy_relaxed']}",
        f"- Overall field accuracy: {overall['overall_field_accuracy']}",
        f"- Similaridade media: {overall['mean_similarity']}",
        f"- Distancia media: {overall['mean_distance']}",
        f"- Taxa documentos perfeitos: {rates['perfect_document_rate']}",
        f"- Taxa documentos aceitaveis: {rates['acceptable_document_rate']}",
        f"- Taxa falha grave: {rates['severe_failure_rate']}",
        f"- OCR error rate: {errors['ocr_error_rate']}",
        f"- Parsing error rate given good OCR: {errors['parsing_error_rate_given_good_ocr']}",
        f"- End-to-end error rate: {errors['end_to_end_error_rate']}",
        f"- Tempo medio OCR retificada: {timing.get('tempo_ocr_retificada', {}).get('media')}",
        f"- Tempo medio OCR pre_ocr: {timing.get('tempo_ocr_pre_ocr', {}).get('media')}",
        f"- Tempo medio parsing: {timing.get('tempo_parsing_total', {}).get('media')}",
        f"- Tempo medio total por documento: {timing.get('tempo_total_pipeline', {}).get('media')}",
        "",
        "## Por tipo de documento",
        "",
    ]
    for document_type, metrics in summary["by_document_type"].items():
        lines.append(f"### {document_type}")
        lines.append(f"- documentos: {metrics['documents']}")
        lines.append(f"- strict: {metrics['document_accuracy_strict']}")
        lines.append(f"- relaxed: {metrics['document_accuracy_relaxed']}")
        lines.append(f"- acerto medio por documento: {metrics['average_document_field_accuracy']}")
        lines.append(f"- falha grave: {metrics['severe_failure_rate']}")
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def write_comparisons_csv(csv_path: Path, comparison_bundles: list[dict[str, Any]]) -> Path:
    """Write flat comparison records to CSV.

    Args:
        csv_path: Destination CSV path.
        comparison_bundles: Comparison bundles from all images.

    Returns:
        Destination path.
    """

    ensure_directory(csv_path.parent)
    records = [
        comparison
        for bundle in comparison_bundles
        for comparison in bundle.get("comparisons", [])
    ]
    fieldnames = sorted({key for record in records for key in record}) if records else []
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                for key, value in record.items()
            }
            writer.writerow(row)
    return csv_path


def write_metrics_markdown(report_path: Path, metrics: dict[str, Any]) -> Path:
    """Write a concise human-readable metrics report.

    Args:
        report_path: Destination Markdown path.
        metrics: Aggregated metrics.

    Returns:
        Destination path.
    """

    ensure_directory(report_path.parent)
    timing = metrics.get("metricas_tempos", {})
    lines = [
        "# Relatorio OCR + Parsing",
        "",
        f"- Imagens com sucesso: {metrics['total_imagens_processadas_com_sucesso']}",
        f"- Imagens com erro: {metrics['total_imagens_com_erro']}",
        f"- Acuracia geral exata: {metrics['acuracia_geral_exata']}",
        f"- Similaridade media geral: {metrics['similaridade_media_geral']}",
        f"- Distancia media geral: {metrics['distancia_media_geral']}",
        f"- Taxa de campos encontrados: {metrics['taxa_campos_encontrados_geral']}",
        f"- Taxa regex valida: {metrics['taxa_regex_valida_geral']}",
        f"- Taxa formato valido: {metrics['taxa_formato_valido_geral']}",
        f"- Taxa CPF 11 digitos: {metrics['taxa_cpf_11_digitos']}",
        f"- Taxa CPF valido: {metrics['taxa_cpf_valido']}",
        f"- Taxa data plausivel: {metrics['taxa_data_plausivel']}",
        f"- Tempo medio OCR retificada: {timing.get('tempo_ocr_retificada', {}).get('media')}",
        f"- Tempo medio OCR pre_ocr: {timing.get('tempo_ocr_pre_ocr', {}).get('media')}",
        f"- Tempo medio parsing: {timing.get('tempo_parsing_total', {}).get('media')}",
        f"- Tempo medio total por documento: {timing.get('tempo_total_pipeline', {}).get('media')}",
        "",
        "## Por campo",
        "",
    ]
    for field_name, field_metrics in metrics.get("metricas_por_campo", {}).items():
        lines.append(f"### {field_name}")
        for key, value in field_metrics.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def save_batch_outputs(
    output_dir: Path,
    results: list[dict[str, Any]],
    comparison_bundles: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> dict[str, str]:
    """Persist consolidated batch outputs.

    Args:
        output_dir: Batch output directory.
        results: Successful pipeline payloads.
        comparison_bundles: Ground truth comparisons.
        metrics: Aggregated metrics.

    Returns:
        Mapping of artifact names to paths.
    """

    ensure_directory(output_dir)
    executive_summary = build_executive_summary(comparison_bundles, metrics, metrics.get("erros", []))
    detailed_errors = build_detailed_error_report(results, comparison_bundles, metrics.get("erros", []))
    result_path = write_json(
        output_dir / "resultados_consolidados.json",
        {
            "resultados": results,
            "metricas_tempos": metrics.get("metricas_tempos", aggregate_timing_metrics(results)),
        },
    )
    comparison_json_path = write_json(
        output_dir / "comparacoes_ground_truth.json",
        {"comparacoes": comparison_bundles},
    )
    comparison_csv_path = write_comparisons_csv(
        output_dir / "comparacoes_ground_truth.csv",
        comparison_bundles,
    )
    metrics_path = write_json(output_dir / "relatorio_metricas.json", metrics)
    markdown_path = write_metrics_markdown(output_dir / "relatorio_metricas.md", metrics)
    executive_path = write_json(output_dir / "resumo_executivo_metricas.json", executive_summary)
    executive_markdown_path = write_executive_markdown(
        output_dir / "resumo_executivo_metricas.md",
        executive_summary,
    )
    detailed_errors_path = write_json(output_dir / "erros_analise_detalhada.json", detailed_errors)
    return {
        "resultados_consolidados": result_path.as_posix(),
        "comparacoes_ground_truth_json": comparison_json_path.as_posix(),
        "comparacoes_ground_truth_csv": comparison_csv_path.as_posix(),
        "relatorio_metricas_json": metrics_path.as_posix(),
        "relatorio_metricas_md": markdown_path.as_posix(),
        "resumo_executivo_metricas_json": executive_path.as_posix(),
        "resumo_executivo_metricas_md": executive_markdown_path.as_posix(),
        "erros_analise_detalhada_json": detailed_errors_path.as_posix(),
    }
