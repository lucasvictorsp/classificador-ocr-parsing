"""Generic metric and timing helpers used by batch evaluation."""

from __future__ import annotations

from typing import Any


def mean(values: list[float]) -> float | None:
    """Calculate mean for a list of values.

    Args:
        values: Numeric values.

    Returns:
        Mean value, or ``None`` for an empty list.
    """

    return round(sum(values) / len(values), 4) if values else None


def boolean_rate(records: list[dict[str, Any]], key: str) -> float | None:
    """Calculate the rate of true values for a nullable boolean key.

    Args:
        records: Metric records.
        key: Boolean key.

    Returns:
        True rate, or ``None`` when the key is not applicable.
    """

    applicable = [record[key] for record in records if record.get(key) is not None]
    if not applicable:
        return None
    return round(sum(1 for value in applicable if value) / len(applicable), 4)


def numeric_stats(values: list[float]) -> dict[str, float | int | None]:
    """Calculate mean, standard deviation, minimum, and maximum.

    Args:
        values: Numeric values.

    Returns:
        Dictionary with mean, std, min, max, and count.
    """

    if not values:
        return {
            "media": None,
            "desvio_padrao": None,
            "minimo": None,
            "maximo": None,
            "quantidade": 0,
        }
    avg = sum(values) / len(values)
    variance = sum((value - avg) ** 2 for value in values) / len(values)
    return {
        "media": round(avg, 6),
        "desvio_padrao": round(variance**0.5, 6),
        "minimo": round(min(values), 6),
        "maximo": round(max(values), 6),
        "quantidade": len(values),
    }


def timing_values(results: list[dict[str, Any]], key: str) -> list[float]:
    """Extract timing values from pipeline payloads.

    Args:
        results: Pipeline payloads.
        key: Timing key under ``tempos``.

    Returns:
        Timing values in seconds.
    """

    values: list[float] = []
    for result in results:
        value = result.get("tempos", {}).get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def aggregate_timing_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate timing metrics from individual image payloads.

    Args:
        results: Pipeline payloads.

    Returns:
        Timing statistics for OCR, parsing, and total in-memory processing.
    """

    return {
        "unidade": "segundos",
        "observacao": "Nao inclui salvamento de imagens, escrita de JSON ou outros artefatos em disco.",
        "tempo_ocr_retificada": numeric_stats(timing_values(results, "tempo_ocr_retificada")),
        "tempo_ocr_pre_ocr": numeric_stats(timing_values(results, "tempo_ocr_pre_ocr")),
        "tempo_parsing_total": numeric_stats(timing_values(results, "tempo_parsing_total")),
        "tempo_total_pipeline": numeric_stats(timing_values(results, "tempo_total_pipeline")),
    }
