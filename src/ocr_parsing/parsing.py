"""Rule-based parsing for known Brazilian document types."""

from __future__ import annotations

from datetime import date
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable

from .ocr_engine import OCRLine, OCRResult


CPF_PATTERN = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")
DATE_PATTERN = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
RG_PATTERN = re.compile(r"\b\d{1,2}\.?\d{3}\.?\d{3}-?[\dXx]?\b")
LONG_DIGIT_PATTERN = re.compile(r"\b\d{8,12}\b")
SITE_PATTERN = re.compile(r"\b(?:WWW\.)?[A-Z0-9.-]+\.(?:COM|BR)(?:\.[A-Z]{2})?\b")


@dataclass(frozen=True)
class ParsedDocument:
    """Parsed output before final pipeline serialization.

    Attributes:
        document_type: Known document type used for parsing.
        fields: Extracted fields.
        confidence_by_field: Estimated OCR confidence for each extracted field.
        parsing_confidence_by_field: Rule-based parsing confidence for each extracted field.
        parsing_signals_by_field: Auditable parsing signals for each extracted field.
        warnings: Missing-field and low-confidence warnings.
    """

    document_type: str
    fields: dict[str, Any]
    confidence_by_field: dict[str, float | None]
    parsing_confidence_by_field: dict[str, float]
    parsing_signals_by_field: dict[str, dict[str, Any]]
    warnings: list[str]


def strip_accents(text: str) -> str:
    """Remove accents from text for label matching.

    Args:
        text: Input text.

    Returns:
        ASCII-like text without combining marks.
    """

    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def normalize_text(text: str) -> str:
    """Normalize OCR text for robust rule matching.

    Args:
        text: Raw OCR text.

    Returns:
        Uppercase accent-free text with compact whitespace.
    """

    without_accents = strip_accents(text)
    compact = re.sub(r"\s+", " ", without_accents)
    return compact.upper().strip()


def normalize_numeric_noise(text: str) -> str:
    """Replace common OCR confusions inside numeric fields.

    Args:
        text: Raw OCR text.

    Returns:
        Text with common letter/digit substitutions for numeric matching.
    """

    table = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "B": "8"})
    return text.translate(table)


def only_digits(text: str) -> str:
    """Keep only numeric characters.

    Args:
        text: Text containing digits and separators.

    Returns:
        Digits-only string.
    """

    return re.sub(r"\D", "", text)


def normalize_for_comparison(text: str) -> str:
    """Normalize text for exact-ish comparison and validation.

    Args:
        text: Raw text value.

    Returns:
        Uppercase accent-free value with only alphanumeric characters.
    """

    normalized = normalize_text(text)
    return re.sub(r"[^A-Z0-9]", "", normalized)


def format_cpf(value: str) -> str:
    """Format a CPF string as ``000.000.000-00`` when possible.

    Args:
        value: Raw CPF candidate.

    Returns:
        Formatted CPF when it has 11 digits; otherwise the original stripped value.
    """

    digits = only_digits(value)
    if len(digits) != 11:
        return value.strip()
    return f"{digits[:3]}.{digits[3:6]}.{digits[6:9]}-{digits[9:]}"


def is_valid_cpf_checksum(value: str) -> bool:
    """Validate CPF check digits.

    Args:
        value: CPF candidate.

    Returns:
        ``True`` when the CPF has 11 digits and valid check digits.
    """

    digits = only_digits(value)
    if len(digits) != 11 or len(set(digits)) == 1:
        return False
    first_sum = sum(int(digit) * weight for digit, weight in zip(digits[:9], range(10, 1, -1)))
    first_digit = (first_sum * 10) % 11
    first_digit = 0 if first_digit == 10 else first_digit
    second_sum = sum(int(digit) * weight for digit, weight in zip(digits[:10], range(11, 1, -1)))
    second_digit = (second_sum * 10) % 11
    second_digit = 0 if second_digit == 10 else second_digit
    return digits[-2:] == f"{first_digit}{second_digit}"


def parse_date_value(value: str) -> date | None:
    """Parse a Brazilian date candidate.

    Args:
        value: Date candidate in ``dd/mm/yyyy`` or similar format.

    Returns:
        Parsed date, or ``None`` when invalid.
    """

    match = DATE_PATTERN.search(normalize_numeric_noise(value))
    if not match:
        return None
    day_text, month_text, year_text = re.split(r"[/-]", match.group(0))
    year = int(year_text)
    if year < 100:
        year += 2000 if year <= 30 else 1900
    try:
        return date(year, int(month_text), int(day_text))
    except ValueError:
        return None


def is_plausible_date(value: str, field_name: str | None = None) -> bool:
    """Check whether a date is valid and plausible for a document field.

    Args:
        value: Date candidate.
        field_name: Optional field name used for stricter birth-date checks.

    Returns:
        ``True`` when the date exists and falls in a reasonable interval.
    """

    parsed = parse_date_value(value)
    if parsed is None:
        return False
    today = date.today()
    if parsed.year < 1900 or parsed.year > today.year + 20:
        return False
    if field_name and "nascimento" in field_name and parsed > today:
        return False
    return True


def clean_value(text: str) -> str:
    """Clean common OCR punctuation around field values.

    Args:
        text: Raw field candidate.

    Returns:
        Cleaned value.
    """

    return re.sub(r"\s+", " ", text.replace("|", " ")).strip(" :;,-\t")


def line_dicts(ocr_result: OCRResult) -> list[dict[str, Any]]:
    """Convert OCR line records to dictionaries enriched with normalized text.

    Args:
        ocr_result: Structured OCR result.

    Returns:
        Ordered line dictionaries.
    """

    return [
        {
            "text": line.text,
            "norm": normalize_text(line.text),
            "confidence": line.confidence,
            "line": line,
        }
        for line in ocr_result.lines
        if line.text.strip()
    ]


def find_regex(
    text: str,
    pattern: re.Pattern[str],
    formatter: Callable[[str], str] | None = None,
) -> str | None:
    """Find and optionally format the first regex match.

    Args:
        text: Text to search.
        pattern: Compiled regex pattern.
        formatter: Optional function applied to the match.

    Returns:
        First matched value, or ``None``.
    """

    match = pattern.search(text)
    if not match:
        return None
    value = match.group(0)
    return formatter(value) if formatter else value


def confidence_for_value(value: str | None, lines: list[dict[str, Any]]) -> float | None:
    """Estimate field confidence from the OCR line containing the value.

    Args:
        value: Extracted value.
        lines: Normalized OCR line dictionaries.

    Returns:
        Mean confidence for a matching line, or ``None`` when unavailable.
    """

    if not value:
        return None
    normalized_value = normalize_text(value)
    value_digits = only_digits(value)
    for line in lines:
        line_norm = line["norm"]
        if normalized_value and normalized_value in line_norm:
            return float(line["confidence"])
        if value_digits and value_digits in only_digits(line["text"]):
            return float(line["confidence"])
    return None


def line_index_for_value(value: str | None, lines: list[dict[str, Any]]) -> int | None:
    """Find the OCR line index that most likely contains a field value.

    Args:
        value: Extracted field value.
        lines: OCR line dictionaries.

    Returns:
        Matching line index, or ``None``.
    """

    if not value:
        return None
    normalized_value = normalize_text(value)
    compact_value = normalize_for_comparison(value)
    value_digits = only_digits(value)
    for index, line in enumerate(lines):
        line_norm = line["norm"]
        line_compact = normalize_for_comparison(line["text"])
        if normalized_value and normalized_value in line_norm:
            return index
        if compact_value and compact_value in line_compact:
            return index
        if value_digits and value_digits in only_digits(line["text"]):
            return index
    return None


def is_label_line(normalized_line: str) -> bool:
    """Check whether a line looks like a document label rather than a value.

    Args:
        normalized_line: Normalized OCR line.

    Returns:
        ``True`` when the line mostly contains expected labels.
    """

    labels = (
        "NOME",
        "CPF",
        "NASCIMENTO",
        "FILIACAO",
        "VALIDADE",
        "REGISTRO",
        "EXPEDICAO",
        "NATURALIDADE",
        "DOC",
        "ORGAO",
        "ASSINATURA",
        "TERRITORIO",
        "MINISTERIO",
        "SECRETARIA",
        "REPUBLICA",
        "CARTEIRA",
    )
    return any(label in normalized_line for label in labels) and len(normalized_line.split()) <= 5


def find_value_after_label(
    lines: list[dict[str, Any]],
    labels: tuple[str, ...],
    max_ahead: int = 4,
) -> str | None:
    """Find a value near a known label in OCR lines.

    Args:
        lines: OCR line dictionaries.
        labels: Normalized label fragments to search.
        max_ahead: Number of nearby lines inspected around the label.

    Returns:
        Cleaned value, or ``None``.
    """

    for index, line in enumerate(lines):
        if not any(label in line["norm"] for label in labels):
            continue
        removed = line["text"]
        for label in labels:
            removed = re.sub(label, "", normalize_text(removed), flags=re.IGNORECASE)
        inline_value = clean_value(removed)
        if inline_value and not is_label_line(normalize_text(inline_value)):
            return inline_value
        for next_line in lines[index + 1 : index + max_ahead + 1]:
            candidate = clean_value(next_line["text"])
            if candidate and not is_label_line(next_line["norm"]):
                return candidate
        previous_start = max(0, index - max_ahead)
        for previous_line in reversed(lines[previous_start:index]):
            candidate = clean_value(previous_line["text"])
            if candidate and not is_label_line(previous_line["norm"]):
                return candidate
    return None


def find_date_after_label(lines: list[dict[str, Any]], labels: tuple[str, ...]) -> str | None:
    """Find a date candidate near a label.

    Args:
        lines: OCR line dictionaries.
        labels: Normalized date label fragments.

    Returns:
        Date string, or ``None``.
    """

    value = find_value_after_label(lines, labels)
    if value:
        return find_regex(normalize_numeric_noise(value), DATE_PATTERN)
    for index, line in enumerate(lines):
        if any(label in line["norm"] for label in labels):
            window_start = max(0, index - 3)
            window = " ".join(item["text"] for item in lines[window_start : index + 5])
            date = find_regex(normalize_numeric_noise(window), DATE_PATTERN)
            if date:
                return date
    return None


def expected_labels_for_field(document_type: str, field_name: str) -> tuple[str, ...]:
    """Return normalized labels expected near a parsed field.

    Args:
        document_type: Known document type.
        field_name: Parsed field name.

    Returns:
        Tuple of normalized label fragments.
    """

    mapping: dict[str, dict[str, tuple[str, ...]]] = {
        "CPF_Frente": {
            "numero_cpf": ("NUMERO DE INSCRICAO", "CPF"),
            "nome": ("NOME",),
            "data_nascimento": ("NASCIMENTO",),
        },
        "CPF_Verso": {
            "emissao": ("EMISSAO",),
            "site": ("WWW", "CORREIOS"),
        },
        "CNH_Frente": {
            "nome": ("NOME",),
            "cpf": ("CPF",),
            "data_nascimento": ("DATA NASCIMENTO", "NASCIMENTO"),
            "documento_identidade": ("DOC IDENTIDADE", "IDENTIDADE"),
            "filiacao": ("FILIACAO",),
            "validade": ("VALIDADE",),
            "primeira_habilitacao": ("1A HABILITACAO", "1 HABILITACAO"),
            "numero_registro": ("N REGISTRO", "REGISTRO"),
            "categoria": ("CAT HAB",),
        },
        "CNH_Verso": {
            "local": ("LOCAL",),
            "data_emissao": ("DATA EMISSAO", "EMISSAO"),
            "numero_registro": ("REGISTRO",),
            "observacoes": ("OBSERVACOES",),
        },
        "RG_Frente": {
            "orgao_emissor": ("SECRETARIA", "INSTITUTO"),
            "observacoes": ("NAO DOADOR", "DOADOR"),
        },
        "RG_Verso": {
            "registro_geral": ("REGISTRO GERAL", "REGISTRO"),
            "data_expedicao": ("DATA EXPEDICAO", "EXPEDICAO"),
            "nome": ("NOME",),
            "filiacao": ("FILIACAO",),
            "naturalidade": ("NATURALIDADE",),
            "doc_origem": ("DOC ORIGEM",),
            "cpf": ("CPF",),
            "data_nascimento": ("DATA DE NASCIMENTO", "NASCIMENTO"),
        },
    }
    return mapping.get(document_type, {}).get(field_name, ())


def regex_for_field(field_name: str) -> re.Pattern[str] | None:
    """Return a validation regex applicable to a parsed field.

    Args:
        field_name: Parsed field name.

    Returns:
        Compiled regex pattern, or ``None`` for free-text fields.
    """

    if field_name in {"cpf", "numero_cpf"}:
        return CPF_PATTERN
    if "data" in field_name or field_name in {"validade", "primeira_habilitacao"}:
        return DATE_PATTERN
    if field_name == "registro_geral":
        return RG_PATTERN
    if field_name == "numero_registro":
        return LONG_DIGIT_PATTERN
    if field_name == "site":
        return SITE_PATTERN
    return None


def find_label_index(lines: list[dict[str, Any]], labels: tuple[str, ...]) -> int | None:
    """Find the first OCR line containing any expected label.

    Args:
        lines: OCR line dictionaries.
        labels: Normalized label fragments.

    Returns:
        Label line index, or ``None``.
    """

    for index, line in enumerate(lines):
        if any(label in line["norm"] for label in labels):
            return index
    return None


def count_regex_candidates(text: str, pattern: re.Pattern[str] | None) -> int:
    """Count unique regex candidates in a text.

    Args:
        text: OCR text.
        pattern: Optional regex used for field extraction.

    Returns:
        Number of unique candidates. Returns zero when no pattern applies.
    """

    if pattern is None:
        return 0
    candidates = {normalize_for_comparison(match.group(0)) for match in pattern.finditer(text)}
    return len({candidate for candidate in candidates if candidate})


def count_label_candidates(
    lines: list[dict[str, Any]],
    label_index: int | None,
    window: int = 4,
) -> int:
    """Count possible non-label values around a label line.

    Args:
        lines: OCR line dictionaries.
        label_index: Index of the label line.
        window: Number of nearby lines inspected.

    Returns:
        Number of candidate value lines.
    """

    if label_index is None:
        return 0
    start = max(0, label_index - window)
    end = min(len(lines), label_index + window + 1)
    count = 0
    for index in range(start, end):
        if index == label_index:
            continue
        candidate = clean_value(lines[index]["text"])
        if candidate and not is_label_line(lines[index]["norm"]):
            count += 1
    return count


def is_format_valid(field_name: str, value: Any) -> bool:
    """Validate a parsed value according to the field type.

    Args:
        field_name: Parsed field name.
        value: Parsed field value.

    Returns:
        ``True`` when the value shape is plausible for the field.
    """

    if isinstance(value, list):
        return bool(value) and all(is_format_valid(field_name, item) for item in value)
    value_text = str(value).strip()
    if not value_text:
        return False
    if field_name in {"cpf", "numero_cpf"}:
        return len(only_digits(value_text)) == 11
    if "data" in field_name or field_name in {"validade", "primeira_habilitacao"}:
        return is_plausible_date(value_text, field_name)
    if field_name == "registro_geral":
        return 5 <= len(normalize_for_comparison(value_text)) <= 14
    if field_name == "numero_registro":
        return 8 <= len(only_digits(value_text)) <= 12
    if field_name == "site":
        return SITE_PATTERN.fullmatch(normalize_text(value_text)) is not None
    return len(normalize_for_comparison(value_text)) >= 2


def regex_matches_perfectly(field_name: str, value: Any) -> bool | None:
    """Check full regex match for fields with regex validation.

    Args:
        field_name: Parsed field name.
        value: Parsed field value.

    Returns:
        ``True`` or ``False`` for regex-backed fields, otherwise ``None``.
    """

    pattern = regex_for_field(field_name)
    if pattern is None:
        return None
    if isinstance(value, list):
        return all(regex_matches_perfectly(field_name, item) is True for item in value)
    value_text = normalize_numeric_noise(str(value)).strip()
    if field_name in {"cpf", "numero_cpf"}:
        value_text = format_cpf(value_text)
    return pattern.fullmatch(value_text) is not None


def distance_score(distance_to_label: int | None) -> float | None:
    """Convert label distance into a normalized score.

    Args:
        distance_to_label: Signed line distance from label to value.

    Returns:
        Score from 0 to 1, or ``None`` when no label was found.
    """

    if distance_to_label is None:
        return None
    absolute_distance = abs(distance_to_label)
    if absolute_distance == 0:
        return 1.0
    return max(0.0, 1.0 - ((absolute_distance - 1) / 4.0))


def score_parsing_signals(signals: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """Calculate an explainable parsing confidence from validation signals.

    Args:
        signals: Field-level parsing signals.

    Returns:
        Tuple with percentage confidence and weighted component contributions.
    """

    checks: list[tuple[str, float, float | None]] = [
        ("field_extracted", 0.22, 1.0 if signals["field_extracted"] else 0.0),
        (
            "came_after_expected_label",
            0.16,
            1.0 if signals["came_after_expected_label"] else 0.0
            if signals["expected_label_found"]
            else None,
        ),
        ("distance_to_label", 0.12, distance_score(signals["distance_to_label"])),
        (
            "regex_full_match",
            0.18,
            1.0 if signals["regex_full_match"] else 0.0
            if signals["regex_full_match"] is not None
            else None,
        ),
        ("format_valid", 0.16, 1.0 if signals["format_valid"] else 0.0),
        ("no_candidate_conflict", 0.08, 0.0 if signals["conflict_with_other_candidates"] else 1.0),
        (
            "minimum_expected_field_found",
            0.04,
            1.0 if signals["minimum_expected_field_found"] else 0.0
            if signals["minimum_expected_field"]
            else None,
        ),
        (
            "cpf_has_11_digits",
            0.06,
            1.0 if signals["cpf_has_11_digits"] else 0.0
            if signals["cpf_has_11_digits"] is not None
            else None,
        ),
        (
            "cpf_check_digits_valid",
            0.06,
            1.0 if signals["cpf_check_digits_valid"] else 0.0
            if signals["cpf_check_digits_valid"] is not None
            else None,
        ),
        (
            "date_plausible",
            0.06,
            1.0 if signals["date_plausible"] else 0.0
            if signals["date_plausible"] is not None
            else None,
        ),
    ]
    applicable = [(name, weight, value) for name, weight, value in checks if value is not None]
    total_weight = sum(weight for _, weight, _ in applicable)
    if not applicable or total_weight == 0:
        return 0.0, {}
    components = {
        name: round((weight * float(value) / total_weight) * 100, 2)
        for name, weight, value in applicable
    }
    return round(sum(components.values()), 2), components


def build_field_parsing_signal(
    document_type: str,
    field_name: str,
    value: Any,
    fields: dict[str, Any],
    lines: list[dict[str, Any]],
    text: str,
) -> dict[str, Any]:
    """Build auditable parsing signals for one field.

    Args:
        document_type: Known document type.
        field_name: Parsed field name.
        value: Parsed field value.
        fields: All parsed fields.
        lines: OCR line dictionaries.
        text: OCR text.

    Returns:
        Dictionary with validation signals and score components.
    """

    value_for_lookup = " ".join(value) if isinstance(value, list) else str(value)
    labels = expected_labels_for_field(document_type, field_name)
    label_index = find_label_index(lines, labels)
    value_index = line_index_for_value(value_for_lookup, lines)
    distance_to_label = (
        value_index - label_index
        if label_index is not None and value_index is not None
        else None
    )
    pattern = regex_for_field(field_name)
    regex_candidate_count = count_regex_candidates(text, pattern)
    label_candidate_count = count_label_candidates(lines, label_index)
    candidate_count = regex_candidate_count if pattern is not None else label_candidate_count
    minimum = field_name in expected_fields(document_type)
    digits = only_digits(value_for_lookup)
    is_cpf_field = field_name in {"cpf", "numero_cpf"}
    is_date_field = "data" in field_name or field_name in {"validade", "primeira_habilitacao"}

    signals: dict[str, Any] = {
        "field_extracted": field_name in fields,
        "expected_labels": labels,
        "expected_label_found": label_index is not None,
        "came_after_expected_label": distance_to_label is not None and distance_to_label >= 0,
        "distance_to_label": distance_to_label,
        "regex_full_match": regex_matches_perfectly(field_name, value),
        "format_valid": is_format_valid(field_name, value),
        "candidate_count": candidate_count,
        "conflict_with_other_candidates": candidate_count > 1,
        "minimum_expected_field": minimum,
        "minimum_expected_field_found": minimum and field_name in fields,
        "cpf_has_11_digits": len(digits) == 11 if is_cpf_field else None,
        "cpf_check_digits_valid": is_valid_cpf_checksum(value_for_lookup) if is_cpf_field else None,
        "date_plausible": is_plausible_date(value_for_lookup, field_name) if is_date_field else None,
    }
    confidence, components = score_parsing_signals(signals)
    signals["parsing_confidence"] = confidence
    signals["score_components"] = components
    return signals


def build_parsing_audit(
    document_type: str,
    fields: dict[str, Any],
    lines: list[dict[str, Any]],
    text: str,
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    """Build parsing confidence and audit signals for all extracted fields.

    Args:
        document_type: Known document type.
        fields: Parsed fields.
        lines: OCR line dictionaries.
        text: OCR text.

    Returns:
        Tuple with confidence by field and signals by field.
    """

    signals_by_field = {
        field_name: build_field_parsing_signal(
            document_type=document_type,
            field_name=field_name,
            value=value,
            fields=fields,
            lines=lines,
            text=text,
        )
        for field_name, value in fields.items()
    }
    confidences = {
        field_name: float(signals["parsing_confidence"])
        for field_name, signals in signals_by_field.items()
    }
    return confidences, signals_by_field


def find_first_cpf(text: str) -> str | None:
    """Extract the first CPF candidate from OCR text.

    Args:
        text: Full OCR text.

    Returns:
        Formatted CPF, or ``None``.
    """

    return find_regex(normalize_numeric_noise(text), CPF_PATTERN, format_cpf)


def collect_filiation(lines: list[dict[str, Any]]) -> list[str]:
    """Collect filiation names near the ``FILIAÇÃO`` label.

    Args:
        lines: OCR line dictionaries.

    Returns:
        List of up to two probable parent names.
    """

    stop_markers = (
        "CPF",
        "CAT",
        "PERMISSAO",
        "VALIDADE",
        "REGISTRO",
        "ASSINATURA",
        "NATURALIDADE",
        "DOC",
        "LOCAL",
    )
    names: list[str] = []
    for index, line in enumerate(lines):
        if "FILIACAO" not in line["norm"]:
            continue
        for candidate in lines[index + 1 : index + 6]:
            if any(marker in candidate["norm"] for marker in stop_markers):
                break
            value = clean_value(candidate["text"])
            if value and not is_label_line(candidate["norm"]) and not DATE_PATTERN.search(value):
                names.append(value)
            if len(names) >= 2:
                return names
    return names


def parse_cpf(lines: list[dict[str, Any]], text: str, document_type: str) -> dict[str, Any]:
    """Parse CPF front or back fields.

    Args:
        lines: OCR line dictionaries.
        text: Full OCR text.
        document_type: Known CPF side.

    Returns:
        Extracted CPF fields.
    """

    fields: dict[str, Any] = {}
    if document_type == "CPF_Frente":
        fields["numero_cpf"] = find_first_cpf(text)
        fields["nome"] = find_value_after_label(lines, ("NOME",))
        fields["data_nascimento"] = find_date_after_label(lines, ("NASCIMENTO",))
    else:
        fields["emissao"] = find_value_after_label(lines, ("EMISSAO",))
        fields["site"] = find_regex(text, SITE_PATTERN)
    return fields


def parse_cnh(lines: list[dict[str, Any]], text: str, document_type: str) -> dict[str, Any]:
    """Parse CNH front or back fields.

    Args:
        lines: OCR line dictionaries.
        text: Full OCR text.
        document_type: Known CNH side.

    Returns:
        Extracted CNH fields.
    """

    fields: dict[str, Any] = {}
    if document_type == "CNH_Frente":
        fields["nome"] = find_value_after_label(lines, ("NOME",))
        fields["cpf"] = find_first_cpf(text)
        fields["data_nascimento"] = find_date_after_label(lines, ("DATA NASCIMENTO", "NASCIMENTO"))
        fields["documento_identidade"] = find_value_after_label(lines, ("DOC IDENTIDADE", "IDENTIDADE"))
        fields["filiacao"] = collect_filiation(lines)
        fields["validade"] = find_date_after_label(lines, ("VALIDADE",))
        fields["primeira_habilitacao"] = find_date_after_label(lines, ("1A HABILITACAO", "1ª HABILITACAO"))
        fields["numero_registro"] = find_value_after_label(lines, ("N REGISTRO", "REGISTRO"))
        fields["categoria"] = find_value_after_label(lines, ("CAT HAB", "CAT. HAB"))
    else:
        fields["local"] = find_value_after_label(lines, ("LOCAL",))
        fields["data_emissao"] = find_date_after_label(lines, ("DATA EMISSAO", "EMISSAO"))
        fields["numero_registro"] = find_regex(text, LONG_DIGIT_PATTERN)
        fields["observacoes"] = find_value_after_label(lines, ("OBSERVACOES",))
    return fields


def parse_rg(lines: list[dict[str, Any]], text: str, document_type: str) -> dict[str, Any]:
    """Parse RG front or back fields.

    Args:
        lines: OCR line dictionaries.
        text: Full OCR text.
        document_type: Known RG side.

    Returns:
        Extracted RG fields.
    """

    fields: dict[str, Any] = {}
    if document_type == "RG_Frente":
        fields["orgao_emissor"] = find_value_after_label(lines, ("SECRETARIA", "INSTITUTO"))
        fields["observacoes"] = find_value_after_label(lines, ("NAO DOADOR", "DOADOR"))
    else:
        fields["registro_geral"] = find_value_after_label(lines, ("REGISTRO GERAL", "REGISTRO"))
        if not fields["registro_geral"]:
            fields["registro_geral"] = find_regex(normalize_numeric_noise(text), RG_PATTERN)
        fields["data_expedicao"] = find_date_after_label(lines, ("DATA EXPEDICAO", "EXPEDICAO"))
        fields["nome"] = find_value_after_label(lines, ("NOME",))
        fields["filiacao"] = collect_filiation(lines)
        fields["naturalidade"] = find_value_after_label(lines, ("NATURALIDADE",))
        fields["doc_origem"] = find_value_after_label(lines, ("DOC ORIGEM",))
        fields["cpf"] = find_first_cpf(text)
        fields["data_nascimento"] = find_date_after_label(lines, ("DATA DE NASCIMENTO", "NASCIMENTO"))
    return fields


def remove_empty_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """Remove empty scalar values while preserving meaningful empty lists.

    Args:
        fields: Raw parser fields.

    Returns:
        Cleaned fields dictionary.
    """

    cleaned: dict[str, Any] = {}
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        cleaned[key] = value
    return cleaned


def expected_fields(document_type: str) -> tuple[str, ...]:
    """Return the minimum expected fields for warnings.

    Args:
        document_type: Known document type.

    Returns:
        Tuple of field names expected for that document side.
    """

    mapping = {
        "CPF_Frente": ("numero_cpf", "nome", "data_nascimento"),
        "CPF_Verso": ("emissao",),
        "CNH_Frente": ("nome", "cpf", "data_nascimento"),
        "CNH_Verso": ("data_emissao",),
        "RG_Frente": ("orgao_emissor",),
        "RG_Verso": ("registro_geral", "nome", "cpf", "data_nascimento"),
    }
    return mapping.get(document_type, ())


def build_confidences(fields: dict[str, Any], lines: list[dict[str, Any]]) -> dict[str, float | None]:
    """Estimate confidence for each extracted field.

    Args:
        fields: Parsed fields.
        lines: OCR line dictionaries.

    Returns:
        Mapping from field name to estimated confidence.
    """

    confidences: dict[str, float | None] = {}
    for key, value in fields.items():
        if isinstance(value, list):
            item_confidences = [confidence_for_value(item, lines) for item in value]
            valid = [item for item in item_confidences if item is not None]
            confidences[key] = sum(valid) / len(valid) if valid else None
        else:
            confidences[key] = confidence_for_value(str(value), lines)
    return confidences


def build_warnings(
    document_type: str,
    fields: dict[str, Any],
    confidences: dict[str, float | None],
    min_confidence: float,
) -> list[str]:
    """Create missing-field and low-confidence warnings.

    Args:
        document_type: Known document type.
        fields: Parsed fields.
        confidences: Confidence estimates by field.
        min_confidence: Minimum acceptable confidence.

    Returns:
        Human-readable warnings.
    """

    warnings: list[str] = []
    for field_name in expected_fields(document_type):
        if field_name not in fields:
            warnings.append(f"Campo esperado não encontrado: {field_name}.")
    for field_name, confidence in confidences.items():
        if confidence is not None and confidence < min_confidence:
            warnings.append(
                f"Campo com baixa confiança OCR: {field_name} ({confidence:.1f}%)."
            )
    return warnings


def build_quality_warnings(
    document_type: str,
    fields: dict[str, Any],
    ocr_confidences: dict[str, float | None],
    parsing_confidences: dict[str, float],
    min_confidence: float,
) -> list[str]:
    """Create missing-field, low-OCR-confidence, and low-parsing-confidence warnings.

    Args:
        document_type: Known document type.
        fields: Parsed fields.
        ocr_confidences: OCR confidence estimates by field.
        parsing_confidences: Parsing confidence estimates by field.
        min_confidence: Minimum acceptable confidence.

    Returns:
        Human-readable warnings.
    """

    warnings: list[str] = []
    for field_name in expected_fields(document_type):
        if field_name not in fields:
            warnings.append(f"Campo esperado não encontrado: {field_name}.")
    for field_name, confidence in ocr_confidences.items():
        if confidence is not None and confidence < min_confidence:
            warnings.append(f"Campo com baixa confiança OCR: {field_name} ({confidence:.1f}%).")
    for field_name, confidence in parsing_confidences.items():
        if confidence < min_confidence:
            warnings.append(f"Campo com baixa confiança de parsing: {field_name} ({confidence:.1f}%).")
    return warnings


def parse_document(
    document_type: str,
    ocr_result: OCRResult,
    min_confidence: float,
) -> ParsedDocument:
    """Parse OCR output according to the known document type.

    Args:
        document_type: Known document type.
        ocr_result: Structured OCR result.
        min_confidence: Threshold used to emit confidence warnings.

    Returns:
        Parsed document with fields, confidences, and warnings.
    """

    lines = line_dicts(ocr_result)
    normalized_text = normalize_numeric_noise(ocr_result.text)
    if document_type.startswith("CPF"):
        fields = parse_cpf(lines, normalized_text, document_type)
    elif document_type.startswith("CNH"):
        fields = parse_cnh(lines, normalized_text, document_type)
    elif document_type.startswith("RG"):
        fields = parse_rg(lines, normalized_text, document_type)
    else:
        fields = {}
    cleaned_fields = remove_empty_fields(fields)
    ocr_confidences = build_confidences(cleaned_fields, lines)
    parsing_confidences, parsing_signals = build_parsing_audit(
        document_type,
        cleaned_fields,
        lines,
        normalized_text,
    )
    warnings = build_quality_warnings(
        document_type,
        cleaned_fields,
        ocr_confidences,
        parsing_confidences,
        min_confidence,
    )
    return ParsedDocument(
        document_type=document_type,
        fields=cleaned_fields,
        confidence_by_field=ocr_confidences,
        parsing_confidence_by_field=parsing_confidences,
        parsing_signals_by_field=parsing_signals,
        warnings=warnings,
    )
