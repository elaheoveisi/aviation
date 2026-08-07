"""Audit the manuscript predictor registry against the NTSB eADMS data dictionary."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import yaml

from .xlsx_stream import iter_dicts

ALIASES: dict[str, list[str]] = {
    "dec_latitude": ["latitude"],
    "dec_longitude": ["longitude"],
    "crew_count": ["crew_no"],
    "female_crew_count": ["crew_sex"],
    "mean_age": ["crew_age"],
    "has_second_pilot": ["crew_no", "second_pilot"],
    "med_certf_mode": ["med_certf"],
    "med_crtf_vldty_mode": ["med_crtf_vldty"],
}


def _unique(values):
    output = []
    seen = set()
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output


def _read_missingness(path: str | Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return {row["feature"]: row for row in csv.DictReader(handle)}


def run_dictionary_audit(
    *,
    dictionary_path: str | Path,
    registry_path: str | Path,
    output_csv: str | Path,
    missingness_csv: str | Path | None = None,
    sheet_name: str = "eADMSPUB_DataDictionary",
) -> list[dict[str, object]]:
    """Generate a field-level timing audit from the dictionary and manual registry."""
    with Path(registry_path).open(encoding="utf-8") as handle:
        registry = yaml.safe_load(handle)["features"]

    rows_by_column: dict[str, list[dict[str, object | None]]] = defaultdict(list)
    for row in iter_dicts(dictionary_path, sheet_name):
        column = row.get("Column")
        if column is not None and str(column).strip():
            rows_by_column[str(column).strip()].append(row)

    missingness = _read_missingness(missingness_csv)
    output: list[dict[str, object]] = []
    for feature, manual in registry.items():
        dictionary_columns = ALIASES.get(feature, [feature])
        rows = [row for column in dictionary_columns for row in rows_by_column.get(column, [])]
        missing = missingness.get(feature, {})
        output.append({
            "feature": feature,
            "current_declared_type": manual.get("current_declared_type"),
            "recommended_model_type": manual.get("recommended_model_type"),
            "source": manual.get("source"),
            "dictionary_column_or_basis": "; ".join(dictionary_columns),
            "dictionary_table": "; ".join(_unique(row.get("Table") for row in rows)),
            "dictionary_data_type": "; ".join(_unique(row.get("Data Type eADMS") for row in rows)),
            "dictionary_label": " | ".join(_unique((row.get("short_desc") or row.get("meaning")) for row in rows)),
            "dictionary_definition": " | ".join(_unique(row.get("Question_Def") for row in rows)),
            "code_meanings_sample": "; ".join(_unique(row.get("meaning") for row in rows)[:12]),
            "missing_rate_2008_2025": missing.get("missing_rate"),
            "n_unique_nonmissing": missing.get("n_unique_nonmissing"),
            "timing_window": manual.get("timing_window"),
            "recommended_decision": manual.get("decision"),
            "timing_confidence": manual.get("confidence"),
            "preprocessing_rule": manual.get("preprocessing"),
            "audit_note": manual.get("note"),
        })

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    return output
