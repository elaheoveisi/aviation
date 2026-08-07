"""Reproducible structural audit for the raw NTSB workbooks."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable
import itertools
import re
import gc
import json
import math
import subprocess
import sys

from .features import load_feature_config
from .utils import clean_text, excel_serial_to_date, is_blank, to_float, to_int, write_csv
from .xlsx_stream import iter_dicts, iter_rows, list_sheets, read_header


def _dimension_shape(reference: str | None) -> tuple[int | None, int | None]:
    if not reference or ":" not in reference:
        return None, None
    end = reference.split(":")[-1]
    match = re.match(r"([A-Z]+)(\d+)", end)
    if not match:
        return None, None
    letters, row_text = match.groups()
    columns = 0
    for char in letters:
        columns = columns * 26 + ord(char) - 64
    return int(row_text) - 1, columns


def _normalized(value):
    if is_blank(value):
        return None
    return str(value).strip()


def _semantically_equal(value_a, value_b) -> bool:
    normalized_a = _normalized(value_a)
    normalized_b = _normalized(value_b)
    if normalized_a == normalized_b:
        return True
    try:
        number_a = float(normalized_a)
        number_b = float(normalized_b)
    except (TypeError, ValueError):
        return False
    return math.isclose(number_a, number_b, rel_tol=1e-7, abs_tol=1e-8)


def compare_sheets(path: str | Path, sheet_a: str, sheet_b: str) -> dict:
    rows_a = iter_rows(path, sheet_a)
    rows_b = iter_rows(path, sheet_b)
    differing_rows = 0
    differing_cells = 0
    representation_only_cells = 0
    raw_string_differences = 0
    missing_rows_a = 0
    missing_rows_b = 0
    column_differences: Counter[int] = Counter()
    representation_columns: Counter[int] = Counter()
    compared_rows = 0
    for row_a, row_b in itertools.zip_longest(rows_a, rows_b):
        if row_a is None:
            missing_rows_a += 1
            continue
        if row_b is None:
            missing_rows_b += 1
            continue
        compared_rows += 1
        width = max(len(row_a), len(row_b))
        row_diff = False
        for index in range(width):
            value_a = row_a[index] if index < len(row_a) else None
            value_b = row_b[index] if index < len(row_b) else None
            if _normalized(value_a) != _normalized(value_b):
                raw_string_differences += 1
                if _semantically_equal(value_a, value_b):
                    representation_only_cells += 1
                    representation_columns[index] += 1
                else:
                    row_diff = True
                    differing_cells += 1
                    column_differences[index] += 1
        differing_rows += int(row_diff)
    header = read_header(path, sheet_a)
    differences_named = {
        header[index] if index < len(header) else f"column_{index+1}": count
        for index, count in column_differences.items()
    }
    representation_named = {
        header[index] if index < len(header) else f"column_{index+1}": count
        for index, count in representation_columns.items()
    }
    return {
        "sheet_a": sheet_a,
        "sheet_b": sheet_b,
        "compared_rows_including_header": compared_rows,
        "raw_string_differences": raw_string_differences,
        "numeric_representation_only_cells": representation_only_cells,
        "substantive_differing_rows": differing_rows,
        "substantive_differing_cells": differing_cells,
        "rows_only_in_sheet_a": missing_rows_b,
        "rows_only_in_sheet_b": missing_rows_a,
        "column_differences": differences_named,
        "representation_only_columns": representation_named,
        "equivalent_after_numeric_tolerance": (
            differing_cells == 0 and missing_rows_a == 0 and missing_rows_b == 0
        ),
    }

def compare_sheets_isolated(path: str | Path, sheet_a: str, sheet_b: str) -> dict:
    """Run the large duplicate-sheet comparison in a clean process."""
    completed = subprocess.run(
        [sys.executable, "-m", "ntsb_severity.compare_cli", str(path), sheet_a, sheet_b],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)

def _feature_source(feature: str, event_header: set[str], aircraft_header: set[str]) -> str:
    derived = {
        "crew_count",
        "female_crew_count",
        "has_second_pilot",
        "mean_age",
        "med_certf_mode",
        "med_crtf_vldty_mode",
    }
    if feature in derived:
        return "derived_from_Flight_Crew"
    in_event = feature in event_header
    in_aircraft = feature in aircraft_header
    if in_event and in_aircraft:
        return "events_and_aircraft"
    if in_event:
        return "events"
    if in_aircraft:
        return "aircraft"
    return "not_found"


def run_audit(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    feature_config: str | Path,
) -> dict:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "events": data_dir / "events.xlsx",
        "aircraft": data_dir / "aircraft.xlsx",
        "crew": data_dir / "Flight_Crew.xlsx",
        "findings": data_dir / "Findings.xlsx",
        "old_results": data_dir / "aviation_model_results_final.xlsx",
    }

    inventory_rows = []
    for file_role, path in paths.items():
        if not path.exists():
            inventory_rows.append({
                "file_role": file_role,
                "file_name": path.name,
                "exists": False,
                "sheet": None,
                "data_rows": None,
                "columns": None,
                "dimension": None,
            })
            continue
        for sheet in list_sheets(path):
            rows, columns = _dimension_shape(sheet.dimension)
            inventory_rows.append({
                "file_role": file_role,
                "file_name": path.name,
                "exists": True,
                "sheet": sheet.name,
                "data_rows": rows,
                "columns": columns,
                "dimension": sheet.dimension,
            })
    write_csv(output_dir / "01_file_inventory.csv", inventory_rows)

    # Run the large duplicate-sheet comparison in a clean process before the
    # parent process accumulates relational audit state.
    comparison_cache = output_dir / "aircraft_compare.json"
    if comparison_cache.exists():
        aircraft_comparison = json.loads(comparison_cache.read_text(encoding="utf-8"))
    else:
        aircraft_comparison = compare_sheets_isolated(paths["aircraft"], "aircraft", "aircraft1")
        comparison_cache.write_text(json.dumps(aircraft_comparison, indent=2), encoding="utf-8")
    comparison_row = {key: value for key, value in aircraft_comparison.items() if key not in {"column_differences", "representation_only_columns"}}
    comparison_row["column_differences"] = "; ".join(
        f"{key}:{value}" for key, value in aircraft_comparison["column_differences"].items()
    )
    comparison_row["representation_only_columns"] = "; ".join(
        f"{key}:{value}" for key, value in aircraft_comparison["representation_only_columns"].items()
    )
    write_csv(output_dir / "02_aircraft_sheet_comparison.csv", [comparison_row])

    # Events audit.
    event_ids: set[str] = set()
    duplicate_event_ids = 0
    missing_event_ids = 0
    year_counts: Counter[int] = Counter()
    severe_by_year: Counter[int] = Counter()
    severe_total = 0
    nonsevere_total = 0
    cohort_2008_2025_rows = 0
    cohort_2008_2025_severe = 0
    event_date_values: list[float] = []
    event_rows = 0
    for event in iter_dicts(paths["events"], "events"):
        event_rows += 1
        event_id = clean_text(event.get("ev_id"))
        if event_id is None:
            missing_event_ids += 1
        elif event_id in event_ids:
            duplicate_event_ids += 1
        else:
            event_ids.add(event_id)
        year = to_int(event.get("ev_year"))
        if year is not None:
            year_counts[year] += 1
        fatal = to_float(event.get("inj_tot_f")) or 0.0
        serious = to_float(event.get("inj_tot_s")) or 0.0
        severe = int(fatal > 0 or serious > 0)
        severe_total += severe
        nonsevere_total += 1 - severe
        if year is not None:
            severe_by_year[year] += severe
            if 2008 <= year <= 2025:
                cohort_2008_2025_rows += 1
                cohort_2008_2025_severe += severe
        date_value = to_float(event.get("ev_date"))
        if date_value is not None:
            event_date_values.append(date_value)

    event_summary = [{
        "rows": event_rows,
        "unique_event_ids": len(event_ids),
        "missing_event_ids": missing_event_ids,
        "duplicate_event_ids": duplicate_event_ids,
        "minimum_year": min(year_counts) if year_counts else None,
        "maximum_year": max(year_counts) if year_counts else None,
        "minimum_event_date": excel_serial_to_date(min(event_date_values)) if event_date_values else None,
        "maximum_event_date": excel_serial_to_date(max(event_date_values)) if event_date_values else None,
        "severe_events": severe_total,
        "nonsevere_events": nonsevere_total,
        "severe_prevalence": severe_total / event_rows if event_rows else None,
        "rows_2008_2025": cohort_2008_2025_rows,
        "severe_2008_2025": cohort_2008_2025_severe,
        "nonsevere_2008_2025": cohort_2008_2025_rows - cohort_2008_2025_severe,
        "rows_2026": year_counts.get(2026, 0),
        "severe_2026": severe_by_year.get(2026, 0),
    }]
    write_csv(output_dir / "03_events_summary.csv", event_summary)
    gc.collect()
    write_csv(
        output_dir / "04_year_distribution.csv",
        [
            {
                "year": year,
                "events": year_counts[year],
                "severe": severe_by_year[year],
                "severe_rate": severe_by_year[year] / year_counts[year],
            }
            for year in sorted(year_counts)
        ],
    )

    # Aircraft key audit.
    aircraft_rows = 0
    aircraft_event_keys: dict[str, list[str | None]] = defaultdict(list)
    aircraft_pair_counts: Counter[tuple[str, str | None]] = Counter()
    for row in iter_dicts(paths["aircraft"], "aircraft"):
        aircraft_rows += 1
        event_id = clean_text(row.get("ev_id"))
        key = clean_text(row.get("Aircraft_Key"))
        if event_id:
            aircraft_event_keys[event_id].append(key)
            aircraft_pair_counts[(event_id, key)] += 1
    multi_aircraft_events = sum(len(keys) > 1 for keys in aircraft_event_keys.values())
    events_without_key_1 = sum("1" not in keys for keys in aircraft_event_keys.values())
    duplicate_aircraft_pairs = sum(count - 1 for count in aircraft_pair_counts.values() if count > 1)
    event_ids_without_aircraft = len(event_ids - set(aircraft_event_keys))
    aircraft_event_ids_not_in_events = len(set(aircraft_event_keys) - event_ids)
    selected_aircraft_key: dict[str, str | None] = {}
    for event_id, keys in aircraft_event_keys.items():
        if "1" in keys:
            selected_aircraft_key[event_id] = "1"
        else:
            numeric = sorted((int(key), key) for key in keys if key and key.isdigit())
            selected_aircraft_key[event_id] = numeric[0][1] if numeric else sorted(key or "" for key in keys)[0]
    aircraft_summary = [{
        "rows": aircraft_rows,
        "events_with_aircraft": len(aircraft_event_keys),
        "events_with_multiple_aircraft": multi_aircraft_events,
        "events_without_aircraft_key_1": events_without_key_1,
        "duplicate_event_aircraft_key_rows": duplicate_aircraft_pairs,
        "events_table_ids_without_aircraft": event_ids_without_aircraft,
        "aircraft_ids_not_in_events_table": aircraft_event_ids_not_in_events,
    }]
    write_csv(output_dir / "05_aircraft_key_summary.csv", aircraft_summary)
    gc.collect()

    # Crew audit and primary-aircraft crew coverage.
    crew_rows = 0
    crew_group_counts: Counter[tuple[str, str | None]] = Counter()
    crew_number_counts: Counter[tuple[str, str | None, str | None]] = Counter()
    crew_categories: Counter[str] = Counter()
    for row in iter_dicts(paths["crew"], "Flight_Crew"):
        crew_rows += 1
        event_id = clean_text(row.get("ev_id"))
        key = clean_text(row.get("Aircraft_Key"))
        crew_number = clean_text(row.get("crew_no"))
        category = clean_text(row.get("crew_category"))
        if category:
            crew_categories[category] += 1
        if event_id:
            crew_group_counts[(event_id, key)] += 1
            crew_number_counts[(event_id, key, crew_number)] += 1
    events_with_primary_aircraft_crew = 0
    events_with_multiple_primary_aircraft_crew = 0
    for event_id in event_ids:
        key = selected_aircraft_key.get(event_id)
        count = crew_group_counts.get((event_id, key), 0)
        events_with_primary_aircraft_crew += int(count > 0)
        events_with_multiple_primary_aircraft_crew += int(count > 1)
    crew_summary = [{
        "rows": crew_rows,
        "event_aircraft_groups": len(crew_group_counts),
        "duplicate_event_aircraft_crew_number_rows": sum(
            count - 1 for count in crew_number_counts.values() if count > 1
        ),
        "events_with_primary_aircraft_crew": events_with_primary_aircraft_crew,
        "events_without_primary_aircraft_crew": len(event_ids) - events_with_primary_aircraft_crew,
        "events_with_multiple_crew_on_primary_aircraft": events_with_multiple_primary_aircraft_crew,
        "crew_category_counts": "; ".join(f"{key}:{value}" for key, value in sorted(crew_categories.items())),
    }]
    write_csv(output_dir / "06_crew_key_summary.csv", crew_summary)
    gc.collect()

    # Findings audit: descriptive only; these records must not enter early-screening models.
    finding_rows = 0
    finding_event_ids: set[str] = set()
    for row in iter_dicts(paths["findings"], "Findings"):
        finding_rows += 1
        event_id = clean_text(row.get("ev_id"))
        if event_id:
            finding_event_ids.add(event_id)
    findings_summary = [{
        "rows": finding_rows,
        "events_with_findings": len(finding_event_ids),
        "events_without_findings": len(event_ids - finding_event_ids),
        "model_use": "EXCLUDE",
        "reason": "Investigation-derived causal/contributing factors are post-investigation.",
    }]
    write_csv(output_dir / "07_findings_exclusion_summary.csv", findings_summary)
    gc.collect()

    # Feature list audit against source tables.
    numeric_features, categorical_features = load_feature_config(feature_config)
    event_header = set(read_header(paths["events"], "events"))
    aircraft_header = set(read_header(paths["aircraft"], "aircraft"))
    feature_rows = []
    for feature_type, feature_names in (
        ("numeric", numeric_features),
        ("categorical", categorical_features),
    ):
        for feature in feature_names:
            source = _feature_source(feature, event_header, aircraft_header)
            feature_rows.append({
                "feature": feature,
                "declared_type": feature_type,
                "source": source,
                "found_or_derived": source != "not_found",
                "timing_status": "reviewed_candidate",
            })
    write_csv(output_dir / "08_manuscript_feature_registry_audit.csv", feature_rows)

    # Old results audit.
    old_features = []
    if paths["old_results"].exists():
        for row in iter_dicts(paths["old_results"], "Feature_List"):
            feature = clean_text(row.get("feature"))
            if feature:
                old_features.append(feature)
    confirmed_leakage = {"ev_type", "damage", "crew_tox_positive_count"}
    findings_derived = {feature for feature in old_features if feature.startswith("find") or feature in {"findings_count", "cause_count", "factor_count"}}
    old_results_summary = [{
        "old_results_input_feature_count": len(old_features),
        "contains_ev_type": "ev_type" in old_features,
        "contains_damage": "damage" in old_features,
        "contains_crew_toxicology": "crew_tox_positive_count" in old_features,
        "contains_findings_derived_features": bool(findings_derived),
        "confirmed_inappropriate_features": "; ".join(sorted(confirmed_leakage & set(old_features))),
        "findings_derived_features": "; ".join(sorted(findings_derived)),
        "status": "Do not use as the final leakage-controlled results source.",
    }]
    write_csv(output_dir / "09_old_results_workbook_audit.csv", old_results_summary)

    split_rows = []
    for scope, maximum_year in (("declared_2008_2025", 2025), ("raw_including_2026", 2026)):
        for cut in (2017, 2019, 2021):
            n_train = sum(count for year, count in year_counts.items() if 2008 <= year <= cut)
            n_test = sum(count for year, count in year_counts.items() if cut < year <= maximum_year)
            split_rows.append({
                "dataset_scope": scope,
                "cut": cut,
                "train_period": f"2008-{cut}",
                "test_period": f"{cut+1}-{maximum_year}",
                "n_train": n_train,
                "n_test": n_test,
            })
    write_csv(output_dir / "10_temporal_split_counts.csv", split_rows)

    cut_2017_train = sum(count for year, count in year_counts.items() if 2008 <= year <= 2017)
    cut_2017_test_2018_2019 = sum(count for year, count in year_counts.items() if 2018 <= year <= 2019)
    cut_2017_test_2018_2025 = sum(count for year, count in year_counts.items() if 2018 <= year <= 2025)
    cut_2017_test_2018_2026 = sum(count for year, count in year_counts.items() if 2018 <= year <= 2026)

    issues = [
        {
            "severity": "critical",
            "issue": "The stated 2026 exclusion was not implemented",
            "evidence": f"The raw file has {year_counts.get(2026, 0)} events from 2026. The reported total n=30,516 and temporal test counts include these records. Restricting the cohort to 2008-2025 gives n={cohort_2008_2025_rows}.",
            "required_action": "Filter ev_year <= 2025 before cohort summaries, splits, preprocessing, model fitting, and all figures.",
        },
        {
            "severity": "critical",
            "issue": "Appendix predictor count mismatch",
            "evidence": f"The supplied Appendix A feature list contains {len(numeric_features) + len(categorical_features)} predictors ({len(numeric_features)} numeric and {len(categorical_features)} categorical), not 85.",
            "required_action": "Choose and document one final predictor registry, then update Methods, Appendix A, and all code-generated counts.",
        },
        {
            "severity": "critical",
            "issue": "Rolling temporal cut text mismatch",
            "evidence": f"Training through 2017 gives n={cut_2017_train}. Testing 2018-2019 gives n={cut_2017_test_2018_2019}; testing 2018-2025 gives n={cut_2017_test_2018_2025}; testing 2018-2026 gives n={cut_2017_test_2018_2026}. The manuscript table value of 13,400 corresponds to 2018-2026.",
            "required_action": "Exclude 2026 and rerun all temporal results, then describe the rolling test windows exactly.",
        },
        {
            "severity": "critical",
            "issue": "Old results workbook is leaky",
            "evidence": "The old results feature list includes ev_type, damage, crew toxicology, and findings-derived variables.",
            "required_action": "Regenerate all final results from the raw events, aircraft, and crew tables using the approved leakage registry.",
        },
        {
            "severity": "major",
            "issue": "Duplicate aircraft worksheets",
            "evidence": f"The sheets have {aircraft_comparison['raw_string_differences']} raw string differences, all attributable to floating-point representation: {aircraft_comparison['equivalent_after_numeric_tolerance']}.",
            "required_action": "Use the aircraft sheet consistently; document that aircraft1 differs only in floating-point storage precision for airframe-hour fields.",
        },
    ]
    write_csv(output_dir / "11_manuscript_issues.csv", issues)

    report_lines = [
        "# NTSB raw-data audit",
        "",
        "## Core cohort",
        f"- Raw event rows: {event_rows:,}",
        f"- Unique event IDs: {len(event_ids):,}",
        f"- Declared 2008-2025 cohort: {cohort_2008_2025_rows:,}",
        f"- Records dated 2026: {year_counts.get(2026, 0):,}",
        f"- Date range: {event_summary[0]['minimum_event_date']} to {event_summary[0]['maximum_event_date']}",
        f"- Raw severe events: {severe_total:,} ({event_summary[0]['severe_prevalence']:.1%})",
        f"- 2008-2025 severe events: {cohort_2008_2025_severe:,} ({cohort_2008_2025_severe/cohort_2008_2025_rows:.1%})",
        f"- 2008-2025 non-severe events: {cohort_2008_2025_rows - cohort_2008_2025_severe:,}",
        "",
        "## Relational integrity",
        f"- Aircraft rows: {aircraft_rows:,}",
        f"- Multi-aircraft events: {multi_aircraft_events:,}",
        f"- Event IDs without an aircraft record: {event_ids_without_aircraft:,}",
        f"- Events without Aircraft_Key=1: {events_without_key_1:,}",
        f"- Crew rows: {crew_rows:,}",
        f"- Events without crew on the selected primary aircraft: {len(event_ids) - events_with_primary_aircraft_crew:,}",
        "",
        "## Confirmed manuscript corrections",
        f"- Appendix A currently contains {len(numeric_features) + len(categorical_features)} predictors, not 85.",
        f"- The reported n=30,516 includes {year_counts.get(2026, 0):,} records from 2026; the stated 2008-2025 cohort contains {cohort_2008_2025_rows:,} events.",
        f"- The 2017 table test count of {cut_2017_test_2018_2026:,} matches 2018-2026, not 2018-2019 ({cut_2017_test_2018_2019:,}) or 2018-2025 ({cut_2017_test_2018_2025:,}).",
        "- The supplied old model-results workbook must not be used for the revised paper because it includes post-investigation and outcome-adjacent predictors.",
        "- Findings.xlsx is retained only for audit documentation and is excluded from model construction.",
        "",
        "## Aircraft worksheet decision",
        f"- aircraft and aircraft1 are equivalent within numeric tolerance: {aircraft_comparison['equivalent_after_numeric_tolerance']}.",
        f"- Raw representation differences: {aircraft_comparison['raw_string_differences']:,}; substantive differences: {aircraft_comparison['substantive_differing_cells']:,}.",
    ]
    (output_dir / "AUDIT_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return {
        "event_rows": event_rows,
        "severe_events": severe_total,
        "nonsevere_events": nonsevere_total,
        "feature_count_from_appendix": len(numeric_features) + len(categorical_features),
        "aircraft_sheets_equivalent": aircraft_comparison["equivalent_after_numeric_tolerance"],
        "output_dir": str(output_dir),
    }
