from __future__ import annotations

from collections import Counter
from pathlib import Path
import csv

from .features import load_feature_config
from .utils import clean_text, to_float, write_csv


def audit_analytical_dataset(
    dataset_csv: str | Path,
    feature_config: str | Path,
    output_csv: str | Path,
) -> list[dict]:
    numeric, categorical = load_feature_config(feature_config)
    features = numeric + categorical
    totals = Counter()
    missing = Counter()
    invalid_numeric = Counter()
    uniques = {feature: set() for feature in features}
    row_count = 0
    with Path(dataset_csv).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        absent = [feature for feature in features if feature not in (reader.fieldnames or [])]
        if absent:
            raise ValueError(f"Features absent from analytical dataset: {absent}")
        for row in reader:
            row_count += 1
            for feature in features:
                totals[feature] += 1
                value = clean_text(row.get(feature))
                if value is None:
                    missing[feature] += 1
                    continue
                uniques[feature].add(value)
                if feature in numeric and to_float(value) is None:
                    invalid_numeric[feature] += 1
    rows = []
    for kind, names in (("numeric", numeric), ("categorical", categorical)):
        for feature in names:
            rows.append({
                "feature": feature,
                "declared_type": kind,
                "rows": row_count,
                "missing_n": missing[feature],
                "missing_rate": missing[feature] / row_count if row_count else None,
                "nonmissing_n": row_count - missing[feature],
                "n_unique_nonmissing": len(uniques[feature]),
                "invalid_numeric_n": invalid_numeric[feature] if kind == "numeric" else None,
                "all_missing": missing[feature] == row_count,
            })
    write_csv(output_csv, rows)
    return rows
