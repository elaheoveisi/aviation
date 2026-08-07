from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence
import csv
import math


def is_blank(value: object | None) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def clean_text(value: object | None) -> str | None:
    if is_blank(value):
        return None
    return str(value).strip()


def to_float(value: object | None) -> float | None:
    if is_blank(value):
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def to_int(value: object | None) -> int | None:
    number = to_float(value)
    return int(number) if number is not None else None


def excel_serial_to_date(value: object | None) -> str | None:
    number = to_float(value)
    if number is None:
        return None
    # Excel's 1900 date system, including the historical leap-year convention.
    date = datetime(1899, 12, 30) + timedelta(days=number)
    return date.date().isoformat()


def mode_or_none(values: Iterable[object | None]) -> str | None:
    cleaned = [clean_text(value) for value in values]
    cleaned = [value for value in cleaned if value is not None]
    if not cleaned:
        return None
    counts = Counter(cleaned)
    maximum = max(counts.values())
    # Deterministic tie handling.
    return sorted(key for key, count in counts.items() if count == maximum)[0]


def write_csv(path: str | Path, rows: Sequence[dict] | Iterable[dict], fieldnames=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv_dicts(path: str | Path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)
