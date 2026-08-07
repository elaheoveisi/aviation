"""Deterministic, leakage-safe feature engineering for NTSB records."""
from __future__ import annotations

import math
from typing import Any

from .utils import to_float, to_int


def _pair(angle: float | None, prefix: str) -> dict[str, float | None]:
    if angle is None:
        return {f"{prefix}_sin": None, f"{prefix}_cos": None}
    return {f"{prefix}_sin": math.sin(angle), f"{prefix}_cos": math.cos(angle)}


def cyclic_month(value: object | None, prefix: str = "ev_month") -> dict[str, float | None]:
    """Encode calendar month 1--12 as sine/cosine."""
    month = to_int(value)
    if month is None or month < 1 or month > 12:
        return _pair(None, prefix)
    angle = 2.0 * math.pi * (month - 1) / 12.0
    return _pair(angle, prefix)


def hhmm_minutes(value: object | None) -> int | None:
    """Convert NTSB HHMM values to minutes after midnight.

    A value of 2400 is normalized to 0000. Other invalid hour/minute values are
    returned as missing rather than silently clipped.
    """
    number = to_int(value)
    if number is None or number < 0:
        return None
    hours, minutes = divmod(number, 100)
    if hours == 24 and minutes == 0:
        return 0
    if hours > 23 or minutes > 59:
        return None
    return hours * 60 + minutes


def cyclic_hhmm(value: object | None, prefix: str) -> dict[str, float | None]:
    minutes = hhmm_minutes(value)
    if minutes is None:
        return _pair(None, prefix)
    angle = 2.0 * math.pi * minutes / 1440.0
    return _pair(angle, prefix)


def normalize_degrees(value: object | None) -> float | None:
    degrees = to_float(value)
    if degrees is None or degrees < 0 or degrees > 360:
        return None
    return degrees % 360.0


def cyclic_degrees(value: object | None, prefix: str) -> dict[str, float | None]:
    degrees = normalize_degrees(value)
    if degrees is None:
        return _pair(None, prefix)
    return _pair(2.0 * math.pi * degrees / 360.0, prefix)


def aircraft_age(event_year: object | None, manufacture_year: object | None) -> float | None:
    event = to_int(event_year)
    manufactured = to_int(manufacture_year)
    if event is None or manufactured is None:
        return None
    age = event - manufactured
    # Protect against data-entry errors while allowing historic aircraft.
    if age < 0 or age > 125:
        return None
    return float(age)


def elapsed_days(event_date_serial: object | None, earlier_date_serial: object | None) -> float | None:
    event = to_float(event_date_serial)
    earlier = to_float(earlier_date_serial)
    if event is None or earlier is None:
        return None
    days = event - earlier
    if days < 0 or days > 36525:  # 100 years; defensive data-quality bound.
        return None
    return float(days)


def engineer_features(row: dict[str, Any]) -> dict[str, float | None]:
    """Return all engineered features used by the final registries."""
    output: dict[str, float | None] = {
        "acft_age": aircraft_age(row.get("ev_year"), row.get("acft_year")),
        "days_since_last_inspection": elapsed_days(
            row.get("ev_date"), row.get("date_last_insp")
        ),
    }
    output.update(cyclic_month(row.get("ev_month"), "ev_month"))
    output.update(cyclic_hhmm(row.get("ev_time"), "ev_time"))
    output.update(cyclic_hhmm(row.get("dprt_time"), "dprt_time"))
    output.update(cyclic_hhmm(row.get("wx_obs_time"), "wx_obs_time"))
    output.update(cyclic_degrees(row.get("apt_dir"), "apt_dir"))
    output.update(cyclic_degrees(row.get("wind_dir_deg"), "wind_dir"))
    output.update(cyclic_degrees(row.get("wx_obs_dir"), "wx_obs_dir"))
    return output
