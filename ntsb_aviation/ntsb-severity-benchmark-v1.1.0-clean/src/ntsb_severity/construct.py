"""Construct one leakage-controlled event-level analytical record per NTSB event."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable
import csv
import itertools

from .utils import clean_text, mode_or_none, to_float, to_int
from .xlsx_stream import iter_dicts
from .transformations import engineer_features


def severe_outcome(event: dict[str, object | None]) -> int:
    fatal = to_float(event.get("inj_tot_f")) or 0.0
    serious = to_float(event.get("inj_tot_s")) or 0.0
    return int(fatal > 0 or serious > 0)


def _aircraft_rank(key: str | None, preferred_key: str) -> tuple[int, int, str]:
    if key == preferred_key:
        return (0, 0, "")
    try:
        return (1, int(key), "")
    except (TypeError, ValueError):
        return (2, 0, key or "")


def select_primary_aircraft(
    records: Iterable[dict[str, object | None]], preferred_key: str = "1"
) -> dict[str, dict[str, object | None]]:
    """Select one aircraft row per event without retaining all candidates."""
    selected: dict[str, dict[str, object | None]] = {}
    selected_rank: dict[str, tuple[int, int, str]] = {}
    for record in records:
        event_id = clean_text(record.get("ev_id"))
        if not event_id:
            continue
        key = clean_text(record.get("Aircraft_Key"))
        rank = _aircraft_rank(key, preferred_key)
        if event_id not in selected or rank < selected_rank[event_id]:
            selected[event_id] = record
            selected_rank[event_id] = rank
    return selected


def aggregate_crew(
    records: Iterable[dict[str, object | None]],
    primary_aircraft: dict[str, dict[str, object | None]],
) -> dict[str, dict[str, object | None]]:
    """Aggregate all crew rows belonging to the selected primary aircraft.

    This is the final crew rule used by the benchmark. It avoids an arbitrary
    primary-pilot selection and represents the crew attached to the event's
    selected primary aircraft. ``has_second_pilot`` is constructed for audit
    purposes but is excluded from final feature registries because the aircraft
    table already contains ``second_pilot``.
    """
    states: dict[str, dict] = {}
    primary_keys = {
        event_id: clean_text(record.get("Aircraft_Key"))
        for event_id, record in primary_aircraft.items()
    }
    for record in records:
        event_id = clean_text(record.get("ev_id"))
        if event_id is None:
            continue
        aircraft_key = clean_text(record.get("Aircraft_Key"))
        if primary_keys.get(event_id) != aircraft_key:
            continue
        state = states.setdefault(event_id, {
            "count": 0,
            "age_sum": 0.0,
            "age_count": 0,
            "female_count": 0,
            "has_second": False,
            "med_certf": Counter(),
            "med_validity": Counter(),
        })
        state["count"] += 1
        age = to_float(record.get("crew_age"))
        if age is not None:
            state["age_sum"] += age
            state["age_count"] += 1
        if (clean_text(record.get("crew_sex")) or "").upper() == "F":
            state["female_count"] += 1
        crew_number = to_int(record.get("crew_no"))
        state["has_second"] = state["has_second"] or (crew_number is not None and crew_number >= 2)
        med = clean_text(record.get("med_certf"))
        if med:
            state["med_certf"][med] += 1
        validity = clean_text(record.get("med_crtf_vldty"))
        if validity:
            state["med_validity"][validity] += 1

    def counter_mode(counter: Counter):
        if not counter:
            return None
        maximum = max(counter.values())
        return sorted(key for key, value in counter.items() if value == maximum)[0]

    output: dict[str, dict[str, object | None]] = {}
    for event_id in primary_aircraft:
        state = states.get(event_id)
        if state is None:
            output[event_id] = {
                "crew_count": None,
                "female_crew_count": None,
                "has_second_pilot": None,
                "mean_age": None,
                "med_certf_mode": None,
                "med_crtf_vldty_mode": None,
            }
            continue
        output[event_id] = {
            "crew_count": state["count"],
            "female_crew_count": state["female_count"],
            "has_second_pilot": int(state["has_second"]),
            "mean_age": state["age_sum"] / state["age_count"] if state["age_count"] else None,
            "med_certf_mode": counter_mode(state["med_certf"]),
            "med_crtf_vldty_mode": counter_mode(state["med_validity"]),
        }
    return output

def build_analytical_dataset(
    *,
    events_path: str | Path,
    aircraft_path: str | Path,
    crew_path: str | Path,
    output_csv: str | Path,
    event_sheet: str = "events",
    aircraft_sheet: str = "aircraft",
    crew_sheet: str = "Flight_Crew",
    start_year: int = 2008,
    end_year: int = 2025,
    preferred_aircraft_key: str = "1",
) -> dict[str, int | str]:
    aircraft = select_primary_aircraft(
        iter_dicts(aircraft_path, aircraft_sheet), preferred_aircraft_key
    )
    crew = aggregate_crew(iter_dicts(crew_path, crew_sheet), aircraft)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    event_rows = iter_dicts(events_path, event_sheet)
    first_event = next(event_rows)
    event_headers = list(first_event.keys())
    aircraft_headers = [
        header for header in next(iter_dicts(aircraft_path, aircraft_sheet)).keys()
        if header not in set(event_headers) | {"ev_id", "ntsb_no"}
    ]
    crew_headers = [
        "crew_count",
        "female_crew_count",
        "has_second_pilot",
        "mean_age",
        "med_certf_mode",
        "med_crtf_vldty_mode",
    ]
    engineered_headers = [
        "acft_age",
        "days_since_last_inspection",
        "ev_month_sin", "ev_month_cos",
        "ev_time_sin", "ev_time_cos",
        "dprt_time_sin", "dprt_time_cos",
        "wx_obs_time_sin", "wx_obs_time_cos",
        "apt_dir_sin", "apt_dir_cos",
        "wind_dir_sin", "wind_dir_cos",
        "wx_obs_dir_sin", "wx_obs_dir_cos",
    ]
    output_headers = event_headers + aircraft_headers + crew_headers + engineered_headers + ["severe"]

    written = 0
    missing_aircraft = 0
    missing_crew = 0
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_headers)
        writer.writeheader()
        for event in itertools.chain([first_event], event_rows):
            year = to_int(event.get("ev_year"))
            if year is None or year < start_year or year > end_year:
                continue
            event_id = clean_text(event.get("ev_id"))
            if event_id is None:
                continue
            row = dict(event)
            aircraft_row = aircraft.get(event_id)
            if aircraft_row is None:
                missing_aircraft += 1
                aircraft_row = {}
            for header in aircraft_headers:
                row[header] = aircraft_row.get(header)
            crew_row = crew.get(event_id)
            if crew_row is None or crew_row.get("crew_count") is None:
                missing_crew += 1
                crew_row = crew_row or {}
            row.update(crew_row)
            row.update(engineer_features(row))
            row["severe"] = severe_outcome(event)
            writer.writerow(row)
            written += 1
    return {
        "events_written": written,
        "events_missing_primary_aircraft": missing_aircraft,
        "events_missing_primary_aircraft_crew": missing_crew,
        "crew_rule": "aggregate_all_crew_for_selected_primary_aircraft",
    }
