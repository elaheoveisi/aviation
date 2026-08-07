import math

from ntsb_severity.transformations import (
    aircraft_age,
    cyclic_degrees,
    cyclic_hhmm,
    cyclic_month,
    elapsed_days,
    hhmm_minutes,
)


def test_aircraft_age_and_elapsed_days():
    assert aircraft_age("2020", "2000") == 20.0
    assert aircraft_age("2000", "2020") is None
    assert elapsed_days("40000", "39900") == 100.0
    assert elapsed_days("39900", "40000") is None


def test_hhmm_validation_and_midnight():
    assert hhmm_minutes("0000") == 0
    assert hhmm_minutes("2400") == 0
    assert hhmm_minutes("1260") is None
    assert hhmm_minutes("2500") is None
    pair = cyclic_hhmm("0600", "time")
    assert math.isclose(pair["time_sin"], 1.0, abs_tol=1e-10)
    assert math.isclose(pair["time_cos"], 0.0, abs_tol=1e-10)


def test_cyclic_month_and_degrees():
    january = cyclic_month("1")
    assert math.isclose(january["ev_month_sin"], 0.0, abs_tol=1e-10)
    assert math.isclose(january["ev_month_cos"], 1.0, abs_tol=1e-10)
    north = cyclic_degrees("360", "direction")
    assert math.isclose(north["direction_sin"], 0.0, abs_tol=1e-10)
    assert math.isclose(north["direction_cos"], 1.0, abs_tol=1e-10)
