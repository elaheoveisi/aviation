from ntsb_severity.construct import select_primary_aircraft


def test_prefers_aircraft_key_one():
    records = [
        {"ev_id": "A", "Aircraft_Key": "2"},
        {"ev_id": "A", "Aircraft_Key": "1"},
    ]
    selected = select_primary_aircraft(records)
    assert selected["A"]["Aircraft_Key"] == "1"


def test_falls_back_to_lowest_numeric_key():
    records = [
        {"ev_id": "A", "Aircraft_Key": "3"},
        {"ev_id": "A", "Aircraft_Key": "2"},
    ]
    selected = select_primary_aircraft(records)
    assert selected["A"]["Aircraft_Key"] == "2"
