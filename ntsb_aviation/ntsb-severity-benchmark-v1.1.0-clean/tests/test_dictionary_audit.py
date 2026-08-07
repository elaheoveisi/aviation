from ntsb_severity.dictionary_audit import ALIASES


def test_required_derived_aliases_are_documented():
    expected = {
        "dec_latitude", "dec_longitude", "crew_count", "female_crew_count",
        "mean_age", "has_second_pilot", "med_certf_mode", "med_crtf_vldty_mode",
    }
    assert expected == set(ALIASES)
