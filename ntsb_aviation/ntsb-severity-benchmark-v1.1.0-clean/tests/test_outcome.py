from ntsb_severity.construct import severe_outcome


def test_severe_for_fatal_injury():
    assert severe_outcome({"inj_tot_f": "1", "inj_tot_s": "0"}) == 1


def test_severe_for_serious_injury():
    assert severe_outcome({"inj_tot_f": "0", "inj_tot_s": "2"}) == 1


def test_nonsevere_when_no_fatal_or_serious():
    assert severe_outcome({"inj_tot_f": "0", "inj_tot_s": "0"}) == 0
