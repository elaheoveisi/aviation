import numpy as np

from ntsb_severity.benchmark import correct_for_oversampling_prior


def test_prior_correction_reduces_probabilities_when_true_prevalence_is_lower():
    raw = np.asarray([0.2, 0.5, 0.8])
    corrected = correct_for_oversampling_prior(raw, original_prevalence=0.30)
    assert np.all(corrected < raw)
    assert np.all((corrected > 0) & (corrected < 1))


def test_no_change_when_priors_match():
    raw = np.asarray([0.2, 0.5, 0.8])
    corrected = correct_for_oversampling_prior(
        raw, original_prevalence=0.5, sampled_prevalence=0.5
    )
    assert np.allclose(corrected, raw)
