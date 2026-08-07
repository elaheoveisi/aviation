import numpy as np
from ntsb_severity.metrics import evaluate_binary, threshold_for_max_fnr


def test_metrics_confusion_counts():
    result = evaluate_binary([0, 0, 1, 1], [0.1, 0.8, 0.3, 0.9], 0.5)
    assert (result.tn, result.fp, result.fn, result.tp) == (1, 1, 1, 1)
    assert result.fnr == 0.5


def test_threshold_is_selected_without_test_information():
    threshold = threshold_for_max_fnr(
        np.array([0, 0, 1, 1, 1]),
        np.array([0.1, 0.2, 0.4, 0.7, 0.9]),
        max_fnr=1/3,
    )
    assert threshold == 0.7
