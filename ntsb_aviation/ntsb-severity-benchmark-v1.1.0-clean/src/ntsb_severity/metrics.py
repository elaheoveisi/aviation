from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable
import math
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class MetricResult:
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    roc_auc: float
    pr_auc: float
    brier: float
    mcc: float
    fnr: float
    flag_rate: float
    tn: int
    fp: int
    fn: int
    tp: int
    threshold: float

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_binary(y_true, probabilities, threshold: float = 0.5) -> MetricResult:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else math.nan
    fnr = fn / (fn + tp) if (fn + tp) else math.nan
    return MetricResult(
        accuracy=float(accuracy_score(y_true, predictions)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, predictions)),
        precision=float(precision_score(y_true, predictions, zero_division=0)),
        recall=float(recall_score(y_true, predictions, zero_division=0)),
        specificity=float(specificity),
        f1=float(f1_score(y_true, predictions, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, probabilities)),
        pr_auc=float(average_precision_score(y_true, probabilities)),
        brier=float(brier_score_loss(y_true, probabilities)),
        mcc=float(matthews_corrcoef(y_true, predictions)),
        fnr=float(fnr),
        flag_rate=float(predictions.mean()),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        threshold=float(threshold),
    )


def threshold_for_max_fnr(y_true, probabilities, max_fnr: float = 0.20) -> float:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    # Highest observed threshold that satisfies the miss-rate target.
    candidates = np.unique(np.concatenate(([0.0, 1.0], probabilities)))
    valid = []
    for threshold in candidates:
        predictions = probabilities >= threshold
        positives = y_true == 1
        denominator = positives.sum()
        fnr = ((~predictions) & positives).sum() / denominator if denominator else math.nan
        if not math.isnan(fnr) and fnr <= max_fnr:
            valid.append(float(threshold))
    return max(valid) if valid else 0.0


def bootstrap_intervals(
    y_true,
    probabilities,
    *,
    threshold: float,
    repetitions: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, tuple[float, float]]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    roc_values = []
    fnr_values = []
    for _ in range(repetitions):
        indices = rng.integers(0, n, n)
        y_sample = y_true[indices]
        p_sample = probabilities[indices]
        if np.unique(y_sample).size < 2:
            continue
        roc_values.append(roc_auc_score(y_sample, p_sample))
        predictions = p_sample >= threshold
        positives = y_sample == 1
        fnr_values.append(((~predictions) & positives).sum() / positives.sum())
    lower = 100 * alpha / 2
    upper = 100 * (1 - alpha / 2)
    return {
        "roc_auc": (float(np.percentile(roc_values, lower)), float(np.percentile(roc_values, upper))),
        "fnr": (float(np.percentile(fnr_values, lower)), float(np.percentile(fnr_values, upper))),
    }


def threshold_table(y_true, probabilities, thresholds: Iterable[float]) -> list[dict]:
    rows = []
    for threshold in thresholds:
        metrics = evaluate_binary(y_true, probabilities, float(threshold))
        rows.append({
            "threshold": float(threshold),
            "fnr": metrics.fnr,
            "flag_rate": metrics.flag_rate,
            "precision": metrics.precision,
            "recall": metrics.recall,
        })
    return rows
