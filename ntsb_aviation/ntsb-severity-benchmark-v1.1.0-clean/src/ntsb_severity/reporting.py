from __future__ import annotations

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score


def _read(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_annual_severity(year_csv: str | Path, output_png: str | Path) -> None:
    rows = _read(year_csv)
    years = [int(row["year"]) for row in rows]
    rates = [float(row["severe_rate"]) * 100 for row in rows]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(years, rates, marker="o")
    ax.set_xlabel("Year")
    ax.set_ylabel("Severe events (%)")
    ax.set_title("Annual severe-event rate")
    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def plot_model_curves(prediction_files: dict[str, str | Path], output_prefix: str | Path) -> None:
    output_prefix = Path(output_prefix)
    fig, ax = plt.subplots(figsize=(8, 7))
    for label, path in prediction_files.items():
        rows = _read(path)
        y = np.asarray([int(row["y_true"]) for row in rows])
        p = np.asarray([float(row["probability"]) for row in rows])
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_prefix.with_name(output_prefix.name + "_roc.png"), dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    for label, path in prediction_files.items():
        rows = _read(path)
        y = np.asarray([int(row["y_true"]) for row in rows])
        p = np.asarray([float(row["probability"]) for row in rows])
        precision, recall, _ = precision_recall_curve(y, p)
        ax.plot(recall, precision, label=f"{label} (AP={average_precision_score(y, p):.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_prefix.with_name(output_prefix.name + "_pr.png"), dpi=300)
    plt.close(fig)


def plot_calibration_curves(
    prediction_files: dict[str, str | Path],
    output_png: str | Path,
    *,
    title: str,
    n_bins: int = 10,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    for label, path in prediction_files.items():
        rows = _read(path)
        y = np.asarray([int(row["y_true"]) for row in rows])
        p = np.asarray([float(row["probability"]) for row in rows])
        observed, predicted = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
        ax.plot(predicted, observed, marker="o", label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed severe-event proportion")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def plot_validation_metric_comparison(
    performance_csv: str | Path,
    output_png: str | Path,
    *,
    metric: str,
    title: str,
) -> None:
    rows = [
        row for row in _read(performance_csv)
        if row.get("operating_point") == "default"
    ]
    models = ["logistic_regression", "random_forest", "xgboost"]
    labels = ["Logistic regression", "Random forest", "XGBoost"]
    random_values = []
    temporal_values = []
    for model in models:
        random_row = next(row for row in rows if row["model"] == model and row["validation_design"] == "random")
        temporal_row = next(row for row in rows if row["model"] == model and row["validation_design"] == "temporal")
        random_values.append(float(random_row[metric]))
        temporal_values.append(float(temporal_row[metric]))
    x = np.arange(len(models))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - width / 2, random_values, width, label="Random split")
    ax.bar(x + width / 2, temporal_values, width, label="Temporal split")
    ax.set_xticks(x, labels)
    ax.set_ylabel(metric.replace("_", " ").upper())
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def plot_threshold_tradeoff(
    threshold_csv: str | Path,
    output_png: str | Path,
    *,
    title: str,
) -> None:
    rows = _read(threshold_csv)
    fig, ax = plt.subplots(figsize=(8, 7))
    for design, label in (("random", "Random split"), ("temporal", "Temporal split")):
        subset = sorted(
            (row for row in rows if row["validation_design"] == design),
            key=lambda row: float(row["threshold"]),
        )
        ax.plot(
            [float(row["flag_rate"]) * 100 for row in subset],
            [float(row["fnr"]) * 100 for row in subset],
            marker="o",
            label=label,
        )
    ax.set_xlabel("Events flagged for review (%)")
    ax.set_ylabel("False negative rate (%)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def plot_shap_importance(
    shap_csv: str | Path,
    output_png: str | Path,
    *,
    top_n: int = 15,
    title: str = "XGBoost SHAP importance",
) -> None:
    rows = _read(shap_csv)[:top_n]
    labels = [row["feature"] for row in rows][::-1]
    values = [float(row["mean_abs_shap"]) for row in rows][::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(labels, values)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    plt.close(fig)
