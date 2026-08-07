from pathlib import Path
from ntsb_severity.reporting import (
    plot_annual_severity,
    plot_calibration_curves,
    plot_model_curves,
    plot_shap_importance,
    plot_threshold_tradeoff,
    plot_validation_metric_comparison,
)

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs" / "audit"
MAIN = ROOT / "outputs" / "benchmark_main"
STRICT = ROOT / "outputs" / "benchmark_strict"
FIGURES = ROOT / "outputs" / "figures_final"
FIGURES.mkdir(parents=True, exist_ok=True)

plot_annual_severity(AUDIT / "04_year_distribution.csv", FIGURES / "figure_1_annual_severity_rate.png")
for label, directory in (("main", MAIN), ("strict", STRICT)):
    random_files = {
        "Logistic regression": directory / "predictions_random_logistic_regression.csv",
        "Random forest": directory / "predictions_random_random_forest.csv",
        "XGBoost": directory / "predictions_random_xgboost.csv",
    }
    temporal_files = {
        "Logistic regression": directory / "predictions_temporal_logistic_regression.csv",
        "Random forest": directory / "predictions_temporal_random_forest.csv",
        "XGBoost": directory / "predictions_temporal_xgboost.csv",
    }
    plot_model_curves(random_files, FIGURES / f"figure_{label}_random")
    plot_calibration_curves(random_files, FIGURES / f"figure_{label}_random_calibration.png", title=f"{label.title()} feature set: random-split calibration")
    plot_calibration_curves(temporal_files, FIGURES / f"figure_{label}_temporal_calibration.png", title=f"{label.title()} feature set: temporal calibration")
    plot_validation_metric_comparison(directory / "model_performance.csv", FIGURES / f"figure_{label}_roc_auc_random_vs_temporal.png", metric="roc_auc", title=f"{label.title()} feature set: random versus temporal ROC-AUC")
    plot_validation_metric_comparison(directory / "model_performance.csv", FIGURES / f"figure_{label}_pr_auc_random_vs_temporal.png", metric="pr_auc", title=f"{label.title()} feature set: random versus temporal PR-AUC")
    plot_threshold_tradeoff(directory / "threshold_sensitivity_xgboost.csv", FIGURES / f"figure_{label}_threshold_tradeoff.png", title=f"{label.title()} XGBoost: miss rate versus review workload")

plot_shap_importance(MAIN / "shap_random_xgboost.csv", FIGURES / "figure_main_shap_random.png", title="Main XGBoost: random-split SHAP importance")
plot_shap_importance(MAIN / "shap_temporal_xgboost.csv", FIGURES / "figure_main_shap_temporal.png", title="Main XGBoost: temporal SHAP importance")
