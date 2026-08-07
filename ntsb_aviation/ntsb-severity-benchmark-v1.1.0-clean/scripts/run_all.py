from pathlib import Path
import argparse
import os

from ntsb_severity.audit import run_audit
from ntsb_severity.analytical_audit import audit_analytical_dataset
from ntsb_severity.construct import build_analytical_dataset
from ntsb_severity.benchmark import run_benchmark
from ntsb_severity.explain import run_shap_analysis
from ntsb_severity.dictionary_audit import run_dictionary_audit
from ntsb_severity.reporting import (
    plot_annual_severity,
    plot_calibration_curves,
    plot_model_curves,
    plot_shap_importance,
    plot_threshold_tradeoff,
    plot_validation_metric_comparison,
)

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_FEATURES = ROOT / "config" / "features_manuscript.yaml"
MAIN_FEATURES = ROOT / "config" / "final_features_early_enriched.yaml"
STRICT_FEATURES = ROOT / "config" / "final_features_strict_initial.yaml"
CONFIG = ROOT / "config" / "default.yaml"
AUDIT = ROOT / "outputs" / "audit"
MAIN = ROOT / "outputs" / "benchmark_main"
STRICT = ROOT / "outputs" / "benchmark_strict"
FIGURES = ROOT / "outputs" / "figures_final"
DATASET = ROOT / "outputs" / "analytical_dataset.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Run the complete NTSB severity benchmark workflow.")
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("NTSB_DATA_DIR", str(ROOT / "data" / "raw")),
        help="Directory containing the raw NTSB Excel workbooks.",
    )
    return parser.parse_args()


ARGS = parse_args()
DATA = Path(ARGS.data_dir).expanduser().resolve()

AUDIT.mkdir(parents=True, exist_ok=True)
MAIN.mkdir(parents=True, exist_ok=True)
STRICT.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

run_audit(data_dir=DATA, output_dir=AUDIT, feature_config=ORIGINAL_FEATURES)
construction = build_analytical_dataset(
    events_path=DATA / "events.xlsx",
    aircraft_path=DATA / "aircraft.xlsx",
    crew_path=DATA / "Flight_Crew.xlsx",
    output_csv=DATASET,
)
(ROOT / "outputs" / "construction_summary.txt").write_text(
    "\n".join(f"{key}: {value}" for key, value in construction.items()),
    encoding="utf-8",
)
audit_analytical_dataset(DATASET, MAIN_FEATURES, AUDIT / "14_final_main_feature_missingness.csv")
audit_analytical_dataset(DATASET, STRICT_FEATURES, AUDIT / "15_final_strict_feature_missingness.csv")
run_dictionary_audit(
    dictionary_path=DATA / "eADMSPUB_DataDictionary.xlsx",
    registry_path=ROOT / "config" / "feature_timing_registry.yaml",
    missingness_csv=AUDIT / "12_feature_missingness_2008_2025.csv",
    output_csv=AUDIT / "13_predictor_timing_registry_regenerated.csv",
)

run_benchmark(
    dataset_csv=DATASET,
    config_path=CONFIG,
    feature_config_path=MAIN_FEATURES,
    output_dir=MAIN,
    feature_set_label="early_enriched_main",
)
run_benchmark(
    dataset_csv=DATASET,
    config_path=CONFIG,
    feature_config_path=STRICT_FEATURES,
    output_dir=STRICT,
    feature_set_label="strict_initial_sensitivity",
)
run_shap_analysis(
    dataset_csv=DATASET,
    config_path=CONFIG,
    feature_config_path=MAIN_FEATURES,
    output_dir=MAIN,
)

plot_annual_severity(AUDIT / "04_year_distribution.csv", FIGURES / "figure_1_annual_severity_rate.png")
for label, directory in (("main", MAIN), ("strict", STRICT)):
    plot_model_curves({
        "Logistic regression": directory / "predictions_random_logistic_regression.csv",
        "Random forest": directory / "predictions_random_random_forest.csv",
        "XGBoost": directory / "predictions_random_xgboost.csv",
    }, FIGURES / f"figure_{label}_random")
    plot_calibration_curves({
        "Logistic regression": directory / "predictions_random_logistic_regression.csv",
        "Random forest": directory / "predictions_random_random_forest.csv",
        "XGBoost": directory / "predictions_random_xgboost.csv",
    }, FIGURES / f"figure_{label}_random_calibration.png", title=f"{label.title()} feature set: random-split calibration")
    plot_calibration_curves({
        "Logistic regression": directory / "predictions_temporal_logistic_regression.csv",
        "Random forest": directory / "predictions_temporal_random_forest.csv",
        "XGBoost": directory / "predictions_temporal_xgboost.csv",
    }, FIGURES / f"figure_{label}_temporal_calibration.png", title=f"{label.title()} feature set: temporal calibration")
    plot_validation_metric_comparison(
        directory / "model_performance.csv",
        FIGURES / f"figure_{label}_roc_auc_random_vs_temporal.png",
        metric="roc_auc",
        title=f"{label.title()} feature set: random versus temporal ROC-AUC",
    )
    plot_validation_metric_comparison(
        directory / "model_performance.csv",
        FIGURES / f"figure_{label}_pr_auc_random_vs_temporal.png",
        metric="pr_auc",
        title=f"{label.title()} feature set: random versus temporal PR-AUC",
    )
    plot_threshold_tradeoff(
        directory / "threshold_sensitivity_xgboost.csv",
        FIGURES / f"figure_{label}_threshold_tradeoff.png",
        title=f"{label.title()} XGBoost: miss rate versus review workload",
    )

plot_shap_importance(MAIN / "shap_random_xgboost.csv", FIGURES / "figure_main_shap_random.png", title="Main XGBoost: random-split SHAP importance")
plot_shap_importance(MAIN / "shap_temporal_xgboost.csv", FIGURES / "figure_main_shap_temporal.png", title="Main XGBoost: temporal SHAP importance")
