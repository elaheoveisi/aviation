from pathlib import Path
from ntsb_severity.explain import run_shap_analysis

ROOT = Path(__file__).resolve().parents[1]
run_shap_analysis(
    dataset_csv=ROOT / "outputs" / "analytical_dataset.csv",
    config_path=ROOT / "config" / "default.yaml",
    feature_config_path=ROOT / "config" / "final_features_early_enriched.yaml",
    output_dir=ROOT / "outputs" / "benchmark_main",
)
