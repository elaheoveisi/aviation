from pathlib import Path
from ntsb_severity.benchmark import run_benchmark

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "outputs" / "analytical_dataset.csv"
CONFIG = ROOT / "config" / "default.yaml"

print(run_benchmark(
    dataset_csv=DATASET,
    config_path=CONFIG,
    feature_config_path=ROOT / "config" / "final_features_early_enriched.yaml",
    output_dir=ROOT / "outputs" / "benchmark_main",
    feature_set_label="early_enriched_main",
))
print(run_benchmark(
    dataset_csv=DATASET,
    config_path=CONFIG,
    feature_config_path=ROOT / "config" / "final_features_strict_initial.yaml",
    output_dir=ROOT / "outputs" / "benchmark_strict",
    feature_set_label="strict_initial_sensitivity",
))
