from pathlib import Path
from ntsb_severity.dictionary_audit import run_dictionary_audit

ROOT = Path(__file__).resolve().parents[1]
run_dictionary_audit(
    dictionary_path="/mnt/data/eADMSPUB_DataDictionary.xlsx",
    registry_path=ROOT / "config/feature_timing_registry.yaml",
    missingness_csv=ROOT / "outputs/audit/12_feature_missingness_2008_2025.csv",
    output_csv=ROOT / "outputs/audit/13_predictor_timing_registry_regenerated.csv",
)
