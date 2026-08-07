from pathlib import Path
from ntsb_severity.construct import build_analytical_dataset
from ntsb_severity.analytical_audit import audit_analytical_dataset

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "outputs" / "analytical_dataset.csv"
print(build_analytical_dataset(
    events_path="/mnt/data/events.xlsx",
    aircraft_path="/mnt/data/aircraft.xlsx",
    crew_path="/mnt/data/Flight_Crew.xlsx",
    output_csv=DATASET,
))
audit_analytical_dataset(
    DATASET,
    ROOT / "config" / "final_features_early_enriched.yaml",
    ROOT / "outputs" / "audit" / "14_final_main_feature_missingness.csv",
)
audit_analytical_dataset(
    DATASET,
    ROOT / "config" / "final_features_strict_initial.yaml",
    ROOT / "outputs" / "audit" / "15_final_strict_feature_missingness.csv",
)
