from pathlib import Path
from ntsb_severity.audit import run_audit

ROOT = Path(__file__).resolve().parents[1]
print(run_audit(
    data_dir=Path("/mnt/data"),
    output_dir=ROOT / "outputs" / "audit",
    feature_config=ROOT / "config" / "features_manuscript.yaml",
))
