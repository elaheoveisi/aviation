from pathlib import Path
from ntsb_severity.features import load_feature_config


def test_supplied_appendix_registry_has_69_features():
    root = Path(__file__).resolve().parents[1]
    numeric, categorical = load_feature_config(root / "config" / "features_manuscript.yaml")
    assert len(numeric) == 34
    assert len(categorical) == 35
    assert len(numeric) + len(categorical) == 69
    assert "ev_type" not in numeric + categorical
    assert "damage" not in numeric + categorical
    assert "elt_oper" not in numeric + categorical
