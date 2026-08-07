from pathlib import Path

from ntsb_severity.features import load_feature_config

ROOT = Path(__file__).resolve().parents[1]


def test_final_registries_exclude_audited_metadata_and_raw_cyclic_fields():
    banned = {
        "latlong_acq", "wx_src_iic", "has_second_pilot", "apt_dir",
        "acft_year", "date_last_insp", "ev_month", "ev_time",
        "dprt_time", "wind_dir_deg", "wx_obs_dir", "wx_obs_time",
    }
    for name in ("final_features_early_enriched.yaml", "final_features_strict_initial.yaml"):
        numeric, categorical = load_feature_config(ROOT / "config" / name)
        features = set(numeric + categorical)
        assert not (features & banned)
        assert "acft_age" in features
        assert "ev_month_sin" in features
        assert "ev_month_cos" in features
        assert "apt_dir_sin" in features
        assert "apt_dir_cos" in features


def test_strict_is_subset_of_main():
    main_num, main_cat = load_feature_config(ROOT / "config" / "final_features_early_enriched.yaml")
    strict_num, strict_cat = load_feature_config(ROOT / "config" / "final_features_strict_initial.yaml")
    assert set(strict_num).issubset(set(main_num))
    assert set(strict_cat).issubset(set(main_cat))
