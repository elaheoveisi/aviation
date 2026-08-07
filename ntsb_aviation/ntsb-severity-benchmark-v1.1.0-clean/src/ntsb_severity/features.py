from __future__ import annotations

from pathlib import Path
import yaml


def load_feature_config(path: str | Path) -> tuple[list[str], list[str]]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    numeric = list(config.get("numeric", []))
    categorical = list(config.get("categorical", []))
    overlap = set(numeric) & set(categorical)
    if overlap:
        raise ValueError(f"Features listed as both numeric and categorical: {sorted(overlap)}")
    return numeric, categorical
