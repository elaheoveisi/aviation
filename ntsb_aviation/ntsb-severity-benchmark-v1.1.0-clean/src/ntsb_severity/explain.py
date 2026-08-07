from __future__ import annotations

from collections import defaultdict
import numpy as np
import shap

from .preprocessing import transformed_feature_names


def _parent_feature(transformed_name: str, numeric_names: list[str], categorical_names: list[str]) -> str:
    if transformed_name.startswith("num__"):
        return transformed_name.removeprefix("num__")
    encoded = transformed_name.removeprefix("cat__")
    # Match the longest configured name to avoid prefix ambiguity.
    matches = [name for name in categorical_names if encoded == name or encoded.startswith(name + "_")]
    return max(matches, key=len) if matches else encoded


def aggregate_xgboost_shap(
    fitted_pipeline,
    X,
    *,
    numeric_names: list[str],
    categorical_names: list[str],
    sample_size: int = 2000,
    seed: int = 42,
) -> list[dict]:
    """Return predictor-level mean absolute SHAP values.

    The fitted pipeline must contain ``preprocess`` and ``model`` steps. One-hot
    columns are aggregated back to their original predictor.
    """
    rng = np.random.default_rng(seed)
    if len(X) > sample_size:
        indices = rng.choice(len(X), size=sample_size, replace=False)
        X = X[indices]
    preprocessor = fitted_pipeline.named_steps["preprocess"]
    model = fitted_pipeline.named_steps["model"]
    transformed = preprocessor.transform(X)
    names = transformed_feature_names(preprocessor, numeric_names, categorical_names)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(transformed)
    if isinstance(values, list):
        values = values[-1]
    mean_abs = np.asarray(np.abs(values).mean(axis=0)).ravel()
    aggregated: dict[str, float] = defaultdict(float)
    for name, importance in zip(names, mean_abs):
        aggregated[_parent_feature(name, numeric_names, categorical_names)] += float(importance)
    return [
        {"feature": feature, "mean_abs_shap": importance}
        for feature, importance in sorted(aggregated.items(), key=lambda item: item[1], reverse=True)
    ]


def run_shap_analysis(
    *,
    dataset_csv,
    config_path,
    feature_config_path,
    output_dir,
):
    """Refit corrected XGBoost models and write random/temporal SHAP tables."""
    from pathlib import Path
    from sklearn.model_selection import train_test_split
    from .benchmark import load_config, load_matrix, _group_rare_country
    from .features import load_feature_config
    from .models import build_model
    from .utils import write_csv

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    numeric, categorical = load_feature_config(feature_config_path)
    X, y, years, *_ = load_matrix(dataset_csv, numeric, categorical)
    seed = int(config["project"]["random_seed"])
    indices = np.arange(len(y))
    random_train, random_test = train_test_split(
        indices, test_size=0.2, random_state=seed, stratify=y
    )
    temporal_train = np.flatnonzero(years <= int(config["validation"]["primary_temporal_cut"]))
    temporal_test = np.flatnonzero(years > int(config["validation"]["primary_temporal_cut"]))
    features = numeric + categorical
    for design, train_indices, test_indices in (
        ("random", random_train, random_test),
        ("temporal", temporal_train, temporal_test),
    ):
        X_train, [X_test], _ = _group_rare_country(
            X[train_indices], [X[test_indices]], features
        )
        model = build_model(
            "xgboost",
            n_numeric=len(numeric),
            n_categorical=len(categorical),
            seed=seed,
            config=config["models"]["xgboost"],
        )
        model.fit(X_train, y[train_indices])
        rows = aggregate_xgboost_shap(
            model,
            X_test,
            numeric_names=numeric,
            categorical_names=categorical,
            sample_size=int(config["reporting"]["shap_sample_size"]),
            seed=seed,
        )
        write_csv(output_dir / f"shap_{design}_xgboost.csv", rows)
