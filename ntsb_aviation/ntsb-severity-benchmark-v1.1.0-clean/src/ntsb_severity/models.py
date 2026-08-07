from __future__ import annotations

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from .preprocessing import make_preprocessor


def build_model(name: str, *, n_numeric: int, n_categorical: int, seed: int, config: dict):
    if name == "logistic_regression":
        estimator = LogisticRegression(
            C=float(config.get("C", 0.5)),
            max_iter=int(config.get("max_iter", 1000)),
            penalty=config.get("penalty", "l2"),
            solver="liblinear",
            random_state=seed,
        )
        preprocessor = make_preprocessor(n_numeric, n_categorical, scale_numeric=True)
    elif name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=int(config.get("n_estimators", 120)),
            max_depth=config.get("max_depth", 20),
            min_samples_leaf=int(config.get("min_samples_leaf", 2)),
            n_jobs=int(config.get("n_jobs", -1)),
            random_state=seed,
        )
        preprocessor = make_preprocessor(n_numeric, n_categorical, scale_numeric=False)
    elif name == "xgboost":
        estimator = XGBClassifier(
            n_estimators=int(config.get("n_estimators", 120)),
            max_depth=int(config.get("max_depth", 5)),
            learning_rate=float(config.get("learning_rate", 0.08)),
            subsample=float(config.get("subsample", 0.8)),
            colsample_bytree=float(config.get("colsample_bytree", 0.8)),
            eval_metric=config.get("eval_metric", "logloss"),
            n_jobs=int(config.get("n_jobs", -1)),
            random_state=seed,
            tree_method="hist",
        )
        preprocessor = make_preprocessor(n_numeric, n_categorical, scale_numeric=False)
    else:
        raise ValueError(f"Unknown model: {name}")
    return ImbPipeline([
        ("preprocess", preprocessor),
        ("oversample", RandomOverSampler(random_state=seed)),
        ("model", estimator),
    ])
