from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def make_preprocessor(n_numeric: int, n_categorical: int, *, scale_numeric: bool):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler(with_mean=False)))
    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(missing_values=None, strategy="constant", fill_value="Missing")),
        ("onehot", _encoder()),
    ])
    numeric_indices = list(range(n_numeric))
    categorical_indices = list(range(n_numeric, n_numeric + n_categorical))
    return ColumnTransformer([
        ("num", numeric_pipeline, numeric_indices),
        ("cat", categorical_pipeline, categorical_indices),
    ], remainder="drop", sparse_threshold=0.3)


def transformed_feature_names(preprocessor, numeric_names, categorical_names) -> list[str]:
    names = list(numeric_names)
    cat_pipe = preprocessor.named_transformers_["cat"]
    encoder = cat_pipe.named_steps["onehot"]
    encoded = list(encoder.get_feature_names_out(categorical_names))
    return [f"num__{name}" for name in names] + [f"cat__{name}" for name in encoded]
