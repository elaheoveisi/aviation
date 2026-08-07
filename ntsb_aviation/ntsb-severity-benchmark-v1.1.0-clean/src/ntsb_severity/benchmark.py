from __future__ import annotations

from collections import Counter
from pathlib import Path
import csv
import json
import math
import gc

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

from .features import load_feature_config
from .metrics import bootstrap_intervals, evaluate_binary, threshold_for_max_fnr, threshold_table
from .models import build_model
from .utils import clean_text, to_float, to_int, write_csv


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def correct_for_oversampling_prior(
    probabilities,
    *,
    original_prevalence: float,
    sampled_prevalence: float = 0.5,
):
    """Correct posterior probabilities for the prevalence changed by oversampling.

    Random oversampling balances the model-fitting sample, which changes the class
    prior seen by the classifier. Under prior-probability shift, the fitted odds
    can be returned to the original training prevalence without using test data.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    eps = np.finfo(float).eps
    p = np.clip(probabilities, eps, 1.0 - eps)
    original = float(np.clip(original_prevalence, eps, 1.0 - eps))
    sampled = float(np.clip(sampled_prevalence, eps, 1.0 - eps))
    odds = p / (1.0 - p)
    prior_ratio = (original / (1.0 - original)) / (sampled / (1.0 - sampled))
    corrected_odds = odds * prior_ratio
    return corrected_odds / (1.0 + corrected_odds)


def bootstrap_fnr_interval(y_true, probabilities, *, threshold=0.5, repetitions=1000, seed=42):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(int(repetitions)):
        idx = rng.integers(0, len(y_true), len(y_true))
        y_sample = y_true[idx]
        positives = y_sample == 1
        if positives.sum() == 0:
            continue
        pred = probabilities[idx] >= threshold
        values.append(float(((~pred) & positives).sum() / positives.sum()))
    if not values:
        return math.nan, math.nan
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def load_matrix(
    dataset_csv: str | Path,
    numeric_features: list[str],
    categorical_features: list[str],
):
    rows = []
    targets = []
    years = []
    event_ids = []
    subgroups = []
    with Path(dataset_csv).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in numeric_features + categorical_features if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Dataset is missing configured features: {missing}")
        for row in reader:
            values = []
            for name in numeric_features:
                value = to_float(row.get(name))
                values.append(np.nan if value is None else value)
            for name in categorical_features:
                values.append(clean_text(row.get(name)))
            rows.append(values)
            targets.append(int(row["severe"]))
            years.append(to_int(row.get("ev_year")))
            event_ids.append(row.get("ev_id"))
            subgroups.append(clean_text(row.get("far_part")))
    return (
        np.asarray(rows, dtype=object),
        np.asarray(targets, dtype=int),
        np.asarray(years, dtype=int),
        np.asarray(event_ids, dtype=object),
        np.asarray(subgroups, dtype=object),
    )


def _group_rare_country(
    X_train: np.ndarray,
    X_others: list[np.ndarray],
    all_features: list[str],
    *,
    minimum_count: int = 50,
):
    if "ev_country" not in all_features:
        return X_train.copy(), [array.copy() for array in X_others], set()
    index = all_features.index("ev_country")
    counts = Counter(value for value in X_train[:, index] if value is not None)
    retained = {value for value, count in counts.items() if count >= minimum_count}

    def transform(array):
        output = array.copy()
        for row_index, value in enumerate(output[:, index]):
            if value is not None and value not in retained:
                output[row_index, index] = "Other"
        return output

    return transform(X_train), [transform(array) for array in X_others], retained


def _fit_evaluate(
    *,
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    numeric_features: list[str],
    categorical_features: list[str],
    model_config: dict,
    seed: int,
    dev_fraction: float,
    target_max_fnr: float,
    bootstrap_repetitions: int,
):
    train_main, dev = train_test_split(
        train_indices,
        test_size=dev_fraction,
        random_state=seed,
        stratify=y[train_indices],
    )
    features = numeric_features + categorical_features
    X_train_main, [X_dev], _ = _group_rare_country(
        X[train_main], [X[dev]], features
    )
    development_model = build_model(
        model_name,
        n_numeric=len(numeric_features),
        n_categorical=len(categorical_features),
        seed=seed,
        config=model_config,
    )
    development_model.fit(X_train_main, y[train_main])
    development_probabilities_raw = development_model.predict_proba(X_dev)[:, 1]
    development_probabilities = correct_for_oversampling_prior(
        development_probabilities_raw,
        original_prevalence=float(y[train_main].mean()),
    )
    safety_threshold = threshold_for_max_fnr(
        y[dev], development_probabilities, max_fnr=target_max_fnr
    )

    X_train, [X_test], retained_countries = _group_rare_country(
        X[train_indices], [X[test_indices]], features
    )
    final_model = build_model(
        model_name,
        n_numeric=len(numeric_features),
        n_categorical=len(categorical_features),
        seed=seed,
        config=model_config,
    )
    final_model.fit(X_train, y[train_indices])
    test_probabilities_raw = final_model.predict_proba(X_test)[:, 1]
    test_probabilities = correct_for_oversampling_prior(
        test_probabilities_raw,
        original_prevalence=float(y[train_indices].mean()),
    )
    default_metrics = evaluate_binary(y[test_indices], test_probabilities, 0.5)
    safety_metrics = evaluate_binary(y[test_indices], test_probabilities, safety_threshold)
    intervals = bootstrap_intervals(
        y[test_indices],
        test_probabilities,
        threshold=0.5,
        repetitions=bootstrap_repetitions,
        seed=seed,
    )
    return {
        "model": final_model,
        "probabilities": test_probabilities,
        "default_metrics": default_metrics,
        "safety_metrics": safety_metrics,
        "safety_threshold": safety_threshold,
        "intervals": intervals,
        "test_indices": test_indices,
        "retained_countries": sorted(retained_countries),
        "training_prevalence": float(y[train_indices].mean()),
        "probability_adjustment": "prior_correction_after_random_oversampling",
    }


def run_benchmark(
    *,
    dataset_csv: str | Path,
    config_path: str | Path,
    feature_config_path: str | Path,
    output_dir: str | Path,
    feature_set_label: str = "unspecified",
) -> dict:
    config = load_config(config_path)
    numeric_features, categorical_features = load_feature_config(feature_config_path)
    X, y, years, event_ids, subgroups = load_matrix(
        dataset_csv, numeric_features, categorical_features
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config["project"]["random_seed"])
    validation = config["validation"]
    model_configs = config["models"]

    all_indices = np.arange(len(y))
    random_train, random_test = train_test_split(
        all_indices,
        test_size=float(validation["random_test_size"]),
        random_state=seed,
        stratify=y,
    )
    temporal_cut = int(validation["primary_temporal_cut"])
    temporal_train = np.flatnonzero(years <= temporal_cut)
    temporal_test = np.flatnonzero(years > temporal_cut)

    performance_rows = []
    ci_rows = []
    fitted = {}
    for design, train_indices, test_indices in (
        ("random", random_train, random_test),
        ("temporal", temporal_train, temporal_test),
    ):
        for model_name in ("logistic_regression", "random_forest", "xgboost"):
            result = _fit_evaluate(
                model_name=model_name,
                X=X,
                y=y,
                train_indices=train_indices,
                test_indices=test_indices,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                model_config=model_configs[model_name],
                seed=seed,
                dev_fraction=float(validation["development_size_within_training"]),
                target_max_fnr=float(validation["target_max_fnr"]),
                bootstrap_repetitions=int(validation["bootstrap_repetitions"]),
            )
            if model_name == "xgboost":
                fitted[(design, model_name)] = result
            prediction_rows = [
                {
                    "ev_id": event_ids[index],
                    "ev_year": int(years[index]),
                    "y_true": int(y[index]),
                    "probability": float(probability),
                }
                for index, probability in zip(result["test_indices"], result["probabilities"])
            ]
            write_csv(output_dir / f"predictions_{design}_{model_name}.csv", prediction_rows)
            for operating_point, metrics in (
                ("default", result["default_metrics"]),
                ("safety_target", result["safety_metrics"]),
            ):
                row = {
                    "feature_set": feature_set_label,
                    "validation_design": design,
                    "model": model_name,
                    "operating_point": operating_point,
                    "n_train": len(train_indices),
                    "n_test": len(test_indices),
                    **metrics.to_dict(),
                }
                performance_rows.append(row)
            for metric_name, (lower, upper) in result["intervals"].items():
                ci_rows.append({
                    "feature_set": feature_set_label,
                    "validation_design": design,
                    "model": model_name,
                    "metric": metric_name,
                    "estimate": getattr(result["default_metrics"], metric_name),
                    "ci_low": lower,
                    "ci_high": upper,
                    "bootstrap_repetitions": int(validation["bootstrap_repetitions"]),
                })
            if model_name != "xgboost":
                del result
                gc.collect()
    write_csv(output_dir / "model_performance.csv", performance_rows)
    write_csv(output_dir / "bootstrap_confidence_intervals.csv", ci_rows)

    # Leakage comparison on the same random split: add only the two variables
    # explicitly identified in the manuscript as inappropriate for early screening.
    leaky_numeric = list(numeric_features)
    leaky_categorical = list(categorical_features) + ["ev_type", "elt_oper"]
    X_leaky, y_leaky, _, _, _ = load_matrix(
        dataset_csv, leaky_numeric, leaky_categorical
    )
    leaky_result = _fit_evaluate(
        model_name="xgboost",
        X=X_leaky,
        y=y_leaky,
        train_indices=random_train,
        test_indices=random_test,
        numeric_features=leaky_numeric,
        categorical_features=leaky_categorical,
        model_config=model_configs["xgboost"],
        seed=seed,
        dev_fraction=float(validation["development_size_within_training"]),
        target_max_fnr=float(validation["target_max_fnr"]),
        bootstrap_repetitions=int(validation["bootstrap_repetitions"]),
    )
    corrected = fitted[("random", "xgboost")]["default_metrics"]
    leaky = leaky_result["default_metrics"]
    leakage_rows = [
        {"feature_set": feature_set_label, "configuration": "leaky_baseline", **leaky.to_dict()},
        {"feature_set": feature_set_label, "configuration": "leakage_corrected", **corrected.to_dict()},
        {
            "feature_set": feature_set_label,
            "configuration": "leaky_minus_corrected",
            "roc_auc": leaky.roc_auc - corrected.roc_auc,
            "pr_auc": leaky.pr_auc - corrected.pr_auc,
            "fnr": leaky.fnr - corrected.fnr,
            "brier": leaky.brier - corrected.brier,
        },
    ]
    write_csv(output_dir / "leakage_comparison_xgboost.csv", leakage_rows)
    write_csv(output_dir / "leakage_effect_summary.csv", [{
        "feature_set": feature_set_label,
        "roc_auc_inflation": leaky.roc_auc - corrected.roc_auc,
        "pr_auc_inflation": leaky.pr_auc - corrected.pr_auc,
        "fnr_understatement": corrected.fnr - leaky.fnr,
        "brier_optimism": corrected.brier - leaky.brier,
    }])
    del leaky_result
    gc.collect()

    # Threshold sensitivity for corrected XGBoost.
    threshold_rows = []
    for design in ("random", "temporal"):
        result = fitted[(design, "xgboost")]
        rows = threshold_table(
            y[result["test_indices"]],
            result["probabilities"],
            [0.25, 0.30, 0.35, 0.40, 0.45, result["safety_threshold"], 0.50, 0.55, 0.60],
        )
        for row in rows:
            row["feature_set"] = feature_set_label
            row["validation_design"] = design
            row["is_safety_threshold"] = math.isclose(row["threshold"], result["safety_threshold"])
            threshold_rows.append(row)
    write_csv(output_dir / "threshold_sensitivity_xgboost.csv", threshold_rows)

    # Transfer the random-split development threshold unchanged to the temporal test.
    transferred_threshold = fitted[("random", "xgboost")]["safety_threshold"]
    transferred_rows = []
    random_reference_fnr = None
    for design in ("random", "temporal"):
        result = fitted[(design, "xgboost")]
        y_test = y[result["test_indices"]]
        metrics = evaluate_binary(y_test, result["probabilities"], transferred_threshold)
        fnr_low, fnr_high = bootstrap_fnr_interval(
            y_test, result["probabilities"], threshold=transferred_threshold,
            repetitions=int(validation["bootstrap_repetitions"]), seed=seed,
        )
        if random_reference_fnr is None:
            random_reference_fnr = metrics.fnr
        transferred_rows.append({
            "feature_set": feature_set_label,
            "validation_design": design,
            "threshold_source": "random_split_training_side_development_set",
            "threshold": transferred_threshold,
            "fnr": metrics.fnr,
            "fnr_ci_low": fnr_low,
            "fnr_ci_high": fnr_high,
            "fnr_change_vs_random": metrics.fnr - random_reference_fnr,
            "flag_rate": metrics.flag_rate,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "tn": metrics.tn, "fp": metrics.fp, "fn": metrics.fn, "tp": metrics.tp,
        })
    write_csv(output_dir / "transferred_threshold_xgboost.csv", transferred_rows)

    # Rolling temporal evaluation. Test data always contain all years after the cut.
    rolling_rows = []
    for cut in validation["rolling_temporal_cuts"]:
        train_indices = np.flatnonzero(years <= int(cut))
        test_indices = np.flatnonzero(years > int(cut))
        result = _fit_evaluate(
            model_name="xgboost",
            X=X,
            y=y,
            train_indices=train_indices,
            test_indices=test_indices,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            model_config=model_configs["xgboost"],
            seed=seed,
            dev_fraction=float(validation["development_size_within_training"]),
            target_max_fnr=float(validation["target_max_fnr"]),
            bootstrap_repetitions=min(250, int(validation["bootstrap_repetitions"])),
        )
        rolling_rows.append({
            "feature_set": feature_set_label,
            "cut_year": int(cut),
            "train_period": f"2008-{int(cut)}",
            "test_period": f"{int(cut)+1}-{years.max()}",
            "n_train": len(train_indices),
            "n_test": len(test_indices),
            **result["default_metrics"].to_dict(),
        })
        del result
        gc.collect()
    write_csv(output_dir / "rolling_temporal_xgboost.csv", rolling_rows)

    # Sensitivity analysis without country/jurisdiction.
    no_country_numeric = list(numeric_features)
    no_country_categorical = [name for name in categorical_features if name != "ev_country"]
    X_nc, y_nc, years_nc, _, _ = load_matrix(
        dataset_csv, no_country_numeric, no_country_categorical
    )
    no_country_rows = []
    for design, train_indices, test_indices in (
        ("random", random_train, random_test),
        ("temporal", temporal_train, temporal_test),
    ):
        result = _fit_evaluate(
            model_name="xgboost",
            X=X_nc,
            y=y_nc,
            train_indices=train_indices,
            test_indices=test_indices,
            numeric_features=no_country_numeric,
            categorical_features=no_country_categorical,
            model_config=model_configs["xgboost"],
            seed=seed,
            dev_fraction=float(validation["development_size_within_training"]),
            target_max_fnr=float(validation["target_max_fnr"]),
            bootstrap_repetitions=min(250, int(validation["bootstrap_repetitions"])),
        )
        no_country_rows.append({
            "feature_set": feature_set_label,
            "validation_design": design,
            "configuration": "without_ev_country",
            **result["default_metrics"].to_dict(),
        })
        del result
        gc.collect()
    write_csv(output_dir / "sensitivity_without_country.csv", no_country_rows)

    # Sensitivity analysis without crew demographic/medical attributes.
    excluded_crew = {"female_crew_count", "mean_age", "med_certf_mode", "med_crtf_vldty_mode"}
    crew_sens_numeric = [name for name in numeric_features if name not in excluded_crew]
    crew_sens_categorical = [name for name in categorical_features if name not in excluded_crew]
    crew_sensitivity_rows = []
    if len(crew_sens_numeric) + len(crew_sens_categorical) < len(numeric_features) + len(categorical_features):
        X_cs, y_cs, _, _, _ = load_matrix(dataset_csv, crew_sens_numeric, crew_sens_categorical)
        for design, train_indices, test_indices in (
            ("random", random_train, random_test),
            ("temporal", temporal_train, temporal_test),
        ):
            result = _fit_evaluate(
                model_name="xgboost",
                X=X_cs, y=y_cs,
                train_indices=train_indices, test_indices=test_indices,
                numeric_features=crew_sens_numeric,
                categorical_features=crew_sens_categorical,
                model_config=model_configs["xgboost"],
                seed=seed,
                dev_fraction=float(validation["development_size_within_training"]),
                target_max_fnr=float(validation["target_max_fnr"]),
                bootstrap_repetitions=min(250, int(validation["bootstrap_repetitions"])),
            )
            crew_sensitivity_rows.append({
                "feature_set": feature_set_label,
                "validation_design": design,
                "configuration": "without_crew_demographic_medical",
                "excluded_features": ";".join(sorted(excluded_crew)),
                **result["default_metrics"].to_dict(),
            })
            del result
            gc.collect()
    write_csv(
        output_dir / "sensitivity_without_crew_demographics.csv",
        crew_sensitivity_rows,
        fieldnames=(list(crew_sensitivity_rows[0].keys()) if crew_sensitivity_rows else [
            "feature_set", "validation_design", "configuration", "excluded_features"
        ]),
    )

    # FAR subgroup performance at the conventional and transferred safety thresholds.
    subgroup_rows = []
    minimum_severe = int(config["reporting"]["subgroup_min_severe"])
    for design in ("random", "temporal"):
        result = fitted[(design, "xgboost")]
        indices = result["test_indices"]
        probabilities = result["probabilities"]
        for subgroup in sorted(set(value for value in subgroups[indices] if value is not None)):
            mask = subgroups[indices] == subgroup
            y_group = y[indices][mask]
            if int(y_group.sum()) < minimum_severe:
                continue
            p_group = probabilities[mask]
            for operating_point, threshold in (
                ("default_0.50", 0.5),
                ("transferred_random_safety_threshold", transferred_threshold),
            ):
                metrics = evaluate_binary(y_group, p_group, threshold)
                fnr_low, fnr_high = bootstrap_fnr_interval(
                    y_group, p_group, threshold=threshold,
                    repetitions=int(validation["bootstrap_repetitions"]),
                    seed=seed,
                )
                subgroup_rows.append({
                    "feature_set": feature_set_label,
                    "validation_design": design,
                    "operating_point": operating_point,
                    "threshold": threshold,
                    "far_part": subgroup,
                    "n": len(y_group),
                    "n_severe": int(y_group.sum()),
                    "fnr": metrics.fnr,
                    "fnr_ci_low": fnr_low,
                    "fnr_ci_high": fnr_high,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "flag_rate": metrics.flag_rate,
                })
    write_csv(output_dir / "subgroup_far_part_xgboost_operating_points.csv", subgroup_rows)

    summary = {
        "feature_set": feature_set_label,
        "n_events": len(y),
        "n_features": len(numeric_features) + len(categorical_features),
        "n_numeric": len(numeric_features),
        "n_categorical": len(categorical_features),
        "severe_events": int(y.sum()),
        "random_train": len(random_train),
        "random_test": len(random_test),
        "temporal_train": len(temporal_train),
        "temporal_test": len(temporal_test),
        "output_dir": str(output_dir),
        "probability_adjustment": "prior_correction_after_random_oversampling",
        "crew_rule": "aggregate_all_crew_for_selected_primary_aircraft",
    }
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
