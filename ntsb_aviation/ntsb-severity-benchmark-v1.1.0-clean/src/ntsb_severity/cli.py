from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from .audit import run_audit
from .benchmark import run_benchmark
from .construct import build_analytical_dataset
from .dictionary_audit import run_dictionary_audit
from .reporting import plot_annual_severity, plot_model_curves


def parser() -> ArgumentParser:
    p = ArgumentParser(prog="ntsb-severity")
    sub = p.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="Audit raw Excel tables")
    audit.add_argument("--data-dir", required=True)
    audit.add_argument("--output-dir", required=True)
    audit.add_argument("--features", required=True)

    build = sub.add_parser("build", help="Build event-level analytical CSV")
    build.add_argument("--data-dir", required=True)
    build.add_argument("--output", required=True)

    bench = sub.add_parser("benchmark", help="Run random and temporal models")
    bench.add_argument("--dataset", required=True)
    bench.add_argument("--config", required=True)
    bench.add_argument("--features", required=True)
    bench.add_argument("--output-dir", required=True)
    bench.add_argument("--feature-set-label", default="unspecified")


    dictionary = sub.add_parser("dictionary-audit", help="Audit predictors against the eADMS data dictionary")
    dictionary.add_argument("--dictionary", required=True)
    dictionary.add_argument("--registry", required=True)
    dictionary.add_argument("--output", required=True)
    dictionary.add_argument("--missingness")

    figures = sub.add_parser("figures", help="Create core manuscript figures")
    figures.add_argument("--audit-dir", required=True)
    figures.add_argument("--benchmark-dir", required=True)
    figures.add_argument("--output-dir", required=True)
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.command == "audit":
        result = run_audit(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            feature_config=args.features,
        )
        print(result)
    elif args.command == "build":
        data_dir = Path(args.data_dir)
        result = build_analytical_dataset(
            events_path=data_dir / "events.xlsx",
            aircraft_path=data_dir / "aircraft.xlsx",
            crew_path=data_dir / "Flight_Crew.xlsx",
            output_csv=args.output,
        )
        print(result)
    elif args.command == "benchmark":
        result = run_benchmark(
            dataset_csv=args.dataset,
            config_path=args.config,
            feature_config_path=args.features,
            output_dir=args.output_dir,
            feature_set_label=args.feature_set_label,
        )
        print(result)
    elif args.command == "dictionary-audit":
        result = run_dictionary_audit(
            dictionary_path=args.dictionary,
            registry_path=args.registry,
            output_csv=args.output,
            missingness_csv=args.missingness,
        )
        print({"features_audited": len(result), "output": args.output})
    elif args.command == "figures":
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        plot_annual_severity(
            Path(args.audit_dir) / "04_year_distribution.csv",
            output / "annual_severity_rate.png",
        )
        benchmark = Path(args.benchmark_dir)
        plot_model_curves(
            {
                "Logistic regression": benchmark / "predictions_random_logistic_regression.csv",
                "Random forest": benchmark / "predictions_random_random_forest.csv",
                "XGBoost": benchmark / "predictions_random_xgboost.csv",
            },
            output / "random_split",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
