import argparse
from pathlib import Path

import numpy as np

from analysis.comparison import compare_methods
from analysis.io import load_variable
from analysis.metrics import classification_metrics, regression_metrics
from analysis.plotting import (
    plot_bias_resolution_comparison,
    plot_classification_comparison,
    plot_metric_comparison,
    plot_residual_comparison,
    save,
)
from analysis.results import MethodResult


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compare two or more regression or classification results."
    )
    parser.add_argument(
        "--task",
        choices=("regression", "classification"),
        required=True,
        help="Analysis task used to select the metrics and plots.",
    )
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        metavar="NAME",
        help="Method label. Repeat once for each result.",
    )
    parser.add_argument(
        "--prediction",
        type=Path,
        action="append",
        required=True,
        metavar="FILE",
        help="Prediction file. Repeat in the same order as --method.",
    )
    parser.add_argument(
        "--prediction-variable",
        action="append",
        default=[],
        metavar="NAME",
        help="Variable in the prediction file, for NPZ/HDF5 inputs.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        action="append",
        default=[],
        metavar="FILE",
        help="Truth file per method. Repeat in the same order as --method.",
    )
    parser.add_argument(
        "--target-variable",
        action="append",
        default=[],
        metavar="NAME",
        help="Variable in each target file, for NPZ/HDF5 inputs.",
    )
    parser.add_argument(
        "--truth",
        type=Path,
        help="One truth file shared by every method.",
    )
    parser.add_argument(
        "--truth-variable",
        metavar="NAME",
        help="Variable in the shared truth file, for NPZ/HDF5 inputs.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("plots/comparison"))
    parser.add_argument("--histogram-bins", type=int, default=60)
    parser.add_argument(
        "--metric",
        help=(
            "Metric to plot; defaults to MAE for regression and "
            "accuracy for classification."
        ),
    )
    return parser


def _variables(values, count, label):
    if not values:
        return [None] * count
    if len(values) != count:
        raise ValueError(f"{label} must be specified once per method")
    return values


def _load_results(args):
    method_count = len(args.method)
    if len(args.prediction) != method_count:
        raise ValueError("--prediction must be specified once per method")
    if args.truth is not None and args.target:
        raise ValueError("use either --truth or repeated --target, not both")
    if args.truth is None and len(args.target) != method_count:
        raise ValueError("provide one shared --truth or one --target for every method")

    prediction_variables = _variables(
        args.prediction_variable,
        method_count,
        "--prediction-variable",
    )
    target_variables = _variables(
        args.target_variable,
        method_count,
        "--target-variable",
    )
    shared_targets = (
        load_variable(args.truth, args.truth_variable)
        if args.truth is not None
        else None
    )
    results = []
    for index, method in enumerate(args.method):
        predictions = load_variable(args.prediction[index], prediction_variables[index])
        targets = (
            shared_targets
            if shared_targets is not None
            else load_variable(args.target[index], target_variables[index])
        )
        predictions = np.asarray(predictions)
        targets = np.asarray(targets)
        if len(predictions) != len(targets):
            raise ValueError(
                f"{method}: predictions and targets have different event counts "
                f"({len(predictions)} != {len(targets)})"
            )
        results.append(
            MethodResult(
                method=method,
                task=args.task,
                predictions=predictions,
                targets=targets,
                event_ids=np.arange(len(targets)),
            )
        )
    return results


def main():
    args = build_parser().parse_args()
    if len(args.method) < 2:
        raise ValueError("at least two --method options are required")
    if args.histogram_bins <= 0:
        raise ValueError("--histogram-bins must be positive")

    results = _load_results(args)
    metric_function = (
        regression_metrics if args.task == "regression" else classification_metrics
    )
    metric = args.metric or ("mae" if args.task == "regression" else "accuracy")
    allowed_metrics = (
        {"mae", "rmse", "bias", "resolution_68", "pearson_r"}
        if args.task == "regression"
        else {"accuracy"}
    )
    if metric not in allowed_metrics:
        raise ValueError(
            f"--metric {metric!r} is not available for {args.task}; "
            f"choose from {sorted(allowed_metrics)}"
        )

    metrics = compare_methods(results, metric_function, align=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_figure = plot_metric_comparison(
        metrics,
        metric=metric,
        title=f"{metric} comparison",
    )
    save(metric_figure, args.output_dir / f"{metric}_comparison.png")
    if args.task == "regression":
        bias_resolution_figure = plot_bias_resolution_comparison(metrics)
        save(
            bias_resolution_figure,
            args.output_dir / "bias_resolution_comparison.png",
        )
        residual_figure = plot_residual_comparison(
            tuple(results),
            bins=args.histogram_bins,
        )
        save(residual_figure, args.output_dir / "residual_comparison.png")
    else:
        classification_figure = plot_classification_comparison(metrics)
        save(
            classification_figure,
            args.output_dir / "classification_comparison.png",
        )

    for method, values in metrics.items():
        print(f"{method}: {metric}={values[metric]:.6g}")


if __name__ == "__main__":
    main()
