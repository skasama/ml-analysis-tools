import argparse
from pathlib import Path

from analysis.comparison import compare_methods
from analysis.io import load_outputs
from analysis.metrics import regression_metrics
from analysis.plotting import plot_metric_comparison, plot_residual_comparison, save
from analysis.results import MethodResult


def _method(value):
    try:
        name, run_dir = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("methods must use NAME=RUN_DIR") from error
    if not name or not run_dir:
        raise argparse.ArgumentTypeError("methods must use NAME=RUN_DIR")
    return name, Path(run_dir)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compare two or more regression results on common events."
    )
    parser.add_argument(
        "--method",
        type=_method,
        action="append",
        required=True,
        metavar="NAME=RUN_DIR",
        help="Repeat for each method, for example ml=run_ml.",
    )
    parser.add_argument("--predictions", type=Path, default=Path("outputs/preds.npy"))
    parser.add_argument("--targets", type=Path, default=Path("outputs/targets.npy"))
    parser.add_argument("--indices", type=Path, default=Path("outputs/indices.npy"))
    parser.add_argument("--output-dir", type=Path, default=Path("plots/comparison"))
    parser.add_argument("--histogram-bins", type=int, default=60)
    parser.add_argument(
        "--metric",
        choices=("mae", "rmse", "bias", "resolution_68", "pearson_r"),
        default="mae",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if len(args.method) < 2:
        raise ValueError("at least two --method options are required")
    if args.histogram_bins <= 0:
        raise ValueError("--histogram-bins must be positive")

    results = []
    for name, run_dir in args.method:
        outputs = load_outputs(
            run_dir,
            args.predictions,
            args.targets,
            args.indices,
            require_indices=True,
        )
        results.append(
            MethodResult(
                method=name,
                task="regression",
                predictions=outputs.predictions,
                targets=outputs.targets,
                event_ids=outputs.indices,
            )
        )

    metrics = compare_methods(results, regression_metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_figure = plot_metric_comparison(
        metrics,
        metric=args.metric,
        title=f"{args.metric} comparison",
    )
    save(metric_figure, args.output_dir / f"{args.metric}_comparison.png")
    residual_figure = plot_residual_comparison(
        tuple(result for result in results),
        bins=args.histogram_bins,
    )
    save(residual_figure, args.output_dir / "residual_comparison.png")

    for method, values in metrics.items():
        print(f"{method}: {args.metric}={values[args.metric]:.6g}")


if __name__ == "__main__":
    main()
