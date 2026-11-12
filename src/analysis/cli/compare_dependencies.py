import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.io import load_variable
from analysis.metrics import resolution68
from analysis.plotting import save


def _variables(values, count, label):
    if not values:
        return [None] * count
    if len(values) == 1 and count > 1:
        return [None] * (count - 1) + values
    if len(values) != count:
        raise ValueError(f"{label} must be specified once per method")
    return values


def _load_array(path, variable):
    return np.asarray(load_variable(path, variable))


def _labels(values, n_classes):
    values = np.asarray(values)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    elif values.ndim == 2:
        if values.shape[1] != n_classes:
            raise ValueError("one-hot targets have an unexpected class count")
        values = np.argmax(values, axis=1)
    return values.astype(int)


def _common_edges(dependencies, n_bins):
    finite = np.concatenate(
        [np.asarray(values, dtype=float).ravel() for values in dependencies]
    )
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("dependency arrays contain no finite values")
    low, high = float(np.min(finite)), float(np.max(finite))
    if high <= low:
        high = low + np.finfo(float).eps
    return np.linspace(low, high, n_bins + 1)


def _bin_mask(values, low, high, last):
    values = np.asarray(values, dtype=float).ravel()
    return (values >= low) & ((values <= high) if last else (values < high))


def _regression_profile(dependency, predictions, targets, edges, mode):
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if predictions.shape != targets.shape or len(predictions) != len(dependency):
        raise ValueError("prediction, target, and dependency lengths must match")
    if predictions.ndim == 1:
        residuals = predictions - targets
    elif predictions.ndim == 2:
        residuals = np.linalg.norm(predictions - targets, axis=1)
    else:
        raise ValueError("regression arrays must have shape (N,) or (N, D)")
    if mode == "absolute" or (mode == "norm" and predictions.ndim == 1):
        residuals = np.abs(residuals)

    bias = np.full(len(edges) - 1, np.nan)
    resolution = np.full(len(edges) - 1, np.nan)
    counts = np.zeros(len(edges) - 1, dtype=int)
    dependency = np.asarray(dependency, dtype=float).ravel()
    valid = np.isfinite(dependency) & np.isfinite(residuals)
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        selected = valid & _bin_mask(dependency, low, high, index == len(edges) - 2)
        values = residuals[selected]
        counts[index] = values.size
        if values.size:
            bias[index] = np.mean(values)
            resolution[index] = resolution68(values)
    return bias, resolution, counts


def _classification_profile(dependency, predictions, targets, edges, n_classes):
    predictions = np.asarray(predictions)
    if predictions.ndim != 2 or predictions.shape[1] != n_classes:
        raise ValueError("classification predictions must have shape (N, n_classes)")
    labels = _labels(targets, n_classes)
    predicted = np.argmax(predictions, axis=1)
    if len(dependency) != len(labels):
        raise ValueError("prediction, target, and dependency lengths must match")
    efficiency = np.full((n_classes, len(edges) - 1), np.nan)
    purity = np.full_like(efficiency, np.nan)
    mis_pid = np.full_like(efficiency, np.nan)
    counts = np.zeros(len(edges) - 1, dtype=int)
    dependency = np.asarray(dependency, dtype=float).ravel()
    valid = np.isfinite(dependency)
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        selected = valid & _bin_mask(dependency, low, high, index == len(edges) - 2)
        counts[index] = int(np.sum(selected))
        for class_index in range(n_classes):
            truth = labels[selected] == class_index
            prediction = predicted[selected] == class_index
            tp = np.sum(truth & prediction)
            fn = np.sum(truth & ~prediction)
            fp = np.sum(~truth & prediction)
            tn = np.sum(~truth & ~prediction)
            efficiency[class_index, index] = tp / (tp + fn) if tp + fn else np.nan
            purity[class_index, index] = tp / (tp + fp) if tp + fp else np.nan
            mis_pid[class_index, index] = fp / (fp + tn) if fp + tn else np.nan
    return efficiency, purity, mis_pid, counts


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compare regression or classification metrics versus a dependency."
    )
    parser.add_argument(
        "--task",
        choices=("regression", "classification"),
        required=True,
    )
    parser.add_argument(
        "--dependency",
        choices=("wall", "towall", "energy", "true"),
        required=True,
        help="Name used for axis labels and output files.",
    )
    parser.add_argument("--method", action="append", required=True, metavar="NAME")
    parser.add_argument("--prediction", action="append", required=True, metavar="FILE")
    parser.add_argument("--prediction-variable", action="append", default=[])
    parser.add_argument("--target", action="append", required=True, metavar="FILE")
    parser.add_argument("--target-variable", action="append", default=[])
    parser.add_argument("--indices", action="append", default=[], metavar="FILE")
    parser.add_argument("--indices-variable", action="append", default=[])
    parser.add_argument("--x-file", action="append", required=True, metavar="FILE")
    parser.add_argument("--x-variable", action="append", default=[])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/comparison-dependencies"),
    )
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--minimum-count", type=int, default=1)
    parser.add_argument(
        "--residual-mode",
        choices=("signed", "absolute", "norm"),
        default="signed",
        help="Regression residual mode; multi-dimensional inputs always use norm.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    return parser


def main():
    args = build_parser().parse_args()
    count = len(args.method)
    if count < 2:
        raise ValueError("at least two --method options are required")
    if len(args.prediction) != count or len(args.target) != count:
        raise ValueError("prediction and target must be specified once per method")
    if args.indices and len(args.indices) != count:
        raise ValueError("--indices must be specified once per method")
    if len(args.x_file) not in {1, count}:
        raise ValueError("--x-file must be specified once or once per method")
    if args.bins <= 0 or args.minimum_count <= 0:
        raise ValueError("--bins and --minimum-count must be positive")

    prediction_variables = _variables(
        args.prediction_variable, count, "--prediction-variable"
    )
    target_variables = _variables(args.target_variable, count, "--target-variable")
    index_variables = _variables(args.indices_variable, count, "--indices-variable")
    x_variables = _variables(args.x_variable, len(args.x_file), "--x-variable")
    x_files = args.x_file if len(args.x_file) == count else args.x_file * count

    predictions = [
        _load_array(path, variable)
        for path, variable in zip(args.prediction, prediction_variables, strict=True)
    ]
    targets = [
        _load_array(path, variable)
        for path, variable in zip(args.target, target_variables, strict=True)
    ]
    dependencies = [
        _load_array(path, variable)
        for path, variable in zip(
            x_files,
            x_variables * (count // len(x_variables)),
            strict=True,
        )
    ]
    indices = None
    if args.indices:
        indices = [
            _load_array(path, variable).ravel()
            for path, variable in zip(args.indices, index_variables, strict=True)
        ]
        if len(args.x_file) == 1:
            raise ValueError(
                "when --indices is used, specify --x-file once per method "
                "so dependencies can be aligned"
            )
        for method, index, prediction, target in zip(
            args.method,
            indices,
            predictions,
            targets,
            strict=True,
        ):
            if len(index) != len(prediction) or len(index) != len(target):
                raise ValueError(
                    f"{method}: index, prediction, and target lengths must match"
                )
        common = set(indices[0].tolist())
        for values in indices[1:]:
            common.intersection_update(values.tolist())
        if not common:
            raise ValueError("methods have no common event indices")
        ordered = [value for value in indices[0] if value in common]
        aligned_predictions = []
        aligned_targets = []
        aligned_dependencies = []
        for index, prediction, target, dependency in zip(
            indices,
            predictions,
            targets,
            dependencies,
            strict=True,
        ):
            positions = {value: position for position, value in enumerate(index)}
            selection = np.asarray([positions[value] for value in ordered])
            aligned_predictions.append(prediction[selection])
            aligned_targets.append(target[selection])
            aligned_dependencies.append(dependency[selection])
        predictions = aligned_predictions
        targets = aligned_targets
        dependencies = aligned_dependencies
    edges = _common_edges(dependencies, args.bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.task == "regression":
        profiles = [
            _regression_profile(
                dependency,
                prediction,
                target,
                edges,
                args.residual_mode,
            )
            for dependency, prediction, target in zip(
                dependencies, predictions, targets, strict=True
            )
        ]
        figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
        for method, (bias, resolution, counts) in zip(
            args.method,
            profiles,
            strict=True,
        ):
            valid = counts >= args.minimum_count
            axes[0].plot(centres[valid], bias[valid], "o-", label=method)
            axes[1].plot(centres[valid], resolution[valid], "o-", label=method)
        axes[0].axhline(0.0, color="black", linestyle="--")
        axes[0].set_ylabel("Bias")
        axes[1].set_ylabel("68% resolution")
        for axis in axes:
            axis.set_xlabel(args.dependency)
            axis.grid(alpha=0.3)
            axis.legend()
        figure.suptitle(f"Regression metrics versus {args.dependency}")
        figure.tight_layout()
        save(figure, args.output_dir / f"{args.dependency}_comparison.png", args.dpi)
    else:
        n_classes = np.asarray(predictions[0]).shape[1]
        profiles = [
            _classification_profile(
                dependency,
                prediction,
                target,
                edges,
                n_classes,
            )
            for dependency, prediction, target in zip(
                dependencies, predictions, targets, strict=True
            )
        ]
        figure, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
        for class_index in range(n_classes):
            for axis, metric_index, label in zip(
                axes,
                range(3),
                ("Efficiency", "Purity", "Mis-PID rate"),
                strict=True,
            ):
                for method, profile in zip(args.method, profiles, strict=True):
                    values = profile[metric_index][class_index]
                    valid = profile[3] >= args.minimum_count
                    axis.plot(centres[valid], values[valid], "o-", label=method)
                axis.set_title(f"{label} class {class_index}")
                axis.set_xlabel(args.dependency)
                axis.grid(alpha=0.3)
        axes[0].set_ylabel("Fraction")
        axes[-1].legend()
        figure.suptitle(f"Classification metrics versus {args.dependency}")
        figure.tight_layout()
        save(
            figure,
            args.output_dir / f"{args.dependency}_classification_comparison.png",
            args.dpi,
        )


if __name__ == "__main__":
    main()
