#!/usr/bin/env python3
"""Plot classification performance versus true event properties."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.io import load_outputs
from analysis.paths import output_path
from analysis.plotting import save

CLASS_NAMES = ("muon", "electron", "pion")
N_CLASSES = len(CLASS_NAMES)

COUNT_LABEL_FONTSIZE = 7
PAIRWISE_ALPHA = 0.40


def prepare_labels(values, name):
    """Convert class targets to integer labels with shape (N,)."""
    values = np.asarray(values)

    if values.ndim == 1:
        labels = values
    elif values.ndim == 2 and values.shape[1] == 1:
        labels = values[:, 0]
    elif values.ndim == 2 and values.shape[1] == N_CLASSES:
        labels = np.argmax(values, axis=1)
    else:
        raise ValueError(
            f"{name} must have shape (N,), (N, 1), "
            f"or (N, {N_CLASSES}), but received {values.shape}."
        )

    labels = labels.astype(np.int64)
    valid_labels = np.arange(N_CLASSES)

    if not np.all(np.isin(labels, valid_labels)):
        raise ValueError(
            f"{name} contains unexpected labels: "
            f"{np.unique(labels).tolist()}. "
            f"Expected labels: {valid_labels.tolist()}."
        )

    return labels


def prepare_logits(values):
    """Validate classification predictions as logits with shape (N, 3)."""
    logits = np.asarray(values, dtype=float)

    if logits.ndim != 2 or logits.shape[1] != N_CLASSES:
        raise ValueError(
            f"Predictions must have shape (N, {N_CLASSES}), "
            f"but received {logits.shape}."
        )

    return logits


def make_edges(
    values,
    width,
    *,
    lower=None,
    upper=None,
):
    """Create fixed-width bin edges."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]

    if finite.size == 0:
        raise ValueError("No finite values are available for binning.")

    if width <= 0:
        raise ValueError("Bin width must be positive.")

    start = width * np.floor(np.min(finite) / width) if lower is None else float(lower)

    stop = width * np.ceil(np.max(finite) / width) if upper is None else float(upper)

    if stop <= start:
        stop = start + width

    return np.arange(
        start,
        stop + 0.5 * width,
        width,
    )


def wilson_half_width(
    successes,
    total,
    *,
    z_value=1.0,
):
    """Return the Wilson-score half-width for a binomial fraction."""
    if total <= 0:
        return np.nan

    probability = successes / total
    denominator = 1.0 + z_value**2 / total

    variance = probability * (1.0 - probability) / total + z_value**2 / (4.0 * total**2)

    return z_value * np.sqrt(variance) / denominator


def calculate_binned_metrics(
    x_values,
    true_labels,
    predicted_labels,
    edges,
    class_index,
    *,
    minimum_count,
):
    """Calculate classification metrics in bins of one true variable."""
    x_values = np.asarray(
        x_values,
        dtype=float,
    )
    true_labels = np.asarray(
        true_labels,
        dtype=int,
    )
    predicted_labels = np.asarray(
        predicted_labels,
        dtype=int,
    )

    if not (len(x_values) == len(true_labels) == len(predicted_labels)):
        raise ValueError(
            "x_values, true_labels, and predicted_labels "
            "must contain the same number of events."
        )

    centres = 0.5 * (edges[:-1] + edges[1:])
    n_bins = len(centres)

    result = {
        "centres": centres,
        "n_true": np.zeros(
            n_bins,
            dtype=int,
        ),
        "n_predicted": np.zeros(
            n_bins,
            dtype=int,
        ),
        "n_correct": np.zeros(
            n_bins,
            dtype=int,
        ),
        "mispid": np.full(
            n_bins,
            np.nan,
        ),
        "mispid_error": np.full(
            n_bins,
            np.nan,
        ),
        "efficiency": np.full(
            n_bins,
            np.nan,
        ),
        "efficiency_error": np.full(
            n_bins,
            np.nan,
        ),
        "purity": np.full(
            n_bins,
            np.nan,
        ),
        "purity_error": np.full(
            n_bins,
            np.nan,
        ),
        "pairwise": np.full(
            (N_CLASSES, n_bins),
            np.nan,
        ),
        "pairwise_error": np.full(
            (N_CLASSES, n_bins),
            np.nan,
        ),
    }

    for bin_index, (low, high) in enumerate(
        zip(
            edges[:-1],
            edges[1:],
            strict=True,
        )
    ):
        in_bin = np.isfinite(x_values) & (x_values >= low)

        if bin_index == n_bins - 1:
            in_bin &= x_values <= high
        else:
            in_bin &= x_values < high

        true_class = in_bin & (true_labels == class_index)
        predicted_class = in_bin & (predicted_labels == class_index)
        correct_classification = true_class & (predicted_labels == class_index)

        n_true = np.count_nonzero(true_class)
        n_predicted = np.count_nonzero(predicted_class)
        n_correct = np.count_nonzero(correct_classification)
        n_wrong = n_true - n_correct

        result["n_true"][bin_index] = n_true
        result["n_predicted"][bin_index] = n_predicted
        result["n_correct"][bin_index] = n_correct

        if n_true >= minimum_count:
            result["mispid"][bin_index] = n_wrong / n_true
            result["mispid_error"][bin_index] = wilson_half_width(
                n_wrong,
                n_true,
            )

            result["efficiency"][bin_index] = n_correct / n_true
            result["efficiency_error"][bin_index] = wilson_half_width(
                n_correct,
                n_true,
            )

            for predicted_index in range(N_CLASSES):
                if predicted_index == class_index:
                    continue

                n_pair = np.count_nonzero(
                    true_class & (predicted_labels == predicted_index)
                )

                result["pairwise"][
                    predicted_index,
                    bin_index,
                ] = n_pair / n_true

                result["pairwise_error"][
                    predicted_index,
                    bin_index,
                ] = wilson_half_width(
                    n_pair,
                    n_true,
                )

        if n_predicted >= minimum_count:
            result["purity"][bin_index] = n_correct / n_predicted
            result["purity_error"][bin_index] = wilson_half_width(
                n_correct,
                n_predicted,
            )

    return result


def add_count_labels(
    axis,
    x_values,
    y_values,
    counts,
):
    """Add event counts above finite points."""
    valid = np.isfinite(x_values) & np.isfinite(y_values)

    for x_value, y_value, count in zip(
        x_values[valid],
        y_values[valid],
        counts[valid],
        strict=True,
    ):
        axis.annotate(
            f"{count:,}",
            xy=(x_value, y_value),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=COUNT_LABEL_FONTSIZE,
            color="dimgray",
        )


def validate_lengths(
    logits,
    output_labels,
    truth_labels,
    event_ids,
    wall_values,
    towall_values,
    energies,
):
    """Validate that all inputs contain the same number of events."""
    lengths = {
        "logits": len(logits),
        "output_labels": len(output_labels),
        "truth_labels": len(truth_labels),
        "event_ids": len(event_ids),
        "wall": len(wall_values),
        "towall": len(towall_values),
        "energies": len(energies),
    }

    if len(set(lengths.values())) != 1:
        details = ", ".join(f"{name}={length:,}" for name, length in lengths.items())

        raise RuntimeError(f"Event-count mismatch among inputs: {details}")


def validate_label_alignment(
    output_labels,
    truth_labels,
    event_ids,
):
    """Validate ordering using output and extracted truth labels."""
    mismatch = np.flatnonzero(output_labels != truth_labels)

    if mismatch.size == 0:
        print("Alignment check passed: output targets match extracted truth labels.")
        return

    examples = []

    for index in mismatch[:10]:
        examples.append(
            f"i={index}, "
            f"event_id={event_ids[index]}, "
            f"output={output_labels[index]}, "
            f"truth={truth_labels[index]}"
        )

    raise RuntimeError(
        "Target/truth order mismatch "
        f"({mismatch.size:,} events). " + "; ".join(examples)
    )


def validate_event_ids(
    indices,
    event_ids,
):
    """Compare output indices with extracted event IDs."""
    if indices is None:
        raise ValueError(
            "--check-event-ids was requested, but no indices file was loaded."
        )

    indices = np.asarray(indices)
    event_ids = np.asarray(event_ids)

    if indices.shape != event_ids.shape:
        raise ValueError(
            "indices.npy and event_ids have different shapes: "
            f"{indices.shape} and {event_ids.shape}."
        )

    if not np.array_equal(
        indices,
        event_ids,
    ):
        mismatch = np.flatnonzero(indices != event_ids)

        examples = []

        for index in mismatch[:10]:
            examples.append(
                f"i={index}, index={indices[index]}, event_id={event_ids[index]}"
            )

        raise ValueError(
            "indices.npy and truth event_ids differ for "
            f"{mismatch.size:,} events. " + "; ".join(examples)
        )

    print("Event-ID check passed: indices.npy matches truth event_ids.")


def load_truth_file(path):
    """Load classification truth information from an NPZ file."""
    path = path.expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Truth file was not found: {path}")

    required_arrays = (
        "event_ids",
        "labels",
        "wall",
        "towall",
        "energies",
    )

    with np.load(
        path,
        allow_pickle=False,
    ) as truth:
        missing = [name for name in required_arrays if name not in truth.files]

        if missing:
            raise KeyError(
                f"Missing truth arrays: {missing}. Available arrays: {truth.files}."
            )

        event_ids = np.asarray(truth["event_ids"])
        truth_labels = prepare_labels(
            truth["labels"],
            "truth labels",
        )
        wall_values = np.asarray(
            truth["wall"],
            dtype=float,
        )
        towall_values = np.asarray(
            truth["towall"],
            dtype=float,
        )
        energies = np.asarray(
            truth["energies"],
            dtype=float,
        )

    return {
        "event_ids": event_ids,
        "labels": truth_labels,
        "wall": wall_values,
        "towall": towall_values,
        "energies": energies,
    }


def save_metrics_csv(
    rows,
    output_dir,
):
    """Save binned classification metrics to CSV."""
    if not rows:
        raise ValueError("No classification metrics are available for CSV output.")

    path = output_path(
        output_dir,
        Path("classification_true_dependencies.csv"),
    )

    fieldnames = sorted({key for row in rows for key in row})

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {path}")


def plot_mispid_dependencies(
    variables,
    true_labels,
    predicted_labels,
    *,
    minimum_count,
    show_event_counts,
    output_dir,
    dpi,
):
    """Plot total and pairwise misidentification rates."""
    fig, axes = plt.subplots(
        N_CLASSES,
        len(variables),
        figsize=(19, 14),
        squeeze=False,
    )

    csv_rows = []

    for class_index, class_name in enumerate(CLASS_NAMES):
        for column, variable in enumerate(variables):
            (
                xlabel,
                x_values,
                edges,
                variable_name,
            ) = variable

            metrics = calculate_binned_metrics(
                x_values,
                true_labels,
                predicted_labels,
                edges,
                class_index,
                minimum_count=minimum_count,
            )

            centres = metrics["centres"]
            axis = axes[
                class_index,
                column,
            ]

            valid = np.isfinite(metrics["mispid"])

            axis.errorbar(
                centres[valid],
                100.0 * metrics["mispid"][valid],
                yerr=(100.0 * metrics["mispid_error"][valid]),
                fmt="o-",
                linewidth=2.0,
                capsize=2,
                label="Total mis-PID",
            )

            for (
                predicted_index,
                predicted_name,
            ) in enumerate(CLASS_NAMES):
                if predicted_index == class_index:
                    continue

                pairwise = metrics["pairwise"][predicted_index]
                pairwise_error = metrics["pairwise_error"][predicted_index]
                pairwise_valid = np.isfinite(pairwise)

                axis.errorbar(
                    centres[pairwise_valid],
                    100.0 * pairwise[pairwise_valid],
                    yerr=(100.0 * pairwise_error[pairwise_valid]),
                    fmt="o--",
                    linewidth=1.0,
                    markersize=3,
                    capsize=2,
                    alpha=PAIRWISE_ALPHA,
                    label=(f"{class_name} to {predicted_name}"),
                )

            if show_event_counts:
                add_count_labels(
                    axis,
                    centres,
                    100.0 * metrics["mispid"],
                    metrics["n_true"],
                )
                axis.margins(y=0.15)

            axis.set(
                xlabel=xlabel,
                ylabel="Mis-PID rate [%]",
                title=f"True {class_name}",
            )
            axis.set_ylim(bottom=0.0)
            axis.grid(alpha=0.3)
            axis.legend(
                fontsize=8,
                frameon=True,
                framealpha=0.8,
            )

            for bin_index in range(len(centres)):
                row = {
                    "variable": variable_name,
                    "true_class": class_name,
                    "bin_low": edges[bin_index],
                    "bin_high": edges[bin_index + 1],
                    "bin_center": centres[bin_index],
                    "n_true": metrics["n_true"][bin_index],
                    "n_predicted": (metrics["n_predicted"][bin_index]),
                    "n_correct": (metrics["n_correct"][bin_index]),
                    "mispid_rate": (metrics["mispid"][bin_index]),
                    "mispid_error": (metrics["mispid_error"][bin_index]),
                    "efficiency": (metrics["efficiency"][bin_index]),
                    "efficiency_error": (metrics["efficiency_error"][bin_index]),
                    "purity": (metrics["purity"][bin_index]),
                    "purity_error": (metrics["purity_error"][bin_index]),
                    "minimum_count": minimum_count,
                }

                for (
                    predicted_index,
                    predicted_name,
                ) in enumerate(CLASS_NAMES):
                    if predicted_index == class_index:
                        continue

                    row[f"to_{predicted_name}"] = metrics["pairwise"][
                        predicted_index,
                        bin_index,
                    ]

                    row[f"to_{predicted_name}_error"] = metrics["pairwise_error"][
                        predicted_index,
                        bin_index,
                    ]

                csv_rows.append(row)

    fig.suptitle(
        "Particle misidentification versus true event geometry and energy",
        fontsize=17,
    )
    fig.tight_layout()

    save(
        fig,
        output_path(
            output_dir,
            Path("classification_mispid_vs_true_wall_towall_energy.png"),
        ),
        dpi,
    )

    return csv_rows


def plot_efficiency_purity_dependencies(
    variables,
    true_labels,
    predicted_labels,
    *,
    minimum_count,
    show_event_counts,
    output_dir,
    dpi,
):
    """Plot classification efficiency and purity."""
    fig, axes = plt.subplots(
        N_CLASSES,
        len(variables),
        figsize=(19, 14),
        squeeze=False,
    )

    for class_index, class_name in enumerate(CLASS_NAMES):
        for column, variable in enumerate(variables):
            (
                xlabel,
                x_values,
                edges,
                _,
            ) = variable

            metrics = calculate_binned_metrics(
                x_values,
                true_labels,
                predicted_labels,
                edges,
                class_index,
                minimum_count=minimum_count,
            )

            centres = metrics["centres"]
            axis = axes[
                class_index,
                column,
            ]

            for metric_name, marker in (
                ("efficiency", "o"),
                ("purity", "s"),
            ):
                values = metrics[metric_name]
                errors = metrics[f"{metric_name}_error"]
                valid = np.isfinite(values)

                axis.errorbar(
                    centres[valid],
                    100.0 * values[valid],
                    yerr=(100.0 * errors[valid]),
                    fmt=f"{marker}-",
                    capsize=2,
                    linewidth=1.5,
                    markersize=4,
                    label=metric_name.capitalize(),
                )

            if show_event_counts:
                efficiency_percent = 100.0 * metrics["efficiency"]

                add_count_labels(
                    axis,
                    centres,
                    efficiency_percent,
                    metrics["n_true"],
                )
                axis.margins(y=0.15)

            axis.set(
                xlabel=xlabel,
                ylabel="Metric [%]",
                title=class_name.capitalize(),
                ylim=(0.0, 105.0),
            )
            axis.grid(alpha=0.3)
            axis.legend()

    fig.suptitle(
        "Classification efficiency and purity versus true event properties",
        fontsize=17,
    )
    fig.tight_layout()

    save(
        fig,
        output_path(
            output_dir,
            Path("classification_efficiency_purity_vs_true_wall_towall_energy.png"),
        ),
        dpi,
    )


def build_parser():
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot classification performance versus true Wall, ToWall, and energy."
        )
    )

    parser.add_argument(
        "--classification-run",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs/preds.npy"),
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path("outputs/targets.npy"),
    )
    parser.add_argument(
        "--indices",
        type=Path,
        default=Path("outputs/indices.npy"),
    )
    parser.add_argument(
        "--truth-file",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/classification_dependencies"),
    )

    parser.add_argument(
        "--wall-bin-width",
        type=float,
        default=50.0,
    )
    parser.add_argument(
        "--towall-bin-width",
        type=float,
        default=50.0,
    )
    parser.add_argument(
        "--energy-bin-width",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--minimum-count",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--show-event-counts",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--check-event-ids",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Require indices.npy to equal truth event_ids. "
            "Enable only when both arrays use the same "
            "identifier definition."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
    )

    return parser


def main():
    """Generate classification-dependency plots and CSV output."""
    args = build_parser().parse_args()

    if args.minimum_count <= 0:
        raise ValueError("--minimum-count must be positive.")

    bin_widths = {
        "--wall-bin-width": (args.wall_bin_width),
        "--towall-bin-width": (args.towall_bin_width),
        "--energy-bin-width": (args.energy_bin_width),
    }

    for option, width in bin_widths.items():
        if width <= 0:
            raise ValueError(f"{option} must be positive.")

    outputs = load_outputs(
        args.classification_run,
        args.predictions,
        args.targets,
        args.indices,
        require_indices=args.check_event_ids,
    )

    logits = prepare_logits(outputs.predictions)
    output_labels = prepare_labels(
        outputs.targets,
        "output targets",
    )

    truth = load_truth_file(args.truth_file)

    validate_lengths(
        logits,
        output_labels,
        truth["labels"],
        truth["event_ids"],
        truth["wall"],
        truth["towall"],
        truth["energies"],
    )

    validate_label_alignment(
        output_labels,
        truth["labels"],
        truth["event_ids"],
    )

    if args.check_event_ids:
        validate_event_ids(
            outputs.indices,
            truth["event_ids"],
        )

    finite = (
        np.all(
            np.isfinite(logits),
            axis=1,
        )
        & np.isfinite(truth["wall"])
        & np.isfinite(truth["towall"])
        & np.isfinite(truth["energies"])
    )

    n_invalid = np.count_nonzero(~finite)

    if n_invalid > 0:
        print(f"Warning: removing {n_invalid:,} events containing non-finite values.")

    logits = logits[finite]
    true_labels = truth["labels"][finite]
    wall_values = truth["wall"][finite]
    towall_values = truth["towall"][finite]
    energies = truth["energies"][finite]

    if len(true_labels) == 0:
        raise ValueError("No valid events remain after filtering.")

    predicted_labels = np.argmax(
        logits,
        axis=1,
    )

    variables = [
        (
            "True Wall [cm]",
            wall_values,
            make_edges(
                wall_values,
                args.wall_bin_width,
            ),
            "wall",
        ),
        (
            "True ToWall [cm]",
            towall_values,
            make_edges(
                towall_values,
                args.towall_bin_width,
                lower=0.0,
            ),
            "towall",
        ),
        (
            "True energy [MeV]",
            energies,
            make_edges(
                energies,
                args.energy_bin_width,
                lower=0.0,
            ),
            "energy",
        ),
    ]

    csv_rows = plot_mispid_dependencies(
        variables,
        true_labels,
        predicted_labels,
        minimum_count=args.minimum_count,
        show_event_counts=(args.show_event_counts),
        output_dir=args.output_dir,
        dpi=args.dpi,
    )

    plot_efficiency_purity_dependencies(
        variables,
        true_labels,
        predicted_labels,
        minimum_count=args.minimum_count,
        show_event_counts=(args.show_event_counts),
        output_dir=args.output_dir,
        dpi=args.dpi,
    )

    save_metrics_csv(
        csv_rows,
        args.output_dir,
    )

    print()
    print("========== Classification dependency summary ==========")
    print(f"Events used: {len(true_labels):,}")
    print(f"Overall accuracy: {np.mean(predicted_labels == true_labels):.6f}")

    for class_index, class_name in enumerate(CLASS_NAMES):
        true_mask = true_labels == class_index
        n_true = np.count_nonzero(true_mask)
        n_correct = np.count_nonzero(true_mask & (predicted_labels == class_index))

        efficiency = n_correct / n_true if n_true > 0 else np.nan

        print(f"True {class_name:8s}: N={n_true:,}, efficiency={efficiency:.6f}")

    print("=======================================================")
    print()


if __name__ == "__main__":
    main()
