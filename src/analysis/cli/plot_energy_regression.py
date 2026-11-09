import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.io import load_outputs
from analysis.metrics import resolution68
from analysis.paths import output_path
from analysis.plotting import save, scatter_identity
from analysis.regression import inverse_minmax, scalar

UNIT = "MeV"
EPSILON = 1.0e-10


def calculate_relative_residuals(predictions, targets):
    """Calculate relative residuals for non-zero target energies."""
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)

    valid = (
        np.isfinite(predictions) & np.isfinite(targets) & (np.abs(targets) > EPSILON)
    )

    relative_residuals = (predictions[valid] - targets[valid]) / targets[valid]

    return relative_residuals, valid


def print_statistics(predictions, targets):
    """Print absolute and relative energy-regression statistics."""
    residuals = predictions - targets

    relative_residuals, relative_valid = calculate_relative_residuals(
        predictions,
        targets,
    )

    bias = np.mean(residuals)
    resolution = resolution68(residuals)

    print()
    print("========== Energy regression statistics ==========")
    print(f"Number of events: {len(targets):,}")
    print(f"Unit: {UNIT}")

    print()
    print("Absolute residual:")
    print(f"  Bias: {bias:.6g} {UNIT}")
    print(f"  Median residual: {np.median(residuals):.6g} {UNIT}")
    print(f"  68% resolution: {resolution:.6g} {UNIT}")
    print(f"  MAE: {np.mean(np.abs(residuals)):.6g} {UNIT}")
    print(f"  RMSE: {np.sqrt(np.mean(residuals**2)):.6g} {UNIT}")

    print()
    print("Relative residual:")

    if relative_residuals.size > 0:
        relative_bias = np.mean(relative_residuals)
        relative_resolution = resolution68(relative_residuals)

        print(f"  Valid events: {np.sum(relative_valid):,} / {len(targets):,}")
        print(f"  Relative bias: {relative_bias:.6g} ({100.0 * relative_bias:.3f}%)")
        print(
            "  Relative 68% resolution: "
            f"{relative_resolution:.6g} "
            f"({100.0 * relative_resolution:.3f}%)"
        )
        print(
            "  Mean absolute percentage error: "
            f"{100.0 * np.mean(np.abs(relative_residuals)):.3f}%"
        )
    else:
        print(
            "  Relative residuals could not be calculated "
            "because no valid non-zero target energies remain."
        )

    print("==================================================")
    print()


def plot_prediction_vs_truth(
    predictions,
    targets,
    *,
    output_dir,
    dpi,
):
    """Plot predicted energy against true energy."""
    fig = scatter_identity(
        predictions,
        targets,
        xlabel="True energy",
        ylabel="Predicted energy",
        title="Energy regression: prediction vs truth",
        unit=UNIT,
    )

    save(
        fig,
        output_path(
            output_dir,
            Path("energy_prediction_vs_truth.png"),
        ),
        dpi,
    )


def plot_residuals(
    predictions,
    targets,
    *,
    histogram_bins,
    output_dir,
    dpi,
):
    """Plot absolute and relative energy residuals."""
    residuals = predictions - targets

    relative_residuals, _ = calculate_relative_residuals(
        predictions,
        targets,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
    )

    absolute_bias = np.mean(residuals)
    absolute_resolution = resolution68(residuals)

    axes[0].hist(
        residuals,
        bins=histogram_bins,
        histtype="step",
        linewidth=1.5,
        color="tab:blue",
    )
    axes[0].axvline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="Zero residual",
    )
    axes[0].axvline(
        absolute_bias,
        color="red",
        linewidth=1.8,
        label=(f"Bias = {absolute_bias:.2f} {UNIT}"),
    )
    axes[0].axvline(
        absolute_resolution,
        color="purple",
        linestyle=":",
        linewidth=1.8,
        label=(f"68% resolution = {absolute_resolution:.2f} {UNIT}"),
    )
    axes[0].axvline(
        -absolute_resolution,
        color="purple",
        linestyle=":",
        linewidth=1.8,
    )
    axes[0].set(
        xlabel=(f"Predicted energy - true energy [{UNIT}]"),
        ylabel="Events",
        title=(
            "Energy residual\n"
            f"Bias = {absolute_bias:.2f} {UNIT}\n"
            f"Resolution = "
            f"{absolute_resolution:.2f} {UNIT}"
        ),
    )
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=9)

    if relative_residuals.size > 0:
        relative_residuals_percent = 100.0 * relative_residuals
        relative_bias_percent = float(np.mean(relative_residuals_percent))
        relative_resolution_percent = 100.0 * resolution68(relative_residuals)

        axes[1].hist(
            relative_residuals_percent,
            bins=histogram_bins,
            histtype="step",
            linewidth=1.5,
            color="tab:green",
        )
        axes[1].axvline(
            0.0,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="Zero residual",
        )
        axes[1].axvline(
            relative_bias_percent,
            color="red",
            linewidth=1.8,
            label=(f"Bias = {relative_bias_percent:.2f}%"),
        )
        axes[1].axvline(
            relative_resolution_percent,
            color="purple",
            linestyle=":",
            linewidth=1.8,
            label=(f"68% resolution = {relative_resolution_percent:.2f}%"),
        )
        axes[1].axvline(
            -relative_resolution_percent,
            color="purple",
            linestyle=":",
            linewidth=1.8,
        )
        axes[1].set_title(
            "Relative energy residual\n"
            f"Bias = {relative_bias_percent:.2f}%\n"
            "Resolution = "
            f"{relative_resolution_percent:.2f}%"
        )
        axes[1].legend(fontsize=9)
    else:
        axes[1].text(
            0.5,
            0.5,
            "Relative residual unavailable\nbecause no valid non-zero targets remain.",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
        axes[1].set_title("Relative energy residual")

    axes[1].set(
        xlabel=("(Predicted energy - true energy) / true energy [%]"),
        ylabel="Events",
    )
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        "Energy regression residuals",
        fontsize=16,
    )
    fig.tight_layout()

    save(
        fig,
        output_path(
            output_dir,
            Path("energy_residuals.png"),
        ),
        dpi,
    )


def calculate_binned_profiles(
    predictions,
    targets,
    *,
    n_bins,
):
    """Calculate energy metrics in true-energy bins."""
    residuals = predictions - targets

    target_min = float(np.min(targets))
    target_max = float(np.max(targets))

    if target_max <= target_min:
        target_max = target_min + EPSILON

    bin_edges = np.linspace(
        target_min,
        target_max,
        n_bins + 1,
    )
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    biases = np.full(
        n_bins,
        np.nan,
    )
    resolutions = np.full(
        n_bins,
        np.nan,
    )
    relative_biases = np.full(
        n_bins,
        np.nan,
    )
    relative_resolutions = np.full(
        n_bins,
        np.nan,
    )
    counts = np.zeros(
        n_bins,
        dtype=int,
    )

    for bin_index in range(n_bins):
        low = bin_edges[bin_index]
        high = bin_edges[bin_index + 1]

        if bin_index == n_bins - 1:
            in_bin = (targets >= low) & (targets <= high)
        else:
            in_bin = (targets >= low) & (targets < high)

        bin_targets = targets[in_bin]
        bin_residuals = residuals[in_bin]

        counts[bin_index] = len(bin_residuals)

        if bin_residuals.size == 0:
            continue

        biases[bin_index] = np.mean(bin_residuals)
        resolutions[bin_index] = resolution68(bin_residuals)

        relative_valid = np.abs(bin_targets) > EPSILON

        if np.any(relative_valid):
            relative_residuals = (
                bin_residuals[relative_valid] / bin_targets[relative_valid]
            )

            relative_biases[bin_index] = np.mean(relative_residuals)
            relative_resolutions[bin_index] = resolution68(relative_residuals)

    return {
        "bin_edges": bin_edges,
        "bin_centres": bin_centres,
        "biases": biases,
        "resolutions": resolutions,
        "relative_biases": relative_biases,
        "relative_resolutions": relative_resolutions,
        "counts": counts,
    }


def plot_binned_profiles(
    predictions,
    targets,
    *,
    profile_bins,
    output_dir,
    dpi,
):
    """Plot energy metrics against true energy."""
    profile = calculate_binned_profiles(
        predictions,
        targets,
        n_bins=profile_bins,
    )

    x_values = profile["bin_centres"]
    counts = profile["counts"]

    valid_absolute = (
        (counts > 0)
        & np.isfinite(profile["biases"])
        & np.isfinite(profile["resolutions"])
    )
    valid_relative = (
        (counts > 0)
        & np.isfinite(profile["relative_biases"])
        & np.isfinite(profile["relative_resolutions"])
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
        sharex=True,
    )

    for ax in axes[0, :]:
        ax.tick_params(
            axis="x",
            labelbottom=True,
        )

    axes[0, 0].plot(
        x_values[valid_absolute],
        profile["biases"][valid_absolute],
        "o-",
        color="tab:blue",
        linewidth=1.8,
        markersize=5,
    )
    axes[0, 0].axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=1.2,
    )
    axes[0, 0].set(
        xlabel=f"True energy [{UNIT}]",
        ylabel=f"Bias [{UNIT}]",
        title="Energy bias",
    )
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(
        x_values[valid_absolute],
        profile["resolutions"][valid_absolute],
        "o-",
        color="tab:orange",
        linewidth=1.8,
        markersize=5,
    )
    axes[0, 1].set(
        xlabel=f"True energy [{UNIT}]",
        ylabel=f"Resolution [{UNIT}]",
        title="Energy resolution",
    )
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(
        x_values[valid_relative],
        100.0 * profile["relative_biases"][valid_relative],
        "o-",
        color="tab:green",
        linewidth=1.8,
        markersize=5,
    )
    axes[1, 0].axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=1.2,
    )
    axes[1, 0].set(
        xlabel=f"True energy [{UNIT}]",
        ylabel="Relative bias [%]",
        title="Relative energy bias",
    )
    axes[1, 0].grid(alpha=0.3)

    relative_resolution_percent = (
        100.0 * profile["relative_resolutions"][valid_relative]
    )

    axes[1, 1].plot(
        x_values[valid_relative],
        relative_resolution_percent,
        "o-",
        color="tab:red",
        linewidth=1.8,
        markersize=5,
    )
    axes[1, 1].set(
        xlabel=f"True energy [{UNIT}]",
        ylabel="Relative resolution [%]",
        title="Relative energy resolution",
    )
    axes[1, 1].grid(alpha=0.3)

    for x_value, resolution, count in zip(
        x_values[valid_relative],
        relative_resolution_percent,
        counts[valid_relative],
        strict=True,
    ):
        axes[1, 1].annotate(
            f"N={count}",
            (x_value, resolution),
            fontsize=7,
            alpha=0.6,
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
        )

    fig.suptitle(
        "Energy regression bias and resolution profiles",
        fontsize=16,
    )
    fig.tight_layout()

    save(
        fig,
        output_path(
            output_dir,
            Path("energy_bias_resolution_profiles.png"),
        ),
        dpi,
    )


def build_parser():
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(description="Plot energy-regression performance.")

    parser.add_argument(
        "--run-dir",
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
        "--output-dir",
        type=Path,
        default=Path("plots/energy"),
    )
    parser.add_argument(
        "--minimum",
        type=float,
        default=0.5,
        help="Minimum energy used for normalization.",
    )
    parser.add_argument(
        "--maximum",
        type=float,
        default=2000.0,
        help="Maximum energy used for normalization.",
    )
    parser.add_argument(
        "--already-mev",
        action="store_true",
        help=("Treat predictions and targets as energies already expressed in MeV."),
    )
    parser.add_argument(
        "--histogram-bins",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--profile-bins",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
    )

    return parser


def main():
    """Load energy outputs and generate all energy plots."""
    args = build_parser().parse_args()

    if args.maximum <= args.minimum:
        raise ValueError("--maximum must be greater than --minimum.")

    if args.histogram_bins <= 0:
        raise ValueError("--histogram-bins must be positive.")

    if args.profile_bins <= 0:
        raise ValueError("--profile-bins must be positive.")

    outputs = load_outputs(
        args.run_dir,
        args.predictions,
        args.targets,
        args.indices,
    )

    predictions_raw = scalar(
        outputs.predictions,
        "predictions",
    )
    targets_raw = scalar(
        outputs.targets,
        "targets",
    )

    if predictions_raw.shape != targets_raw.shape:
        raise ValueError(
            "Prediction and target shapes do not match: "
            f"{predictions_raw.shape} and "
            f"{targets_raw.shape}."
        )

    valid = np.isfinite(predictions_raw) & np.isfinite(targets_raw)

    if np.any(~valid):
        n_removed = int(np.sum(~valid))
        print(f"Warning: removing {n_removed} events containing non-finite values.")

        predictions_raw = predictions_raw[valid]
        targets_raw = targets_raw[valid]

    if targets_raw.size == 0:
        raise ValueError("No valid events remain.")

    print("Normalized/raw ranges:")
    print(
        f"  Predictions: {np.min(predictions_raw):.6g} to {np.max(predictions_raw):.6g}"
    )
    print(f"  Targets:     {np.min(targets_raw):.6g} to {np.max(targets_raw):.6g}")

    if args.already_mev:
        predictions = predictions_raw
        targets = targets_raw

        print("Inputs are treated as energies already expressed in MeV.")
    else:
        predictions = inverse_minmax(
            predictions_raw,
            args.minimum,
            args.maximum,
        )
        targets = inverse_minmax(
            targets_raw,
            args.minimum,
            args.maximum,
        )

        print(
            "Applied inverse energy normalization: "
            f"[0, 1] to [{args.minimum}, "
            f"{args.maximum}] {UNIT}."
        )

    print()
    print(f"Prediction range [{UNIT}]:")
    print(f"  {np.min(predictions):.6g} to {np.max(predictions):.6g}")
    print(f"Target range [{UNIT}]:")
    print(f"  {np.min(targets):.6g} to {np.max(targets):.6g}")

    print_statistics(
        predictions,
        targets,
    )

    plot_prediction_vs_truth(
        predictions,
        targets,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )

    plot_residuals(
        predictions,
        targets,
        histogram_bins=args.histogram_bins,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )

    plot_binned_profiles(
        predictions,
        targets,
        profile_bins=args.profile_bins,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
