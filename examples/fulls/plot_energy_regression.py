"""Plot full energy-regression performance.

Edit the settings below and run:

    uv run python examples/plot_energy_regression.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis_tools.data import as_scalar_array
from analysis_tools.io import load_array_pair
from analysis_tools.metrics import (
    binned_absolute_quantile,
    binned_bias,
    energy_residuals,
    fractional_energy_residuals,
    quantile,
)
from analysis_tools.plotting import save_figure
from analysis_tools.plotting.common import plot_histogram, plot_series
from analysis_tools.transforms import MinMaxTransform

# ============================================================
# Input and output
# ============================================================

PREDICTIONS_PATH = Path("path/to/outputs/preds.npy")
TARGETS_PATH = Path("path/to/outputs/targets.npy")

OUTPUT_DIRECTORY = Path("outputs/energy")


# ============================================================
# Normalization
# ============================================================

ENERGY_MINIMUM_MEV = 0.5
ENERGY_MAXIMUM_MEV = 2000.0


# ============================================================
# Analysis settings
# ============================================================

QUANTILE_LEVEL = 0.68

N_HISTOGRAM_BINS = 100
N_PROFILE_BINS = 20
PROFILE_MINIMUM_EVENTS = 1


# ============================================================
# Plot settings
# ============================================================

DPI = 200
SHOW_FIGURES = True

SCATTER_COLOR = "tab:blue"
SCATTER_ALPHA = 0.25
SCATTER_SIZE = 5.0

HISTOGRAM_COLOR = "tab:blue"
RELATIVE_ERROR_COLOR = "tab:green"

BIAS_COLOR = "red"
RESOLUTION_COLOR = "purple"
REFERENCE_COLOR = "black"

PROFILE_BIAS_COLOR = "tab:blue"
PROFILE_RESOLUTION_COLOR = "tab:orange"


# ============================================================
# Output filenames
# ============================================================

RESULT_FILENAME = "energy_regression_results.png"
PROFILE_FILENAME = "energy_regression_profiles.png"


def make_profile_edges(
    values: np.ndarray,
    number_of_bins: int,
) -> np.ndarray:
    """Create equally spaced bin edges."""
    finite_values = values[np.isfinite(values)]

    if finite_values.size == 0:
        raise ValueError("No finite values are available.")

    minimum = float(np.min(finite_values))
    maximum = float(np.max(finite_values))

    if maximum <= minimum:
        raise ValueError("Profile values must span a non-zero range.")

    return np.linspace(
        minimum,
        maximum,
        number_of_bins + 1,
    )


def print_statistics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> None:
    """Print absolute and fractional energy statistics."""
    residuals = energy_residuals(
        predictions,
        targets,
    )
    fractional_residuals = fractional_energy_residuals(
        predictions,
        targets,
    )

    finite_fractional = fractional_residuals[np.isfinite(fractional_residuals)]

    bias = float(np.mean(residuals))
    resolution = quantile(
        np.abs(residuals),
        QUANTILE_LEVEL,
    )

    print("\n========== Energy regression statistics ==========")
    print(f"Number of events: {len(predictions):,}")
    print("Unit: MeV")

    print("\nPrediction and target:")
    print(
        f"  Prediction range: "
        f"{np.min(predictions):.6g} to "
        f"{np.max(predictions):.6g} MeV"
    )
    print(f"  Target range: {np.min(targets):.6g} to {np.max(targets):.6g} MeV")

    print("\nAbsolute residual:")
    print(f"  Bias: {bias:+.6g} MeV")
    print(f"  Mean absolute error: {np.mean(np.abs(residuals)):.6g} MeV")
    print(f"  Median absolute error: {np.median(np.abs(residuals)):.6g} MeV")
    print(f"  {100.0 * QUANTILE_LEVEL:.0f}% resolution: {resolution:.6g} MeV")
    print(f"  RMS: {np.sqrt(np.mean(residuals**2)):.6g} MeV")

    if finite_fractional.size:
        fractional_bias = float(np.mean(finite_fractional))
        fractional_resolution = quantile(
            np.abs(finite_fractional),
            QUANTILE_LEVEL,
        )

        print("\nFractional residual:")
        print(f"  Bias: {100.0 * fractional_bias:+.6g}%")
        print(
            f"  Median absolute error: "
            f"{100.0 * np.median(np.abs(finite_fractional)):.6g}%"
        )
        print(
            f"  {100.0 * QUANTILE_LEVEL:.0f}% resolution: "
            f"{100.0 * fractional_resolution:.6g}%"
        )

    print("=================================================\n")


def plot_energy_result(
    predictions: np.ndarray,
    targets: np.ndarray,
):
    """Plot prediction, residual, and fractional residual."""
    residuals = energy_residuals(
        predictions,
        targets,
    )
    fractional_residuals = fractional_energy_residuals(
        predictions,
        targets,
    )

    finite_fractional = np.isfinite(fractional_residuals)

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(18.0, 5.0),
    )

    prediction_axis = axes[0]
    prediction_axis.scatter(
        targets,
        predictions,
        color=SCATTER_COLOR,
        alpha=SCATTER_ALPHA,
        s=SCATTER_SIZE,
        rasterized=True,
    )

    minimum = float(
        min(
            np.min(targets),
            np.min(predictions),
        )
    )
    maximum = float(
        max(
            np.max(targets),
            np.max(predictions),
        )
    )

    prediction_axis.plot(
        [minimum, maximum],
        [minimum, maximum],
        color=REFERENCE_COLOR,
        linestyle="--",
        linewidth=1.5,
    )
    prediction_axis.set_xlabel("True energy [MeV]")
    prediction_axis.set_ylabel("Predicted energy [MeV]")
    prediction_axis.set_title("Energy prediction")
    prediction_axis.grid(alpha=0.3)

    residual_bias = float(np.mean(residuals))
    residual_resolution = quantile(
        np.abs(residuals),
        QUANTILE_LEVEL,
    )

    residual_axis = axes[1]

    plot_histogram(
        residuals,
        ax=residual_axis,
        bins=N_HISTOGRAM_BINS,
        xlabel="Predicted - true energy [MeV]",
        ylabel="Events",
        title="Energy residual",
        color=HISTOGRAM_COLOR,
        histtype="step",
        linewidth=1.5,
    )

    residual_axis.axvline(
        0.0,
        color=REFERENCE_COLOR,
        linestyle="--",
        linewidth=1.2,
    )
    residual_axis.axvline(
        residual_bias,
        color=BIAS_COLOR,
        linewidth=1.5,
        label=f"Bias = {residual_bias:.2f} MeV",
    )
    residual_axis.axvline(
        residual_resolution,
        color=RESOLUTION_COLOR,
        linestyle=":",
        linewidth=1.5,
    )
    residual_axis.axvline(
        -residual_resolution,
        color=RESOLUTION_COLOR,
        linestyle=":",
        linewidth=1.5,
        label=(
            f"{100.0 * QUANTILE_LEVEL:.0f}% resolution = {residual_resolution:.2f} MeV"
        ),
    )
    residual_axis.grid(alpha=0.3)
    residual_axis.legend(fontsize=8)

    fractional_axis = axes[2]

    if np.any(finite_fractional):
        fractional_values = 100.0 * fractional_residuals[finite_fractional]
        fractional_bias = float(np.mean(fractional_values))
        fractional_resolution = quantile(
            np.abs(fractional_values),
            QUANTILE_LEVEL,
        )

        plot_histogram(
            fractional_values,
            ax=fractional_axis,
            bins=N_HISTOGRAM_BINS,
            xlabel="Fractional energy residual [%]",
            ylabel="Events",
            title="Fractional energy residual",
            color=RELATIVE_ERROR_COLOR,
            histtype="step",
            linewidth=1.5,
        )

        fractional_axis.axvline(
            0.0,
            color=REFERENCE_COLOR,
            linestyle="--",
            linewidth=1.2,
        )
        fractional_axis.axvline(
            fractional_bias,
            color=BIAS_COLOR,
            linewidth=1.5,
            label=f"Bias = {fractional_bias:.2f}%",
        )
        fractional_axis.axvline(
            fractional_resolution,
            color=RESOLUTION_COLOR,
            linestyle=":",
            linewidth=1.5,
        )
        fractional_axis.axvline(
            -fractional_resolution,
            color=RESOLUTION_COLOR,
            linestyle=":",
            linewidth=1.5,
            label=(
                f"{100.0 * QUANTILE_LEVEL:.0f}% resolution "
                f"= {fractional_resolution:.2f}%"
            ),
        )
        fractional_axis.grid(alpha=0.3)
        fractional_axis.legend(fontsize=8)
    else:
        fractional_axis.text(
            0.5,
            0.5,
            "No finite fractional residuals",
            ha="center",
            va="center",
            transform=fractional_axis.transAxes,
        )
        fractional_axis.set_axis_off()

    figure.suptitle(
        "Energy regression performance",
        fontsize=16,
    )
    figure.tight_layout()

    return figure, axes


def plot_energy_profiles(
    predictions: np.ndarray,
    targets: np.ndarray,
):
    """Plot absolute and fractional performance versus true energy."""
    residuals = energy_residuals(
        predictions,
        targets,
    )
    fractional_residuals = fractional_energy_residuals(
        predictions,
        targets,
    )

    edges = make_profile_edges(
        targets,
        N_PROFILE_BINS,
    )

    centers, absolute_bias, absolute_bias_counts = binned_bias(
        targets,
        residuals,
        edges,
        minimum_events=PROFILE_MINIMUM_EVENTS,
    )
    _, absolute_resolution, absolute_resolution_counts = binned_absolute_quantile(
        targets,
        residuals,
        edges,
        quantile_level=QUANTILE_LEVEL,
        minimum_events=PROFILE_MINIMUM_EVENTS,
    )

    _, fractional_bias, fractional_bias_counts = binned_bias(
        targets,
        fractional_residuals,
        edges,
        minimum_events=PROFILE_MINIMUM_EVENTS,
    )
    _, fractional_resolution, fractional_resolution_counts = binned_absolute_quantile(
        targets,
        fractional_residuals,
        edges,
        quantile_level=QUANTILE_LEVEL,
        minimum_events=PROFILE_MINIMUM_EVENTS,
    )

    valid_absolute_bias = np.isfinite(absolute_bias) & (
        absolute_bias_counts >= PROFILE_MINIMUM_EVENTS
    )
    valid_absolute_resolution = np.isfinite(absolute_resolution) & (
        absolute_resolution_counts >= PROFILE_MINIMUM_EVENTS
    )
    valid_fractional_bias = np.isfinite(fractional_bias) & (
        fractional_bias_counts >= PROFILE_MINIMUM_EVENTS
    )
    valid_fractional_resolution = np.isfinite(fractional_resolution) & (
        fractional_resolution_counts >= PROFILE_MINIMUM_EVENTS
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13.0, 10.0),
    )

    plot_series(
        centers[valid_absolute_bias],
        absolute_bias[valid_absolute_bias],
        ax=axes[0, 0],
        xlabel="True energy [MeV]",
        ylabel="Bias [MeV]",
        title="Absolute energy bias",
        color=PROFILE_BIAS_COLOR,
        marker="o",
    )
    axes[0, 0].axhline(
        0.0,
        color=REFERENCE_COLOR,
        linestyle="--",
        linewidth=1.2,
    )
    axes[0, 0].grid(alpha=0.3)

    plot_series(
        centers[valid_absolute_resolution],
        absolute_resolution[valid_absolute_resolution],
        ax=axes[1, 0],
        xlabel="True energy [MeV]",
        ylabel=(f"{100.0 * QUANTILE_LEVEL:.0f}% resolution [MeV]"),
        title="Absolute energy resolution",
        color=PROFILE_RESOLUTION_COLOR,
        marker="o",
    )
    axes[1, 0].grid(alpha=0.3)

    plot_series(
        centers[valid_fractional_bias],
        100.0 * fractional_bias[valid_fractional_bias],
        ax=axes[0, 1],
        xlabel="True energy [MeV]",
        ylabel="Fractional bias [%]",
        title="Fractional energy bias",
        color=PROFILE_BIAS_COLOR,
        marker="o",
    )
    axes[0, 1].axhline(
        0.0,
        color=REFERENCE_COLOR,
        linestyle="--",
        linewidth=1.2,
    )
    axes[0, 1].grid(alpha=0.3)

    plot_series(
        centers[valid_fractional_resolution],
        100.0 * fractional_resolution[valid_fractional_resolution],
        ax=axes[1, 1],
        xlabel="True energy [MeV]",
        ylabel=(f"{100.0 * QUANTILE_LEVEL:.0f}% fractional resolution [%]"),
        title="Fractional energy resolution",
        color=PROFILE_RESOLUTION_COLOR,
        marker="o",
    )
    axes[1, 1].grid(alpha=0.3)

    figure.suptitle(
        "Energy performance versus true energy",
        fontsize=16,
    )
    figure.tight_layout()

    return figure, axes


def main() -> None:
    predictions, targets = load_array_pair(
        PREDICTIONS_PATH,
        TARGETS_PATH,
        first_name="predictions",
        second_name="targets",
    )

    predictions = as_scalar_array(
        predictions,
        name="predictions",
    ).astype(float)
    targets = as_scalar_array(
        targets,
        name="targets",
    ).astype(float)

    finite = np.isfinite(predictions)
    finite &= np.isfinite(targets)

    removed_event_count = int(np.count_nonzero(~finite))

    predictions = predictions[finite]
    targets = targets[finite]

    if len(predictions) == 0:
        raise ValueError("No finite energy events remain.")

    transform = MinMaxTransform(
        minimum=ENERGY_MINIMUM_MEV,
        maximum=ENERGY_MAXIMUM_MEV,
    )

    predictions_mev = transform.inverse(predictions)
    targets_mev = transform.inverse(targets)

    print(f"Removed non-finite events: {removed_event_count:,}")
    print_statistics(
        predictions_mev,
        targets_mev,
    )

    result_figure, _ = plot_energy_result(
        predictions_mev,
        targets_mev,
    )
    profile_figure, _ = plot_energy_profiles(
        predictions_mev,
        targets_mev,
    )

    output_figures = (
        (
            result_figure,
            OUTPUT_DIRECTORY / RESULT_FILENAME,
        ),
        (
            profile_figure,
            OUTPUT_DIRECTORY / PROFILE_FILENAME,
        ),
    )

    for figure, path in output_figures:
        save_figure(
            figure,
            path,
            dpi=DPI,
            close=not SHOW_FIGURES,
        )
        print(f"Saved: {path}")

    if SHOW_FIGURES:
        plt.show()


if __name__ == "__main__":
    main()
