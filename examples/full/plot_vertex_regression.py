"""Plot full vertex-regression performance.

Edit the settings below and run:

    uv run python examples/full/plot_vertex_regression.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis_tools.analysis import analyze_vertex_regression
from analysis_tools.data import as_vector_array
from analysis_tools.io import load_array_pair
from analysis_tools.metrics import (
    bias_magnitude,
    bias_vector,
    binned_absolute_quantile,
    binned_bias,
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

OUTPUT_DIRECTORY = Path("outputs/vertex")


# ============================================================
# Detector and normalization
# ============================================================

DETECTOR_RADIUS_CM = 1690.0
DETECTOR_HALF_HEIGHT_CM = 1810.0

VERTEX_MINIMUM_CM = [-1690.0, -1690.0, -1810.0]
VERTEX_MAXIMUM_CM = [1690.0, 1690.0, 1810.0]


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

HISTOGRAM_COLOR = "tab:blue"
ERROR_COLOR = "tab:green"

BIAS_COLOR = "red"
RESOLUTION_COLOR = "purple"
REFERENCE_COLOR = "black"

PROFILE_BIAS_COLOR = "tab:blue"
PROFILE_RESOLUTION_COLOR = "tab:orange"

SCATTER_COLOR = "tab:blue"
SCATTER_ALPHA = 0.25
SCATTER_SIZE = 5.0


# ============================================================
# Output filenames
# ============================================================

PREDICTION_FILENAME = "vertex_prediction_vs_truth_cylindrical_cm.png"
RESIDUAL_FILENAME = "vertex_residuals_cylindrical_cm.png"
PROFILE_FILENAME = "vertex_cylindrical_profiles_cm.png"


def make_profile_edges(
    values: np.ndarray,
    number_of_bins: int,
) -> np.ndarray:
    """Create equally spaced edges spanning the supplied values."""
    minimum = float(np.min(values))
    maximum = float(np.max(values))

    if maximum <= minimum:
        raise ValueError("Profile values must span a non-zero range.")

    return np.linspace(
        minimum,
        maximum,
        number_of_bins + 1,
    )


def print_statistics(result) -> None:
    """Print Cartesian, cylindrical, Wall, and 3D statistics."""
    component_biases = np.asarray(
        bias_vector(result.residuals),
        dtype=float,
    )

    print("\n========== Vertex regression statistics ==========")
    print(f"Number of events: {result.event_count:,}")
    print(f"Removed non-finite events: {result.removed_event_count:,}")
    print("Unit: cm")

    print("\nCartesian residuals:")

    for index, label in enumerate(("x", "y", "z")):
        residuals = result.residuals[:, index]
        resolution = quantile(
            np.abs(residuals),
            QUANTILE_LEVEL,
        )

        print(
            f"  {label}: "
            f"bias={component_biases[index]:.6g}, "
            f"resolution={resolution:.6g} cm"
        )

    print("\nCylindrical and Wall residuals:")

    for label, residuals in (
        ("R", result.radius_residuals),
        ("Wall", result.wall_residuals),
    ):
        print(
            f"  {label}: "
            f"bias={np.mean(residuals):.6g}, "
            f"resolution="
            f"{quantile(np.abs(residuals), QUANTILE_LEVEL):.6g} cm"
        )

    print("\nBias vector:")

    print(
        "  "
        f"({component_biases[0]:+.6g}, "
        f"{component_biases[1]:+.6g}, "
        f"{component_biases[2]:+.6g}) cm"
    )
    print(f"  Magnitude: {bias_magnitude(result.residuals):.6g} cm")

    print("\n3D position error:")
    print(f"  Mean: {np.mean(result.errors):.6g} cm")
    print(f"  Median: {np.median(result.errors):.6g} cm")
    print(
        f"  {100.0 * QUANTILE_LEVEL:.0f}% quantile: "
        f"{quantile(result.errors, QUANTILE_LEVEL):.6g} cm"
    )
    print(f"  RMS: {np.sqrt(np.mean(result.errors**2)):.6g} cm")

    print("==================================================\n")


def plot_prediction_vs_truth(result):
    """Plot x, y, z, R, and Wall predictions against truth."""
    figure, axes = plt.subplots(
        1,
        5,
        figsize=(28.0, 5.0),
    )

    quantities = (
        (
            result.targets[:, 0],
            result.predictions[:, 0],
            "x",
        ),
        (
            result.targets[:, 1],
            result.predictions[:, 1],
            "y",
        ),
        (
            result.targets[:, 2],
            result.predictions[:, 2],
            "z",
        ),
        (
            result.target_radius,
            result.predicted_radius,
            "R",
        ),
        (
            result.target_wall,
            result.predicted_wall,
            "Wall",
        ),
    )

    for axis, (targets, predictions, label) in zip(
        axes,
        quantities,
        strict=True,
    ):
        axis.scatter(
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

        axis.plot(
            [minimum, maximum],
            [minimum, maximum],
            color=REFERENCE_COLOR,
            linestyle="--",
            linewidth=1.5,
        )

        axis.set_xlabel(f"True {label} [cm]")
        axis.set_ylabel(f"Predicted {label} [cm]")
        axis.set_title(f"{label} prediction")
        axis.grid(alpha=0.3)

    figure.suptitle(
        "GAT vertex regression: prediction vs truth",
        fontsize=16,
    )
    figure.tight_layout()

    return figure, axes


def plot_residuals(result):
    """Plot Cartesian, radial, Wall, and 3D errors."""
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(18.0, 10.0),
    )

    residual_quantities = (
        (result.x_residuals, "x"),
        (result.y_residuals, "y"),
        (result.z_residuals, "z"),
        (result.radius_residuals, "R"),
        (result.wall_residuals, "Wall"),
    )

    for axis, (residuals, label) in zip(
        axes.flat[:5],
        residual_quantities,
        strict=True,
    ):
        component_bias = float(np.mean(residuals))
        resolution = quantile(
            np.abs(residuals),
            QUANTILE_LEVEL,
        )

        plot_histogram(
            residuals,
            ax=axis,
            bins=N_HISTOGRAM_BINS,
            xlabel=f"Predicted {label} - true {label} [cm]",
            ylabel="Events",
            title=f"{label} residual",
            color=HISTOGRAM_COLOR,
            histtype="step",
            linewidth=1.5,
        )

        axis.axvline(
            0.0,
            color=REFERENCE_COLOR,
            linestyle="--",
            linewidth=1.2,
        )
        axis.axvline(
            component_bias,
            color=BIAS_COLOR,
            linewidth=1.5,
            label=f"Bias = {component_bias:.2f}",
        )
        axis.axvline(
            resolution,
            color=RESOLUTION_COLOR,
            linestyle=":",
            linewidth=1.5,
        )
        axis.axvline(
            -resolution,
            color=RESOLUTION_COLOR,
            linestyle=":",
            linewidth=1.5,
            label=(f"{100.0 * QUANTILE_LEVEL:.0f}% resolution = {resolution:.2f}"),
        )

        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)

    error_axis = axes.flat[5]

    median_error = float(np.median(result.errors))
    error_quantile = quantile(
        result.errors,
        QUANTILE_LEVEL,
    )

    plot_histogram(
        result.errors,
        ax=error_axis,
        bins=N_HISTOGRAM_BINS,
        xlabel="3D position error [cm]",
        ylabel="Events",
        title="3D position error",
        color=ERROR_COLOR,
        histtype="step",
        linewidth=1.5,
    )

    error_axis.axvline(
        median_error,
        color=BIAS_COLOR,
        linestyle="--",
        linewidth=1.5,
        label=f"Median = {median_error:.2f}",
    )
    error_axis.axvline(
        error_quantile,
        color=RESOLUTION_COLOR,
        linestyle=":",
        linewidth=1.5,
        label=(f"{100.0 * QUANTILE_LEVEL:.0f}% quantile = {error_quantile:.2f}"),
    )

    error_axis.grid(alpha=0.3)
    error_axis.legend(fontsize=8)

    figure.suptitle(
        "GAT vertex regression residuals",
        fontsize=16,
    )
    figure.tight_layout()

    return figure, axes


def plot_profiles(result):
    """Plot bias and resolution versus true R, z, and Wall."""
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(18.0, 9.0),
    )

    profiles = (
        (
            result.target_radius,
            result.radius_residuals,
            "True R [cm]",
            "R residual",
        ),
        (
            result.targets[:, 2],
            result.z_residuals,
            "True z [cm]",
            "z residual",
        ),
        (
            result.target_wall,
            result.errors,
            "True Wall [cm]",
            "3D position error",
        ),
    )

    for column, (x_values, values, xlabel, title) in enumerate(profiles):
        edges = make_profile_edges(
            x_values,
            N_PROFILE_BINS,
        )

        centers, biases, bias_counts = binned_bias(
            x_values,
            values,
            edges,
            minimum_events=PROFILE_MINIMUM_EVENTS,
        )
        _, resolutions, resolution_counts = binned_absolute_quantile(
            x_values,
            values,
            edges,
            quantile_level=QUANTILE_LEVEL,
            minimum_events=PROFILE_MINIMUM_EVENTS,
        )

        valid_bias = np.isfinite(biases) & (bias_counts >= PROFILE_MINIMUM_EVENTS)
        valid_resolution = np.isfinite(resolutions) & (
            resolution_counts >= PROFILE_MINIMUM_EVENTS
        )

        bias_axis = axes[0, column]
        resolution_axis = axes[1, column]

        plot_series(
            centers[valid_bias],
            biases[valid_bias],
            ax=bias_axis,
            xlabel=xlabel,
            ylabel="Bias [cm]",
            title=f"{title} bias",
            color=PROFILE_BIAS_COLOR,
            marker="o",
            linestyle="-",
        )

        bias_axis.axhline(
            0.0,
            color=REFERENCE_COLOR,
            linestyle="--",
            linewidth=1.2,
        )
        bias_axis.grid(alpha=0.3)

        plot_series(
            centers[valid_resolution],
            resolutions[valid_resolution],
            ax=resolution_axis,
            xlabel=xlabel,
            ylabel=(f"{100.0 * QUANTILE_LEVEL:.0f}% resolution [cm]"),
            title=f"{title} resolution",
            color=PROFILE_RESOLUTION_COLOR,
            marker="o",
            linestyle="-",
        )

        resolution_axis.grid(alpha=0.3)

    figure.suptitle(
        "Vertex regression in SK cylindrical geometry",
        fontsize=16,
    )
    figure.tight_layout()

    return figure, axes


def main() -> None:
    predictions_normalized, targets_normalized = load_array_pair(
        PREDICTIONS_PATH,
        TARGETS_PATH,
        first_name="predictions",
        second_name="targets",
    )

    predictions_normalized = as_vector_array(
        predictions_normalized,
        size=3,
        name="predictions",
    )
    targets_normalized = as_vector_array(
        targets_normalized,
        size=3,
        name="targets",
    )

    transform = MinMaxTransform(
        minimum=VERTEX_MINIMUM_CM,
        maximum=VERTEX_MAXIMUM_CM,
    )

    result = analyze_vertex_regression(
        transform.inverse(predictions_normalized),
        transform.inverse(targets_normalized),
        detector_radius=DETECTOR_RADIUS_CM,
        detector_half_height=DETECTOR_HALF_HEIGHT_CM,
    )

    print_statistics(result)

    prediction_figure, _ = plot_prediction_vs_truth(result)
    residual_figure, _ = plot_residuals(result)
    profile_figure, _ = plot_profiles(result)

    output_figures = (
        (
            prediction_figure,
            OUTPUT_DIRECTORY / PREDICTION_FILENAME,
        ),
        (
            residual_figure,
            OUTPUT_DIRECTORY / RESIDUAL_FILENAME,
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
