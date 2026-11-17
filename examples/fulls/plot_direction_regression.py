"""Plot full direction-regression performance.

Edit the settings below and run:

    uv run python examples/plot_direction_regression.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis_tools.data import as_vector_array
from analysis_tools.io import load_array_pair
from analysis_tools.metrics import (
    binned_absolute_quantile,
    binned_bias,
    direction_errors,
    quantile,
)
from analysis_tools.plotting import save_figure
from analysis_tools.plotting.common import plot_histogram, plot_series
from analysis_tools.transforms import MinMaxTransform
from analysis_tools.transforms.geometry import normalize_vectors

# ============================================================
# Input and output
# ============================================================

PREDICTIONS_PATH = Path("path/to/outputs/preds.npy")
TARGETS_PATH = Path("path/to/outputs/targets.npy")

OUTPUT_DIRECTORY = Path("outputs/direction")


# ============================================================
# Normalization
# ============================================================

# Set False if outputs already represent components in [-1, 1].
APPLY_INVERSE_COMPONENT_NORMALIZATION = True

DIRECTION_MINIMUM = [-1.0, -1.0, -1.0]
DIRECTION_MAXIMUM = [1.0, 1.0, 1.0]


# ============================================================
# Analysis settings
# ============================================================

QUANTILE_LEVEL = 0.68
N_HISTOGRAM_BINS = 100
N_PROFILE_BINS = 20
PROFILE_MINIMUM_EVENTS = 1

COMPONENT_LABELS = ("dx", "dy", "dz")


# ============================================================
# Plot settings
# ============================================================

DPI = 200
SHOW_FIGURES = True

SCATTER_COLOR = "tab:blue"
SCATTER_ALPHA = 0.25
SCATTER_SIZE = 5.0

HISTOGRAM_COLOR = "tab:blue"
ANGLE_COLOR = "tab:green"

BIAS_COLOR = "red"
RESOLUTION_COLOR = "purple"
REFERENCE_COLOR = "black"

PROFILE_BIAS_COLOR = "tab:blue"
PROFILE_RESOLUTION_COLOR = "tab:orange"


# ============================================================
# Output filenames
# ============================================================

RESULT_FILENAME = "direction_regression_results.png"
PROFILE_FILENAME = "direction_regression_profiles.png"
ANGLE_FILENAME = "direction_angle_results.png"


def wrapped_angle_difference(
    predicted: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Return predicted minus target angle in [-180, 180)."""
    return (predicted - target + 180.0) % 360.0 - 180.0


def direction_angles(
    vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return zenith and azimuth angles in degrees."""
    unit_vectors = normalize_vectors(vectors)

    zenith = np.degrees(np.arccos(np.clip(unit_vectors[:, 2], -1.0, 1.0)))
    azimuth = np.degrees(
        np.arctan2(
            unit_vectors[:, 1],
            unit_vectors[:, 0],
        )
    )

    return zenith, azimuth


def make_profile_edges(
    values: np.ndarray,
    number_of_bins: int,
) -> np.ndarray:
    """Create equally spaced bin edges."""
    minimum = float(np.min(values))
    maximum = float(np.max(values))

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
    """Print component, norm, and angular statistics."""
    residuals = predictions - targets
    opening_angles = direction_errors(
        predictions,
        targets,
        degrees=True,
    )

    prediction_norms = np.linalg.norm(
        predictions,
        axis=1,
    )
    target_norms = np.linalg.norm(
        targets,
        axis=1,
    )

    print("\n========== Direction regression statistics ==========")
    print(f"Number of events: {len(predictions):,}")

    print("\nDirection-component residuals:")

    for index, label in enumerate(COMPONENT_LABELS):
        values = residuals[:, index]
        bias = float(np.mean(values))
        resolution = quantile(
            np.abs(values),
            QUANTILE_LEVEL,
        )

        print(f"  {label}: bias={bias:+.6g}, resolution={resolution:.6g}")

    print("\nVector norms:")
    print(
        "  Prediction: "
        f"mean={np.mean(prediction_norms):.6g}, "
        f"std={np.std(prediction_norms):.6g}"
    )
    print(f"  Target: mean={np.mean(target_norms):.6g}, std={np.std(target_norms):.6g}")

    print("\nOpening-angle error:")
    print(f"  Mean: {np.mean(opening_angles):.6g} degree")
    print(f"  Median: {np.median(opening_angles):.6g} degree")
    print(
        f"  {100.0 * QUANTILE_LEVEL:.0f}% quantile: "
        f"{quantile(opening_angles, QUANTILE_LEVEL):.6g} degree"
    )
    print(f"  RMS: {np.sqrt(np.mean(opening_angles**2)):.6g} degree")

    print("=====================================================\n")


def plot_direction_result(
    predictions: np.ndarray,
    targets: np.ndarray,
):
    """Plot component predictions, residuals, and opening angles."""
    residuals = predictions - targets
    opening_angles = direction_errors(
        predictions,
        targets,
        degrees=True,
    )

    figure, axes = plt.subplots(
        3,
        3,
        figsize=(17.0, 14.0),
    )

    for index, label in enumerate(COMPONENT_LABELS):
        prediction_axis = axes[0, index]
        residual_axis = axes[1, index]

        prediction_axis.scatter(
            targets[:, index],
            predictions[:, index],
            color=SCATTER_COLOR,
            alpha=SCATTER_ALPHA,
            s=SCATTER_SIZE,
            rasterized=True,
        )

        minimum = float(
            min(
                np.min(targets[:, index]),
                np.min(predictions[:, index]),
            )
        )
        maximum = float(
            max(
                np.max(targets[:, index]),
                np.max(predictions[:, index]),
            )
        )

        prediction_axis.plot(
            [minimum, maximum],
            [minimum, maximum],
            color=REFERENCE_COLOR,
            linestyle="--",
            linewidth=1.5,
        )
        prediction_axis.set_xlabel(f"True {label}")
        prediction_axis.set_ylabel(f"Predicted {label}")
        prediction_axis.set_title(f"{label} prediction")
        prediction_axis.grid(alpha=0.3)

        component_residuals = residuals[:, index]
        component_bias = float(np.mean(component_residuals))
        component_resolution = quantile(
            np.abs(component_residuals),
            QUANTILE_LEVEL,
        )

        plot_histogram(
            component_residuals,
            ax=residual_axis,
            bins=N_HISTOGRAM_BINS,
            xlabel=f"Predicted {label} - true {label}",
            ylabel="Events",
            title=f"{label} residual",
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
            component_bias,
            color=BIAS_COLOR,
            linewidth=1.5,
            label=f"Bias = {component_bias:.4f}",
        )
        residual_axis.axvline(
            component_resolution,
            color=RESOLUTION_COLOR,
            linestyle=":",
            linewidth=1.5,
        )
        residual_axis.axvline(
            -component_resolution,
            color=RESOLUTION_COLOR,
            linestyle=":",
            linewidth=1.5,
            label=(
                f"{100.0 * QUANTILE_LEVEL:.0f}% resolution = {component_resolution:.4f}"
            ),
        )
        residual_axis.grid(alpha=0.3)
        residual_axis.legend(fontsize=8)

    prediction_norms = np.linalg.norm(
        predictions,
        axis=1,
    )
    target_norms = np.linalg.norm(
        targets,
        axis=1,
    )

    plot_histogram(
        prediction_norms,
        ax=axes[2, 0],
        bins=N_HISTOGRAM_BINS,
        xlabel="Vector norm",
        ylabel="Events",
        title="Direction-vector norms",
        label="Prediction",
        color="tab:blue",
        histtype="step",
        linewidth=1.5,
    )
    plot_histogram(
        target_norms,
        ax=axes[2, 0],
        bins=N_HISTOGRAM_BINS,
        label="Target",
        color="tab:orange",
        histtype="step",
        linewidth=1.5,
    )
    axes[2, 0].axvline(
        1.0,
        color=REFERENCE_COLOR,
        linestyle="--",
        linewidth=1.2,
    )
    axes[2, 0].grid(alpha=0.3)
    axes[2, 0].legend()

    axes[2, 1].scatter(
        targets[:, 2],
        opening_angles,
        color=SCATTER_COLOR,
        alpha=SCATTER_ALPHA,
        s=SCATTER_SIZE,
        rasterized=True,
    )
    axes[2, 1].set_xlabel("True dz")
    axes[2, 1].set_ylabel("Opening-angle error [degree]")
    axes[2, 1].set_title("Angular error versus true dz")
    axes[2, 1].grid(alpha=0.3)

    angle_quantile = quantile(
        opening_angles,
        QUANTILE_LEVEL,
    )
    angle_median = float(np.median(opening_angles))

    plot_histogram(
        opening_angles,
        ax=axes[2, 2],
        bins=N_HISTOGRAM_BINS,
        xlabel="Opening-angle error [degree]",
        ylabel="Events",
        title="Opening-angle error",
        color=ANGLE_COLOR,
        histtype="step",
        linewidth=1.5,
    )
    axes[2, 2].axvline(
        angle_median,
        color=BIAS_COLOR,
        linestyle="--",
        linewidth=1.5,
        label=f"Median = {angle_median:.2f} degree",
    )
    axes[2, 2].axvline(
        angle_quantile,
        color=RESOLUTION_COLOR,
        linestyle=":",
        linewidth=1.5,
        label=(f"{100.0 * QUANTILE_LEVEL:.0f}% quantile = {angle_quantile:.2f} degree"),
    )
    axes[2, 2].grid(alpha=0.3)
    axes[2, 2].legend(fontsize=8)

    figure.suptitle(
        "Direction regression performance",
        fontsize=16,
    )
    figure.tight_layout()

    return figure, axes


def plot_binned_profiles(
    predictions: np.ndarray,
    targets: np.ndarray,
):
    """Plot angular performance versus true dz."""
    opening_angles = direction_errors(
        predictions,
        targets,
        degrees=True,
    )

    edges = make_profile_edges(
        targets[:, 2],
        N_PROFILE_BINS,
    )

    centers, angular_bias, bias_counts = binned_bias(
        targets[:, 2],
        opening_angles,
        edges,
        minimum_events=PROFILE_MINIMUM_EVENTS,
    )
    _, angular_resolution, resolution_counts = binned_absolute_quantile(
        targets[:, 2],
        opening_angles,
        edges,
        quantile_level=QUANTILE_LEVEL,
        minimum_events=PROFILE_MINIMUM_EVENTS,
    )

    valid_bias = np.isfinite(angular_bias) & (bias_counts >= PROFILE_MINIMUM_EVENTS)
    valid_resolution = np.isfinite(angular_resolution) & (
        resolution_counts >= PROFILE_MINIMUM_EVENTS
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.0, 5.0),
    )

    plot_series(
        centers[valid_bias],
        angular_bias[valid_bias],
        ax=axes[0],
        xlabel="True dz",
        ylabel="Mean opening-angle error [degree]",
        title="Mean angular error versus true dz",
        color=PROFILE_BIAS_COLOR,
        marker="o",
        linestyle="-",
    )
    axes[0].grid(alpha=0.3)

    plot_series(
        centers[valid_resolution],
        angular_resolution[valid_resolution],
        ax=axes[1],
        xlabel="True dz",
        ylabel=(f"{100.0 * QUANTILE_LEVEL:.0f}% angular resolution [degree]"),
        title="Angular resolution versus true dz",
        color=PROFILE_RESOLUTION_COLOR,
        marker="o",
        linestyle="-",
    )
    axes[1].grid(alpha=0.3)

    figure.suptitle(
        "Direction performance versus true zenith component",
        fontsize=16,
    )
    figure.tight_layout()

    return figure, axes


def plot_angle_results(
    predictions: np.ndarray,
    targets: np.ndarray,
):
    """Plot zenith and azimuth prediction performance."""
    predicted_zenith, predicted_azimuth = direction_angles(predictions)
    target_zenith, target_azimuth = direction_angles(targets)

    zenith_residuals = predicted_zenith - target_zenith
    azimuth_residuals = wrapped_angle_difference(
        predicted_azimuth,
        target_azimuth,
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.0, 10.0),
    )

    angle_quantities = (
        (
            axes[0, 0],
            target_zenith,
            predicted_zenith,
            "Zenith angle [degree]",
            (0.0, 180.0),
        ),
        (
            axes[0, 1],
            target_azimuth,
            predicted_azimuth,
            "Azimuth angle [degree]",
            (-180.0, 180.0),
        ),
    )

    for axis, target, prediction, label, limits in angle_quantities:
        axis.scatter(
            target,
            prediction,
            color=SCATTER_COLOR,
            alpha=SCATTER_ALPHA,
            s=SCATTER_SIZE,
            rasterized=True,
        )
        axis.plot(
            limits,
            limits,
            color=REFERENCE_COLOR,
            linestyle="--",
            linewidth=1.5,
        )
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_xlabel(f"True {label}")
        axis.set_ylabel(f"Predicted {label}")
        axis.grid(alpha=0.3)

    for axis, residuals, label in (
        (
            axes[1, 0],
            zenith_residuals,
            "Zenith residual [degree]",
        ),
        (
            axes[1, 1],
            azimuth_residuals,
            "Azimuth residual [degree]",
        ),
    ):
        bias = float(np.mean(residuals))
        resolution = quantile(
            np.abs(residuals),
            QUANTILE_LEVEL,
        )

        plot_histogram(
            residuals,
            ax=axis,
            bins=N_HISTOGRAM_BINS,
            xlabel=label,
            ylabel="Events",
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
            bias,
            color=BIAS_COLOR,
            linewidth=1.5,
            label=f"Bias = {bias:.2f} degree",
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
            label=(
                f"{100.0 * QUANTILE_LEVEL:.0f}% resolution = {resolution:.2f} degree"
            ),
        )
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)

    figure.suptitle(
        "Direction regression in angular coordinates",
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

    predictions = as_vector_array(
        predictions,
        size=3,
        name="predictions",
    ).astype(float)

    targets = as_vector_array(
        targets,
        size=3,
        name="targets",
    ).astype(float)

    finite = np.all(np.isfinite(predictions), axis=1)
    finite &= np.all(np.isfinite(targets), axis=1)

    removed_event_count = int(np.count_nonzero(~finite))

    predictions = predictions[finite]
    targets = targets[finite]

    if len(predictions) == 0:
        raise ValueError("No finite direction events remain.")

    if APPLY_INVERSE_COMPONENT_NORMALIZATION:
        transform = MinMaxTransform(
            minimum=DIRECTION_MINIMUM,
            maximum=DIRECTION_MAXIMUM,
        )
        predictions = transform.inverse(predictions)
        targets = transform.inverse(targets)

    print(f"Removed non-finite events: {removed_event_count:,}")
    print_statistics(predictions, targets)

    result_figure, _ = plot_direction_result(
        predictions,
        targets,
    )
    profile_figure, _ = plot_binned_profiles(
        predictions,
        targets,
    )
    angle_figure, _ = plot_angle_results(
        predictions,
        targets,
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
        (
            angle_figure,
            OUTPUT_DIRECTORY / ANGLE_FILENAME,
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
