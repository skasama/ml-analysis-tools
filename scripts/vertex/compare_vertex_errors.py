"""Compare vertex-error distributions from two reconstruction results."""

from pathlib import Path

import numpy as np

from analysis_tools.data import as_vector_array
from analysis_tools.io import load_array_pair
from analysis_tools.metrics import (
    bias_magnitude,
    bias_vector,
    median,
    quantile,
    vertex_errors,
    vertex_residuals,
)
from analysis_tools.plotting import plot_vertex_errors, save_figure
from analysis_tools.transforms import MinMaxTransform

# Input files
FIRST_PREDICTIONS_PATH = Path("path/to/first/preds.npy")
FIRST_TARGETS_PATH = Path("path/to/first/targets.npy")

SECOND_PREDICTIONS_PATH = Path("path/to/second/preds.npy")
SECOND_TARGETS_PATH = Path("path/to/second/targets.npy")

# Labels and colors
FIRST_LABEL = "Result 1"
SECOND_LABEL = "Result 2"

FIRST_COLOR = "tab:blue"
SECOND_COLOR = "tab:orange"

# Output
OUTPUT_PATH = Path("outputs/vertex_error_comparison.png")

# Vertex normalization range [cm]
VERTEX_MINIMUM = [-1690.0, -1690.0, -1810.0]
VERTEX_MAXIMUM = [1690.0, 1690.0, 1810.0]

# Plot settings
BINS = np.linspace(0.0, 500.0, 101)
X_LIMITS = (0.0, 500.0)
QUANTILE = 0.69


def load_vertex_data(
    predictions_path: Path,
    targets_path: Path,
    transform: MinMaxTransform,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and inverse-transform vertex predictions and targets."""
    predictions, targets = load_array_pair(
        predictions_path,
        targets_path,
        first_name="predictions",
        second_name="targets",
    )

    predictions = as_vector_array(
        predictions,
        size=3,
        name="predictions",
    )
    targets = as_vector_array(
        targets,
        size=3,
        name="targets",
    )

    return (
        transform.inverse(predictions),
        transform.inverse(targets),
    )


def print_summary(
    label: str,
    residuals: np.ndarray,
    errors: np.ndarray,
) -> None:
    """Print vertex reconstruction statistics."""
    vector = bias_vector(residuals)
    magnitude = bias_magnitude(residuals)
    median_error = median(errors)
    quantile_error = quantile(errors, QUANTILE)

    print(label)
    print(f"  Events: {len(errors):,}")
    print(f"  Bias x: {vector[0]:+.2f} cm")
    print(f"  Bias y: {vector[1]:+.2f} cm")
    print(f"  Bias z: {vector[2]:+.2f} cm")
    print(f"  Bias magnitude: {magnitude:.2f} cm")
    print(f"  Median error: {median_error:.2f} cm")
    print(f"  {100 * QUANTILE:.0f}% quantile: {quantile_error:.2f} cm")


def main() -> None:
    transform = MinMaxTransform(
        minimum=VERTEX_MINIMUM,
        maximum=VERTEX_MAXIMUM,
    )

    first_predictions, first_targets = load_vertex_data(
        FIRST_PREDICTIONS_PATH,
        FIRST_TARGETS_PATH,
        transform,
    )
    second_predictions, second_targets = load_vertex_data(
        SECOND_PREDICTIONS_PATH,
        SECOND_TARGETS_PATH,
        transform,
    )

    first_residuals = vertex_residuals(
        first_predictions,
        first_targets,
    )
    second_residuals = vertex_residuals(
        second_predictions,
        second_targets,
    )

    first_errors = vertex_errors(
        first_predictions,
        first_targets,
    )
    second_errors = vertex_errors(
        second_predictions,
        second_targets,
    )

    first_median = median(first_errors)
    second_median = median(second_errors)

    first_quantile = quantile(first_errors, QUANTILE)
    second_quantile = quantile(second_errors, QUANTILE)

    figure, axis = plot_vertex_errors(
        first_errors,
        bins=BINS,
        xlabel="3D vertex error [cm]",
        xlim=X_LIMITS,
        label=FIRST_LABEL,
        color=FIRST_COLOR,
        linewidth=2.0,
    )

    figure, axis = plot_vertex_errors(
        second_errors,
        ax=axis,
        bins=BINS,
        label=SECOND_LABEL,
        color=SECOND_COLOR,
        linewidth=2.0,
    )

    axis.axvline(
        first_median,
        color=FIRST_COLOR,
        linestyle=":",
        linewidth=1.5,
        label=f"{FIRST_LABEL} median: {first_median:.1f} cm",
    )
    axis.axvline(
        first_quantile,
        color=FIRST_COLOR,
        linestyle="--",
        linewidth=1.5,
        label=(f"{FIRST_LABEL} {100 * QUANTILE:.0f}%: {first_quantile:.1f} cm"),
    )

    axis.axvline(
        second_median,
        color=SECOND_COLOR,
        linestyle=":",
        linewidth=1.5,
        label=f"{SECOND_LABEL} median: {second_median:.1f} cm",
    )
    axis.axvline(
        second_quantile,
        color=SECOND_COLOR,
        linestyle="--",
        linewidth=1.5,
        label=(f"{SECOND_LABEL} {100 * QUANTILE:.0f}%: {second_quantile:.1f} cm"),
    )

    axis.legend()

    save_figure(
        figure,
        OUTPUT_PATH,
        close=True,
    )

    print_summary(
        FIRST_LABEL,
        first_residuals,
        first_errors,
    )
    print_summary(
        SECOND_LABEL,
        second_residuals,
        second_errors,
    )

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
