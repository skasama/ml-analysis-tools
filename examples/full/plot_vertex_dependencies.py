"""Example: plot vertex resolution and bias versus true Wall."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from analysis_tools.analysis import VertexQuantities
from analysis_tools.data import make_binned_selection
from analysis_tools.metrics import standard_error
from analysis_tools.plotting import (
    annotate_values,
    plot_dependency,
    plot_event_counts,
)


@dataclass(frozen=True)
class VertexSample:
    """Event-level vertex information for one sample."""

    name: str
    quantities: VertexQuantities
    wall: np.ndarray
    color: str


@dataclass(frozen=True)
class DependencyResult:
    """One metric and event counts versus a dependency."""

    centers: np.ndarray
    values: np.ndarray
    errors: np.ndarray
    counts: np.ndarray


# ============================================================
# Settings
# ============================================================

WALL_EDGES_CM = np.arange(
    0.0,
    1650.0,
    50.0,
)

MINIMUM_BIN_EVENTS = 10

ANNOTATE_VALUES = False
ANNOTATE_COUNTS = False


# ============================================================
# Calculations
# ============================================================


def calculate_resolution_dependency(
    sample: VertexSample,
    event_values: np.ndarray,
) -> DependencyResult:
    """Calculate the mean resolution and central 68% interval.

    Parameters
    ----------
    sample
        Vertex sample containing true Wall values.
    event_values
        Non-negative event-level resolution quantity.

        Examples:

            abs(x residual)
            abs(y residual)
            abs(z residual)
            abs(R residual)
            3D vertex error
    """
    event_values = np.asarray(
        event_values,
        dtype=float,
    )

    if event_values.ndim != 1:
        raise ValueError("event_values must be one-dimensional.")

    if len(event_values) != len(sample.wall):
        raise ValueError("event_values and Wall must have the same length.")

    bins = make_binned_selection(
        sample.wall,
        WALL_EDGES_CM,
    )

    values = np.full(
        bins.number_of_bins,
        np.nan,
    )

    errors = np.full(
        (2, bins.number_of_bins),
        np.nan,
    )

    for bin_index in range(bins.number_of_bins):
        if bins.counts[bin_index] < MINIMUM_BIN_EVENTS:
            continue

        selected_values = event_values[bins.mask(bin_index)]

        mean_resolution = float(np.mean(selected_values))

        lower, upper = np.quantile(
            selected_values,
            [0.16, 0.84],
        )

        values[bin_index] = mean_resolution

        errors[0, bin_index] = max(
            mean_resolution - lower,
            0.0,
        )

        errors[1, bin_index] = max(
            upper - mean_resolution,
            0.0,
        )

    return DependencyResult(
        centers=bins.centers,
        values=values,
        errors=errors,
        counts=bins.counts,
    )


def calculate_bias_dependency(
    sample: VertexSample,
    residuals: np.ndarray,
) -> DependencyResult:
    """Calculate mean signed residual and its standard error."""
    residuals = np.asarray(
        residuals,
        dtype=float,
    )

    if residuals.ndim != 1:
        raise ValueError("residuals must be one-dimensional.")

    if len(residuals) != len(sample.wall):
        raise ValueError("residuals and Wall must have the same length.")

    bins = make_binned_selection(
        sample.wall,
        WALL_EDGES_CM,
    )

    values = np.full(
        bins.number_of_bins,
        np.nan,
    )

    errors = np.full(
        bins.number_of_bins,
        np.nan,
    )

    for bin_index in range(bins.number_of_bins):
        if bins.counts[bin_index] < MINIMUM_BIN_EVENTS:
            continue

        selected_values = residuals[bins.mask(bin_index)]

        values[bin_index] = float(np.mean(selected_values))

        errors[bin_index] = float(standard_error(selected_values))

    return DependencyResult(
        centers=bins.centers,
        values=values,
        errors=errors,
        counts=bins.counts,
    )


# ============================================================
# Plotting
# ============================================================


def plot_dependency_with_counts(
    results: list[
        tuple[
            VertexSample,
            DependencyResult,
        ]
    ],
    *,
    ylabel: str,
    show_zero_line: bool = False,
) -> tuple[
    plt.Figure,
    np.ndarray,
]:
    """Plot a dependency and event counts on separate axes."""
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10.0, 7.0),
        sharex=True,
        gridspec_kw={
            "height_ratios": (
                4.0,
                1.0,
            ),
            "hspace": 0.08,
        },
    )

    bin_width = float(np.min(np.diff(WALL_EDGES_CM)))

    bar_width = 0.8 * bin_width / len(results)

    for sample_index, (
        sample,
        result,
    ) in enumerate(results):
        plot_dependency(
            result.centers,
            result.values,
            errors=result.errors,
            ax=axes[0],
            label=sample.name,
            color=sample.color,
            marker="o",
            markersize=4.0,
            linewidth=1.5,
            capsize=2.0,
        )

        offset = (sample_index - 0.5 * (len(results) - 1)) * bar_width

        count_centers = result.centers + offset

        plot_event_counts(
            count_centers,
            result.counts,
            ax=axes[1],
            label=sample.name,
            width=bar_width,
            color=sample.color,
            alpha=0.6,
        )

        if ANNOTATE_VALUES:
            annotate_values(
                result.centers,
                result.values,
                ax=axes[0],
                fmt=".3g",
                fontsize=7,
            )

        if ANNOTATE_COUNTS:
            annotate_values(
                count_centers,
                result.counts,
                ax=axes[1],
                fmt=".0f",
                fontsize=6,
            )

    if show_zero_line:
        axes[0].axhline(
            0.0,
            color="black",
            linestyle="--",
            linewidth=1.2,
        )

    # Give both axes labels so either panel can be cropped
    # and used independently.
    axes[0].set_xlabel("True Wall [cm]")
    axes[0].set_ylabel(ylabel)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("True Wall [cm]")
    axes[1].set_ylabel("Events")
    axes[1].grid(
        axis="y",
        alpha=0.25,
    )
    axes[1].legend(
        fontsize=8,
    )

    figure.tight_layout()

    return figure, axes
