"""Evaluate and plot synthetic vertex results in a few lines."""

from pathlib import Path

import numpy as np

from analysis_tools import (
    calculate_vertex_quantities,
    evaluate_dependency,
    plot_evaluation,
)
from analysis_tools.plotting import save_figure

RNG = np.random.default_rng(42)

TARGETS = np.column_stack(
    (
        RNG.normal(0.0, 300.0, 300),
        RNG.normal(0.0, 300.0, 300),
        RNG.normal(0.0, 800.0, 300),
    )
)
PREDICTIONS = TARGETS + RNG.normal(0.0, 40.0, TARGETS.shape)
WALL = np.abs(TARGETS[:, 2])
EDGES = np.arange(0.0, 850.0, 100.0)
OUTPUT_PATH = Path("outputs/vertex_resolution_dependency.png")


def central68_error(values: np.ndarray) -> tuple[float, float]:
    """Return asymmetric lower/upper deviations from the mean."""
    mean = float(np.mean(values))
    lower, upper = np.quantile(values, [0.16, 0.84])

    return (
        max(mean - float(lower), 0.0),
        max(float(upper) - mean, 0.0),
    )


def main() -> None:
    quantities = calculate_vertex_quantities(
        PREDICTIONS,
        TARGETS,
    )

    evaluation = evaluate_dependency(
        WALL,
        quantities.errors_3d,
        EDGES,
        metric=np.mean,
        error=central68_error,
        minimum_events=10,
    )

    metric_ax, count_ax = plot_evaluation(
        evaluation,
        label="ML",
        color="tab:blue",
        show_counts=True,
    )

    metric_ax.set(
        xlabel="Absolute z [cm]",
        ylabel="Mean 3D vertex error [cm]",
        title="Vertex resolution",
    )
    count_ax.set_xlabel("Absolute z [cm]")

    save_figure(
        metric_ax.figure,
        OUTPUT_PATH,
        close=True,
    )


if __name__ == "__main__":
    main()
