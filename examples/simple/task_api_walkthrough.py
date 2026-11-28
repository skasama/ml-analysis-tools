"""Run an input-free walkthrough of selections and task evaluation APIs.

Run from the repository root:

    uv run python examples/simple/task_api_walkthrough.py
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from analysis_tools import (
    EventSelection,
    calculate_classification_quantities,
    calculate_direction_quantities,
    calculate_energy_quantities,
    calculate_vertex_quantities,
    evaluate_dependency,
    plot_evaluation,
)
from analysis_tools.metrics import confusion_matrix, standard_error
from analysis_tools.plotting import save_figure
from analysis_tools.quantities import (
    ClassificationQuantities,
    DirectionQuantities,
    EnergyQuantities,
    VertexQuantities,
)
from analysis_tools.transforms.geometry import compute_wall, normalize_vectors

RANDOM_SEED = 42
EVENT_COUNT = 3000
CLASS_LABELS = np.array([11, 13, 22], dtype=np.int64)
MINIMUM_NHIT = 100
MINIMUM_WALL_CM = 100.0
DETECTOR_RADIUS_CM = 1690.0
DETECTOR_HALF_HEIGHT_CM = 1810.0
WALL_EDGES_CM = np.arange(0.0, 1701.0, 200.0)
MINIMUM_EVENTS_PER_BIN = 20
OUTPUT_DIRECTORY = Path("outputs/task_api_walkthrough")


@dataclass(frozen=True)
class DemoData:
    """Selected synthetic event quantities and their shared selection."""

    vertex: VertexQuantities
    direction: DirectionQuantities
    energy: EnergyQuantities
    classification: ClassificationQuantities
    wall: NDArray[np.float64]
    selection: EventSelection


def build_demo_data() -> DemoData:
    """Create correlated synthetic task data and apply one shared selection."""
    rng = np.random.default_rng(RANDOM_SEED)

    radial_distance = np.sqrt(rng.uniform(0.0, 1.0, EVENT_COUNT)) * 1780.0
    azimuth = rng.uniform(-np.pi, np.pi, EVENT_COUNT)
    target_positions = np.column_stack(
        (
            radial_distance * np.cos(azimuth),
            radial_distance * np.sin(azimuth),
            rng.uniform(-1900.0, 1900.0, EVENT_COUNT),
        )
    )
    predicted_positions = target_positions + rng.normal(
        0.0,
        35.0,
        size=target_positions.shape,
    )
    wall = compute_wall(
        target_positions,
        radius=DETECTOR_RADIUS_CM,
        half_height=DETECTOR_HALF_HEIGHT_CM,
    )

    target_directions = normalize_vectors(rng.normal(size=(EVENT_COUNT, 3)))
    predicted_directions = normalize_vectors(
        target_directions + rng.normal(0.0, 0.05, size=target_directions.shape)
    )

    target_energies = rng.uniform(50.0, 1500.0, EVENT_COUNT)
    predicted_energies = target_energies + rng.normal(
        0.0,
        0.08 * target_energies,
    )

    target_labels = rng.choice(CLASS_LABELS, EVENT_COUNT, p=[0.35, 0.45, 0.20])
    class_indices = np.searchsorted(CLASS_LABELS, target_labels)
    scores = rng.normal(0.0, 1.0, size=(EVENT_COUNT, len(CLASS_LABELS)))
    scores[np.arange(EVENT_COUNT), class_indices] += 2.0
    nhit = rng.integers(30, 250, EVENT_COUNT)

    vertex = calculate_vertex_quantities(predicted_positions, target_positions)
    direction = calculate_direction_quantities(
        predicted_directions,
        target_directions,
    )
    energy = calculate_energy_quantities(predicted_energies, target_energies)
    classification = calculate_classification_quantities(
        scores,
        target_labels,
        class_labels=CLASS_LABELS,
    )

    selection = EventSelection(EVENT_COUNT)
    quality_flag = rng.random(EVENT_COUNT) >= 0.02
    selection.add("passed reconstruction quality", quality_flag)
    selection.add(
        f"nhit >= {MINIMUM_NHIT}",
        nhit >= MINIMUM_NHIT,
    )
    selection.add(
        f"Wall >= {MINIMUM_WALL_CM:g} cm",
        wall >= MINIMUM_WALL_CM,
    )
    mask = selection.mask

    return DemoData(
        vertex=vertex.select(mask),
        direction=direction.select(mask),
        energy=energy.select(mask),
        classification=classification.select(mask),
        wall=wall[mask],
        selection=selection,
    )


def _central_68_magnitude(values: NDArray[np.float64]) -> float:
    """Return the 68th percentile of an error magnitude."""
    return float(np.quantile(values, 0.68))


def make_demo_plots(data: DemoData) -> dict[str, plt.Figure]:
    """Create task distributions, dependency profiles, and a confusion matrix."""
    distributions, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    axes[0, 0].hist(data.vertex.errors_3d, bins=50, histtype="step")
    axes[0, 0].set(xlabel="3D vertex error [cm]", ylabel="Events")
    axes[0, 1].hist(data.direction.opening_angles, bins=50, histtype="step")
    axes[0, 1].set(xlabel="Opening-angle error [degree]", ylabel="Events")
    axes[1, 0].hist(data.energy.residuals, bins=50, histtype="step")
    axes[1, 0].set(xlabel="Energy residual [MeV]", ylabel="Events")
    axes[1, 1].hist(
        data.energy.relative_residuals[np.isfinite(data.energy.relative_residuals)],
        bins=50,
        histtype="step",
    )
    axes[1, 1].set(xlabel="Fractional energy residual", ylabel="Events")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    distributions.tight_layout()

    dependencies, (resolution_axis, accuracy_axis) = plt.subplots(
        1,
        2,
        figsize=(11.0, 4.5),
    )
    vertex_resolution = evaluate_dependency(
        data.wall,
        data.vertex.errors_3d,
        WALL_EDGES_CM,
        metric=_central_68_magnitude,
        minimum_events=MINIMUM_EVENTS_PER_BIN,
    )
    plot_evaluation(
        vertex_resolution,
        ax=resolution_axis,
        label="Vertex 68% resolution",
        color="tab:blue",
        marker="o",
    )
    resolution_axis.set(
        xlabel="True Wall [cm]",
        ylabel="68th percentile of 3D error [cm]",
    )
    resolution_axis.grid(alpha=0.25)

    accuracy = evaluate_dependency(
        data.wall,
        data.classification.correct_mask.astype(float),
        WALL_EDGES_CM,
        metric=np.mean,
        error=standard_error,
        minimum_events=MINIMUM_EVENTS_PER_BIN,
    )
    plot_evaluation(
        accuracy,
        ax=accuracy_axis,
        label="Classification accuracy",
        color="tab:orange",
        marker="o",
        capsize=2,
    )
    accuracy_axis.set(
        xlabel="True Wall [cm]",
        ylabel="Accuracy",
        ylim=(0.0, 1.0),
    )
    accuracy_axis.grid(alpha=0.25)
    dependencies.tight_layout()

    matrix, labels = confusion_matrix(
        data.classification.predicted_labels,
        data.classification.target_labels,
        class_labels=CLASS_LABELS,
    )
    confusion, axis = plt.subplots(figsize=(5.5, 4.5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(
        xlabel="Predicted class",
        ylabel="True class",
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
    )
    for row in range(len(labels)):
        for column in range(len(labels)):
            axis.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center")
    confusion.colorbar(image, ax=axis, label="Events")
    confusion.tight_layout()

    return {
        "task_distributions": distributions,
        "wall_dependencies": dependencies,
        "classification_confusion_matrix": confusion,
    }


def main() -> None:
    """Run the walkthrough and save all example figures."""
    data = build_demo_data()
    data.selection.print_cut_flow()
    print(f"\nSelected events: {data.vertex.event_count:,}")
    print(
        f"Median opening angle: {np.median(data.direction.opening_angles):.3f} degree"
    )
    print(
        "Median absolute energy residual: "
        f"{np.median(np.abs(data.energy.residuals)):.3f} MeV"
    )
    print(f"Classification accuracy: {np.mean(data.classification.correct_mask):.3f}")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for name, figure in make_demo_plots(data).items():
        path = OUTPUT_DIRECTORY / f"{name}.png"
        save_figure(figure, path, close=True)
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
