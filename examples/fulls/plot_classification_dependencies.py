"""Plot ML classification efficiency and purity dependencies.

Edit the settings below and run:

    uv run python examples/plot_classification_dependencies.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis_tools.data import (
    as_integer_array,
    as_scalar_array,
    as_vector_array,
    validate_same_length,
)
from analysis_tools.io import (
    load_array_pair,
    load_npz,
    save_csv,
)
from analysis_tools.metrics import (
    binned_classification_metrics,
    make_bin_edges,
    predicted_labels,
    softmax,
)
from analysis_tools.plotting import save_figure
from analysis_tools.plotting.common import plot_series
from analysis_tools.transforms.geometry import (
    compute_towall,
    compute_wall,
)

# ============================================================
# Input files
# ============================================================

SCORES_PATH = Path("path/to/outputs/preds.npy")
TARGETS_PATH = Path("path/to/outputs/targets.npy")

# Contains positions, directions, energies, and optionally labels.
TRUTH_PATH = Path("path/to/classification_truth.npz")


# ============================================================
# Keys in the truth NPZ
# ============================================================

POSITIONS_KEY = "positions"
DIRECTIONS_KEY = "directions"
ENERGIES_KEY = "energies"
LABELS_KEY = "labels"


# ============================================================
# Output
# ============================================================

OUTPUT_DIRECTORY = Path("outputs/classification_dependencies")

EFFICIENCY_FILENAME = "classification_efficiency_dependencies.png"
PURITY_FILENAME = "classification_purity_dependencies.png"
CSV_FILENAME = "classification_dependencies.csv"


# ============================================================
# Classification settings
# ============================================================

# The order must match the score columns.
CLASS_LABELS = np.array([0, 1, 2], dtype=np.int64)

CLASS_NAMES = (
    "Electron",
    "Muon",
    "Gamma",
)

CLASS_COLORS = (
    "tab:blue",
    "tab:orange",
    "tab:green",
)

# True when preds.npy contains logits.
APPLY_SOFTMAX = True


# ============================================================
# Detector geometry
# ============================================================

DETECTOR_RADIUS_CM = 1690.0
DETECTOR_HALF_HEIGHT_CM = 1810.0


# ============================================================
# Binning
# ============================================================

MINIMUM_EVENTS = 20

WALL_EDGES = make_bin_edges(
    minimum=0.0,
    maximum=1000.0,
    width=50.0,
)

TOWALL_EDGES = make_bin_edges(
    minimum=0.0,
    maximum=4000.0,
    width=100.0,
)

ENERGY_EDGES = make_bin_edges(
    minimum=0.0,
    maximum=2000.0,
    width=100.0,
)


# ============================================================
# Plot settings
# ============================================================

DPI = 200
SHOW_FIGURES = True
FIGSIZE = (17.0, 14.0)

MARKER = "o"
LINESTYLE = "-"
LINEWIDTH = 1.8


def load_data() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Load classification outputs and truth quantities."""
    scores, targets = load_array_pair(
        SCORES_PATH,
        TARGETS_PATH,
        first_name="scores",
        second_name="targets",
    )

    truth = load_npz(
        TRUTH_PATH,
        keys=[
            POSITIONS_KEY,
            DIRECTIONS_KEY,
            ENERGIES_KEY,
            LABELS_KEY,
        ],
    )

    scores = np.asarray(scores, dtype=float)
    targets = as_integer_array(
        targets,
        name="targets",
    )

    positions = as_vector_array(
        truth[POSITIONS_KEY],
        size=3,
        name="positions",
    ).astype(float)

    directions = as_vector_array(
        truth[DIRECTIONS_KEY],
        size=3,
        name="directions",
    ).astype(float)

    energies = as_scalar_array(
        truth[ENERGIES_KEY],
        name="energies",
    ).astype(float)

    truth_labels = as_integer_array(
        truth[LABELS_KEY],
        name="truth_labels",
    )

    validate_same_length(
        scores,
        targets,
        positions,
        directions,
        energies,
        truth_labels,
        names=(
            "scores",
            "targets",
            "positions",
            "directions",
            "energies",
            "truth_labels",
        ),
    )

    if scores.ndim != 2:
        raise ValueError(f"scores must have shape (N, C), got {scores.shape}.")

    if scores.shape[1] != len(CLASS_LABELS):
        raise ValueError(
            "The number of score columns does not match "
            f"CLASS_LABELS: {scores.shape[1]} != "
            f"{len(CLASS_LABELS)}."
        )

    if not np.array_equal(targets, truth_labels):
        mismatch = np.flatnonzero(targets != truth_labels)
        first = int(mismatch[0])

        raise ValueError(
            "Classification targets and truth labels are not aligned. "
            f"First mismatch at index {first}: "
            f"target={targets[first]}, "
            f"truth={truth_labels[first]}."
        )

    finite = np.all(
        np.isfinite(scores),
        axis=1,
    )
    finite &= np.all(
        np.isfinite(positions),
        axis=1,
    )
    finite &= np.all(
        np.isfinite(directions),
        axis=1,
    )
    finite &= np.isfinite(energies)

    removed_count = int(np.count_nonzero(~finite))

    scores = scores[finite]
    targets = targets[finite]
    positions = positions[finite]
    directions = directions[finite]
    energies = energies[finite]

    if len(scores) == 0:
        raise ValueError("No finite classification events remain.")

    unknown_labels = np.setdiff1d(
        np.unique(targets),
        CLASS_LABELS,
    )

    if unknown_labels.size:
        raise ValueError(
            "Targets contain labels not listed in CLASS_LABELS: "
            f"{unknown_labels.tolist()}"
        )

    print(f"Removed non-finite events: {removed_count:,}")

    return (
        scores,
        targets,
        positions,
        directions,
        energies,
    )


def create_probabilities(
    scores: np.ndarray,
) -> np.ndarray:
    """Convert scores to probabilities."""
    if APPLY_SOFTMAX:
        return softmax(scores)

    if np.any(scores < 0.0):
        raise ValueError("Probabilities must be non-negative.")

    if not np.allclose(
        np.sum(scores, axis=1),
        1.0,
    ):
        raise ValueError("Probability rows must sum to one.")

    return scores


def calculate_predictions(
    probabilities: np.ndarray,
) -> np.ndarray:
    """Convert probability-column indices to class labels."""
    prediction_indices = predicted_labels(probabilities)

    return CLASS_LABELS[prediction_indices]


def calculate_dependency_metrics(
    dependency: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    edges: np.ndarray,
    *,
    class_label: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Calculate classification metrics in dependency bins."""
    return binned_classification_metrics(
        dependency,
        predictions,
        targets,
        edges,
        class_label=class_label,
        minimum_events=MINIMUM_EVENTS,
    )


def plot_metric_dependencies(
    dependencies: tuple[
        tuple[str, np.ndarray, np.ndarray, str],
        ...,
    ],
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    metric: str,
):
    """Plot efficiency or purity for every class and dependency."""
    if metric not in {"efficiency", "purity"}:
        raise ValueError("metric must be 'efficiency' or 'purity'.")

    figure, axes = plt.subplots(
        len(CLASS_LABELS),
        len(dependencies),
        figsize=FIGSIZE,
        squeeze=False,
    )

    csv_rows: list[list[object]] = []

    for row, (
        class_label,
        class_name,
        color,
    ) in enumerate(
        zip(
            CLASS_LABELS,
            CLASS_NAMES,
            CLASS_COLORS,
            strict=True,
        )
    ):
        for column, (
            dependency_name,
            dependency_values,
            edges,
            xlabel,
        ) in enumerate(dependencies):
            axis = axes[row, column]

            (
                centers,
                efficiency,
                purity,
                counts,
            ) = calculate_dependency_metrics(
                dependency_values,
                predictions,
                targets,
                edges,
                class_label=int(class_label),
            )

            values = efficiency if metric == "efficiency" else purity

            valid = np.isfinite(values)

            plot_series(
                centers[valid],
                values[valid],
                ax=axis,
                xlabel=xlabel,
                ylabel=metric.capitalize(),
                title=(f"{class_name} {metric} vs {dependency_name}"),
                color=color,
                marker=MARKER,
                linestyle=LINESTYLE,
                linewidth=LINEWIDTH,
            )

            axis.set_ylim(0.0, 1.05)
            axis.grid(alpha=0.3)

            for center, value, count in zip(
                centers,
                values,
                counts,
                strict=True,
            ):
                csv_rows.append(
                    [
                        metric,
                        int(class_label),
                        class_name,
                        dependency_name,
                        center,
                        value,
                        count,
                    ]
                )

    figure.suptitle(
        (f"Classification {metric} dependencies"),
        fontsize=16,
    )
    figure.tight_layout()

    return figure, axes, csv_rows


def save_results_csv(
    path: Path,
    rows: list[list[object]],
) -> None:
    """Save classification-dependency results."""
    save_csv(
        path,
        np.asarray(
            rows,
            dtype=object,
        ),
        column_names=(
            "metric",
            "class_label",
            "class_name",
            "dependency",
            "bin_center",
            "value",
            "event_count",
        ),
        fmt="%s",
    )


def main() -> None:
    (
        scores,
        targets,
        positions,
        directions,
        energies,
    ) = load_data()

    probabilities = create_probabilities(scores)
    predictions = calculate_predictions(probabilities)

    wall = compute_wall(
        positions,
        radius=DETECTOR_RADIUS_CM,
        half_height=DETECTOR_HALF_HEIGHT_CM,
    )

    towall = compute_towall(
        positions,
        directions,
        radius=DETECTOR_RADIUS_CM,
        half_height=DETECTOR_HALF_HEIGHT_CM,
    )

    dependencies = (
        (
            "Wall",
            wall,
            WALL_EDGES,
            "True Wall [cm]",
        ),
        (
            "ToWall",
            towall,
            TOWALL_EDGES,
            "True ToWall [cm]",
        ),
        (
            "energy",
            energies,
            ENERGY_EDGES,
            "True energy [MeV]",
        ),
    )

    efficiency_figure, _, efficiency_rows = plot_metric_dependencies(
        dependencies,
        predictions,
        targets,
        metric="efficiency",
    )

    purity_figure, _, purity_rows = plot_metric_dependencies(
        dependencies,
        predictions,
        targets,
        metric="purity",
    )

    efficiency_path = OUTPUT_DIRECTORY / EFFICIENCY_FILENAME
    purity_path = OUTPUT_DIRECTORY / PURITY_FILENAME
    csv_path = OUTPUT_DIRECTORY / CSV_FILENAME

    save_figure(
        efficiency_figure,
        efficiency_path,
        dpi=DPI,
        close=not SHOW_FIGURES,
    )
    save_figure(
        purity_figure,
        purity_path,
        dpi=DPI,
        close=not SHOW_FIGURES,
    )

    save_results_csv(
        csv_path,
        efficiency_rows + purity_rows,
    )

    print(f"Events: {len(targets):,}")
    print(f"Saved: {efficiency_path}")
    print(f"Saved: {purity_path}")
    print(f"Saved: {csv_path}")

    if SHOW_FIGURES:
        plt.show()


if __name__ == "__main__":
    main()
