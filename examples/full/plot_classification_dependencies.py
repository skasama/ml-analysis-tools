"""Plot ML classification efficiency and purity dependencies.

Edit the settings below and run:

    uv run python examples/full/plot_classification_dependencies.py
"""

from dataclasses import dataclass
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
    binned_classification_counts,
    binomial_errors,
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


@dataclass(frozen=True)
class ClassificationDependencyData:
    """Classification results and truth quantities for one method."""

    name: str

    predictions: np.ndarray
    targets: np.ndarray

    wall: np.ndarray
    towall: np.ndarray
    energy: np.ndarray

    color: str
    marker: str
    linestyle: str

    @property
    def event_count(self) -> int:
        """Return the number of events."""
        return len(self.targets)

    def __post_init__(self) -> None:
        """Validate that all event arrays have the same length."""
        event_count = len(self.targets)

        arrays = {
            "predictions": self.predictions,
            "wall": self.wall,
            "towall": self.towall,
            "energy": self.energy,
        }

        for array_name, array in arrays.items():
            if len(array) != event_count:
                raise ValueError(
                    f"{array_name} has {len(array)} events, "
                    f"but targets has {event_count}."
                )


# ============================================================
# Input files
# ============================================================

SCORES_PATH = Path("path/to/outputs/preds.npy")
TARGETS_PATH = Path("path/to/outputs/targets.npy")

# Contains positions, directions, energies, and labels.
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
CLASS_LABELS = np.array(
    [0, 1, 2],
    dtype=np.int64,
)

CLASS_NAMES = {
    0: "Electron",
    1: "Muon",
    2: "Gamma",
}

# True when preds.npy contains logits.
# False when preds.npy already contains probabilities.
APPLY_SOFTMAX = True


# ============================================================
# Detector geometry
# ============================================================

DETECTOR_RADIUS_CM = 1690.0
DETECTOR_HALF_HEIGHT_CM = 1810.0

# z-axis
DETECTOR_AXIS = 2


# ============================================================
# Statistical uncertainty
# ============================================================

BINOMIAL_ERROR_METHOD = "wilson"
CONFIDENCE_LEVEL = 0.6827

ERRORBAR_CAPSIZE = 2.5
ERRORBAR_LINEWIDTH = 1.0


# ============================================================
# Binning
# ============================================================

# A bin is displayed only when its total event count is at least
# this value.
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

FIGURE_WIDTH_PER_COLUMN = 5.5
FIGURE_HEIGHT_PER_ROW = 4.5

ML_COLOR = "tab:blue"
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

    scores = np.asarray(
        scores,
        dtype=float,
    )

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

    if not np.array_equal(
        targets,
        truth_labels,
    ):
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
    """Convert classification scores to probabilities."""
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
    method: ClassificationDependencyData,
    *,
    dependency_name: str,
    edges: np.ndarray,
    class_label: int,
    metric: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Calculate one classification metric and its binomial errors."""
    if metric not in {
        "efficiency",
        "purity",
    }:
        raise ValueError("metric must be 'efficiency' or 'purity'.")

    dependency = getattr(
        method,
        dependency_name,
    )

    (
        centers,
        true_positive_counts,
        true_class_counts,
        predicted_class_counts,
        event_counts,
    ) = binned_classification_counts(
        dependency,
        method.predictions,
        method.targets,
        edges,
        class_label=class_label,
    )

    # Efficiency:
    #   true positives / true events of the selected class
    #
    # Purity:
    #   true positives / predicted events of the selected class
    denominator = (
        true_class_counts if metric == "efficiency" else predicted_class_counts
    )

    values = np.divide(
        true_positive_counts,
        denominator,
        out=np.full(
            len(centers),
            np.nan,
            dtype=float,
        ),
        where=denominator > 0,
    )

    lower_error, upper_error = binomial_errors(
        true_positive_counts,
        denominator,
        method=BINOMIAL_ERROR_METHOD,
        confidence_level=CONFIDENCE_LEVEL,
    )

    lower_error = np.asarray(
        lower_error,
        dtype=float,
    )
    upper_error = np.asarray(
        upper_error,
        dtype=float,
    )

    valid = (
        (event_counts >= MINIMUM_EVENTS)
        & (denominator > 0)
        & np.isfinite(values)
        & np.isfinite(lower_error)
        & np.isfinite(upper_error)
    )

    values[~valid] = np.nan
    lower_error[~valid] = np.nan
    upper_error[~valid] = np.nan

    return (
        centers,
        values,
        lower_error,
        upper_error,
        event_counts,
    )


def plot_metric_dependencies(
    methods: list[ClassificationDependencyData],
    dependencies: tuple[
        tuple[str, np.ndarray, str, str],
        ...,
    ],
    class_labels: np.ndarray,
    *,
    metric: str,
):
    """Plot efficiency or purity dependencies with binomial errors."""
    if metric not in {
        "efficiency",
        "purity",
    }:
        raise ValueError("metric must be 'efficiency' or 'purity'.")

    figure, axes = plt.subplots(
        len(class_labels),
        len(dependencies),
        figsize=(
            FIGURE_WIDTH_PER_COLUMN * len(dependencies),
            FIGURE_HEIGHT_PER_ROW * len(class_labels),
        ),
        squeeze=False,
    )

    csv_rows: list[list[object]] = []

    for row, class_label in enumerate(class_labels):
        label = int(class_label)

        class_name = CLASS_NAMES.get(
            label,
            str(label),
        )

        for column, (
            dependency_name,
            edges,
            xlabel,
            dependency_title,
        ) in enumerate(dependencies):
            axis = axes[row, column]

            for method in methods:
                (
                    centers,
                    values,
                    lower_error,
                    upper_error,
                    event_counts,
                ) = calculate_dependency_metrics(
                    method,
                    dependency_name=dependency_name,
                    edges=edges,
                    class_label=label,
                    metric=metric,
                )

                valid = (
                    np.isfinite(values)
                    & np.isfinite(lower_error)
                    & np.isfinite(upper_error)
                )

                yerr = np.vstack(
                    (
                        lower_error[valid],
                        upper_error[valid],
                    )
                )

                plot_series(
                    centers[valid],
                    values[valid],
                    ax=axis,
                    label=method.name,
                    color=method.color,
                    marker=method.marker,
                    linestyle=method.linestyle,
                    linewidth=LINEWIDTH,
                    yerr=yerr,
                    capsize=ERRORBAR_CAPSIZE,
                    elinewidth=ERRORBAR_LINEWIDTH,
                )

                for (
                    center,
                    value,
                    error_low,
                    error_high,
                    event_count,
                ) in zip(
                    centers,
                    values,
                    lower_error,
                    upper_error,
                    event_counts,
                    strict=True,
                ):
                    csv_rows.append(
                        [
                            method.name,
                            metric,
                            label,
                            class_name,
                            dependency_name,
                            center,
                            value,
                            error_low,
                            error_high,
                            event_count,
                        ]
                    )

            axis.set_xlabel(xlabel)
            axis.set_ylabel(metric.capitalize())
            axis.set_ylim(0.0, 1.05)
            axis.set_title(f"{class_name} {metric} vs {dependency_title}")
            axis.grid(alpha=0.3)
            axis.legend()

    comparison_enabled = len(methods) > 1

    title_suffix = "ML vs fiTQun" if comparison_enabled else "ML"

    if BINOMIAL_ERROR_METHOD == "wilson":
        error_description = f"Wilson interval, {100.0 * CONFIDENCE_LEVEL:.2f}% CL"
    else:
        error_description = "Standard binomial error"

    figure.suptitle(
        (f"Classification {metric} dependencies: {title_suffix}\n{error_description}"),
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
            "method",
            "metric",
            "class_label",
            "class_name",
            "dependency",
            "bin_center",
            "value",
            "error_low",
            "error_high",
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
        axis=DETECTOR_AXIS,
    )

    towall = compute_towall(
        positions,
        directions,
        radius=DETECTOR_RADIUS_CM,
        half_height=DETECTOR_HALF_HEIGHT_CM,
        axis=DETECTOR_AXIS,
    )

    ml_data = ClassificationDependencyData(
        name="ML",
        predictions=predictions,
        targets=targets,
        wall=wall,
        towall=towall,
        energy=energies,
        color=ML_COLOR,
        marker=MARKER,
        linestyle=LINESTYLE,
    )

    methods = [ml_data]

    # Each entry contains:
    #   attribute name, bin edges, x-axis label, title label
    dependencies = (
        (
            "wall",
            WALL_EDGES,
            "True Wall [cm]",
            "Wall",
        ),
        (
            "towall",
            TOWALL_EDGES,
            "True ToWall [cm]",
            "ToWall",
        ),
        (
            "energy",
            ENERGY_EDGES,
            "True energy [MeV]",
            "energy",
        ),
    )

    (
        efficiency_figure,
        _,
        efficiency_rows,
    ) = plot_metric_dependencies(
        methods,
        dependencies,
        CLASS_LABELS,
        metric="efficiency",
    )

    (
        purity_figure,
        _,
        purity_rows,
    ) = plot_metric_dependencies(
        methods,
        dependencies,
        CLASS_LABELS,
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

    print(f"Events: {ml_data.event_count:,}")
    print(f"Saved: {efficiency_path}")
    print(f"Saved: {purity_path}")
    print(f"Saved: {csv_path}")

    if SHOW_FIGURES:
        plt.show()


if __name__ == "__main__":
    main()
