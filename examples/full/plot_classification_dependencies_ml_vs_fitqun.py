"""Plot ML classification dependencies, optionally compared with fiTQun.

Set FITQUN_PATH to None to plot ML results only.

Edit the settings below and run:

    uv run python examples/plot_classification_dependencies.py
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
    FitQunKeys,
    load_array_pair,
    load_fitqun,
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
# ML input files
# ============================================================

SCORES_PATH = Path("path/to/outputs/preds.npy")
TARGETS_PATH = Path("path/to/outputs/targets.npy")

# Contains positions, directions, energies, and labels.
TRUTH_PATH = Path("path/to/classification_truth.npz")


# ============================================================
# Keys in the ML truth NPZ
# ============================================================

POSITIONS_KEY = "positions"
DIRECTIONS_KEY = "directions"
ENERGIES_KEY = "energies"
LABELS_KEY = "labels"


# ============================================================
# Optional fiTQun input
# ============================================================

# Set to None to plot ML results only.
FITQUN_PATH: Path | None = Path(
    "path/to/common_multi_fitqun.hy"
)

# FITQUN_PATH: Path | None = None

FITQUN_KEYS = FitQunKeys(
    labels="multi/labels",
    truth_positions="multi/positions",
    truth_directions="multi/directions",
    truth_energies="multi/energies",
    electron_nll="fitqun/e_1rnll",
    muon_nll="fitqun/mu_1rnll",
    electron_momentum="fitqun/e_1rmom",
)

# fiTQun e/m score:
#     muon_nll - electron_nll - 0.2 * electron_momentum
#
# score > threshold -> electron
# score <= threshold -> muon
FITQUN_SCORE_THRESHOLD = 0.0


# ============================================================
# Class definitions
# ============================================================

# The order must match the ML score columns.
CLASS_LABELS = np.array([0, 1, 2], dtype=np.int64)

CLASS_NAMES = {
    0: "Electron",
    1: "Muon",
    2: "Gamma",
}

CLASS_COLORS = {
    0: "tab:blue",
    1: "tab:orange",
    2: "tab:green",
}

ELECTRON_LABEL = 0
MUON_LABEL = 1

# When fiTQun is enabled, compare only these supported classes.
COMPARISON_CLASS_LABELS = np.array(
    [ELECTRON_LABEL, MUON_LABEL],
    dtype=np.int64,
)


# ============================================================
# ML input interpretation
# ============================================================

# True if preds.npy contains logits.
# False if preds.npy already contains probabilities.
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
# Output
# ============================================================

OUTPUT_DIRECTORY = Path(
    "outputs/classification_dependencies"
)

ML_ONLY_EFFICIENCY_FILENAME = (
    "classification_efficiency_dependencies.png"
)
ML_ONLY_PURITY_FILENAME = (
    "classification_purity_dependencies.png"
)
ML_ONLY_CSV_FILENAME = (
    "classification_dependencies.csv"
)

COMPARISON_EFFICIENCY_FILENAME = (
    "classification_efficiency_ml_vs_fitqun.png"
)
COMPARISON_PURITY_FILENAME = (
    "classification_purity_ml_vs_fitqun.png"
)
COMPARISON_CSV_FILENAME = (
    "classification_dependencies_ml_vs_fitqun.csv"
)


# ============================================================
# Plot settings
# ============================================================

DPI = 200
SHOW_FIGURES = True
FIGURE_WIDTH_PER_COLUMN = 5.5
FIGURE_HEIGHT_PER_ROW = 4.5

ML_COLOR = "tab:blue"
ML_MARKER = "o"
ML_LINESTYLE = "-"

FITQUN_COLOR = "black"
FITQUN_MARKER = "s"
FITQUN_LINESTYLE = "--"

LINEWIDTH = 1.8


@dataclass(frozen=True)
class ClassificationDependencyData:
    """Classification results and corresponding truth quantities."""

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
        return len(self.targets)


def remove_nonfinite_events(
    *arrays: np.ndarray,
) -> tuple[tuple[np.ndarray, ...], int]:
    """Remove events with non-finite values in any array."""
    if not arrays:
        raise ValueError("At least one array is required.")

    validate_same_length(*arrays)

    valid_mask = np.ones(
        len(arrays[0]),
        dtype=bool,
    )

    for array in arrays:
        array = np.asarray(array)

        if array.ndim == 1:
            valid_mask &= np.isfinite(array)
        else:
            axes = tuple(
                range(1, array.ndim)
            )
            valid_mask &= np.all(
                np.isfinite(array),
                axis=axes,
            )

    removed_count = int(
        np.count_nonzero(~valid_mask)
    )

    selected = tuple(
        np.asarray(array)[valid_mask]
        for array in arrays
    )

    if len(selected[0]) == 0:
        raise ValueError(
            "No finite events remain after filtering."
        )

    return selected, removed_count


def select_classes(
    data: ClassificationDependencyData,
    class_labels: np.ndarray,
) -> ClassificationDependencyData:
    """Select events belonging to the requested truth classes."""
    selected = np.isin(
        data.targets,
        class_labels,
    )

    if not np.any(selected):
        raise ValueError(
            f"{data.name} has no events for classes "
            f"{class_labels.tolist()}."
        )

    return ClassificationDependencyData(
        name=data.name,
        predictions=data.predictions[selected],
        targets=data.targets[selected],
        wall=data.wall[selected],
        towall=data.towall[selected],
        energy=data.energy[selected],
        color=data.color,
        marker=data.marker,
        linestyle=data.linestyle,
    )


def create_ml_probabilities(
    scores: np.ndarray,
) -> np.ndarray:
    """Convert ML outputs to probabilities."""
    if APPLY_SOFTMAX:
        return softmax(scores)

    if np.any(scores < 0.0):
        raise ValueError(
            "ML probabilities must be non-negative."
        )

    if not np.allclose(
        np.sum(scores, axis=1),
        1.0,
    ):
        raise ValueError(
            "ML probability rows must sum to one."
        )

    return scores


def load_ml_data() -> ClassificationDependencyData:
    """Load ML classification outputs and truth information."""
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
        raise ValueError(
            f"scores must have shape (N, C), got {scores.shape}."
        )

    if scores.shape[1] != len(CLASS_LABELS):
        raise ValueError(
            "The number of ML score columns does not match "
            f"CLASS_LABELS: {scores.shape[1]} != "
            f"{len(CLASS_LABELS)}."
        )

    if not np.array_equal(
        targets,
        truth_labels,
    ):
        mismatch = np.flatnonzero(
            targets != truth_labels
        )
        first = int(mismatch[0])

        raise ValueError(
            "ML targets and truth labels are not aligned. "
            f"First mismatch at index {first}: "
            f"target={targets[first]}, "
            f"truth={truth_labels[first]}."
        )

    (
        arrays,
        removed_count,
    ) = remove_nonfinite_events(
        scores,
        targets,
        positions,
        directions,
        energies,
    )

    (
        scores,
        targets,
        positions,
        directions,
        energies,
    ) = arrays

    unknown_labels = np.setdiff1d(
        np.unique(targets),
        CLASS_LABELS,
    )

    if unknown_labels.size:
        raise ValueError(
            "ML targets contain labels not listed in "
            f"CLASS_LABELS: {unknown_labels.tolist()}."
        )

    probabilities = create_ml_probabilities(
        scores
    )

    prediction_indices = predicted_labels(
        probabilities
    )
    predictions = CLASS_LABELS[
        prediction_indices
    ]

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

    print(
        "ML non-finite events removed: "
        f"{removed_count:,}"
    )

    return ClassificationDependencyData(
        name="ML",
        predictions=predictions,
        targets=targets,
        wall=wall,
        towall=towall,
        energy=energies,
        color=ML_COLOR,
        marker=ML_MARKER,
        linestyle=ML_LINESTYLE,
    )


def load_fitqun_data(
    path: Path,
) -> ClassificationDependencyData:
    """Load fiTQun classification outputs and truth information."""
    data = load_fitqun(
        path,
        keys=FITQUN_KEYS,
    )

    (
        arrays,
        removed_count,
    ) = remove_nonfinite_events(
        data.electron_nll,
        data.muon_nll,
        data.electron_momentum,
        data.labels,
        data.truth_positions,
        data.truth_directions,
        data.truth_energies,
    )

    (
        electron_nll,
        muon_nll,
        electron_momentum,
        targets,
        positions,
        directions,
        energies,
    ) = arrays

    score = (
        muon_nll
        - electron_nll
        - 0.2 * electron_momentum
    )

    predictions = np.where(
        score > FITQUN_SCORE_THRESHOLD,
        ELECTRON_LABEL,
        MUON_LABEL,
    ).astype(np.int64)

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

    print(
        "fiTQun non-finite events removed: "
        f"{removed_count:,}"
    )

    return ClassificationDependencyData(
        name="fiTQun",
        predictions=predictions,
        targets=targets,
        wall=wall,
        towall=towall,
        energy=energies,
        color=FITQUN_COLOR,
        marker=FITQUN_MARKER,
        linestyle=FITQUN_LINESTYLE,
    )


def calculate_dependency_metrics(
    method: ClassificationDependencyData,
    *,
    dependency_name: str,
    edges: np.ndarray,
    class_label: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Calculate binned efficiency and purity."""
    dependency = getattr(
        method,
        dependency_name,
    )

    return binned_classification_metrics(
        dependency,
        method.predictions,
        method.targets,
        edges,
        class_label=class_label,
        minimum_events=MINIMUM_EVENTS,
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
    """Plot efficiency or purity for all methods and classes."""
    if metric not in {
        "efficiency",
        "purity",
    }:
        raise ValueError(
            "metric must be 'efficiency' or 'purity'."
        )

    figure, axes = plt.subplots(
        len(class_labels),
        len(dependencies),
        figsize=(
            FIGURE_WIDTH_PER_COLUMN
            * len(dependencies),
            FIGURE_HEIGHT_PER_ROW
            * len(class_labels),
        ),
        squeeze=False,
    )

    csv_rows: list[list[object]] = []

    for row, class_label in enumerate(
        class_labels
    ):
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
                    efficiency,
                    purity,
                    counts,
                ) = calculate_dependency_metrics(
                    method,
                    dependency_name=dependency_name,
                    edges=edges,
                    class_label=label,
                )

                values = (
                    efficiency
                    if metric == "efficiency"
                    else purity
                )

                valid = np.isfinite(values)

                plot_series(
                    centers[valid],
                    values[valid],
                    ax=axis,
                    label=method.name,
                    color=method.color,
                    marker=method.marker,
                    linestyle=method.linestyle,
                    linewidth=LINEWIDTH,
                )

                for center, value, count in zip(
                    centers,
                    values,
                    counts,
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
                            count,
                        ]
                    )

            axis.set_xlabel(xlabel)
            axis.set_ylabel(
                metric.capitalize()
            )
            axis.set_ylim(0.0, 1.05)
            axis.set_title(
                f"{class_name} {metric} "
                f"vs {dependency_title}"
            )
            axis.grid(alpha=0.3)
            axis.legend()

    comparison_enabled = len(methods) > 1

    title_suffix = (
        "ML vs fiTQun"
        if comparison_enabled
        else "ML"
    )

    figure.suptitle(
        (
            f"Classification {metric} dependencies: "
            f"{title_suffix}"
        ),
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
            "event_count",
        ),
        fmt="%s",
    )


def main() -> None:
    ml_data = load_ml_data()

    methods = [ml_data]

    if FITQUN_PATH is not None:
        fitqun_data = load_fitqun_data(
            FITQUN_PATH
        )

        # Compare only classes supported by fiTQun.
        class_labels = COMPARISON_CLASS_LABELS

        methods = [
            select_classes(
                ml_data,
                class_labels,
            ),
            select_classes(
                fitqun_data,
                class_labels,
            ),
        ]
    else:
        class_labels = CLASS_LABELS

    for method in methods:
        print(
            f"{method.name} events: "
            f"{method.event_count:,}"
        )

        unique_labels, counts = np.unique(
            method.targets,
            return_counts=True,
        )

        for label, count in zip(
            unique_labels,
            counts,
            strict=True,
        ):
            class_name = CLASS_NAMES.get(
                int(label),
                str(label),
            )
            print(
                f"  {class_name}: {count:,}"
            )

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
        class_labels,
        metric="efficiency",
    )

    (
        purity_figure,
        _,
        purity_rows,
    ) = plot_metric_dependencies(
        methods,
        dependencies,
        class_labels,
        metric="purity",
    )

    comparison_enabled = len(methods) > 1

    efficiency_filename = (
        COMPARISON_EFFICIENCY_FILENAME
        if comparison_enabled
        else ML_ONLY_EFFICIENCY_FILENAME
    )
    purity_filename = (
        COMPARISON_PURITY_FILENAME
        if comparison_enabled
        else ML_ONLY_PURITY_FILENAME
    )
    csv_filename = (
        COMPARISON_CSV_FILENAME
        if comparison_enabled
        else ML_ONLY_CSV_FILENAME
    )

    efficiency_path = (
        OUTPUT_DIRECTORY
        / efficiency_filename
    )
    purity_path = (
        OUTPUT_DIRECTORY
        / purity_filename
    )
    csv_path = (
        OUTPUT_DIRECTORY
        / csv_filename
    )

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

    print(f"Saved: {efficiency_path}")
    print(f"Saved: {purity_path}")
    print(f"Saved: {csv_path}")

    if SHOW_FIGURES:
        plt.show()


if __name__ == "__main__":
    main()