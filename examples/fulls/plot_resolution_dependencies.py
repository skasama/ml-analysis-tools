"""Plot vertex, direction, and energy resolution dependencies.

The three prediction files must correspond to the same events in the
same order.

Edit the settings below and run:

    uv run python examples/plot_resolution_dependencies.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis_tools.data import (
    as_scalar_array,
    as_vector_array,
    validate_same_length,
)
from analysis_tools.io import (
    load_array_pair,
    save_csv,
)
from analysis_tools.metrics import (
    binned_absolute_quantile,
    direction_errors,
    fractional_energy_residuals,
    make_bin_edges,
    pearson_correlation,
    vertex_errors,
)
from analysis_tools.plotting import save_figure
from analysis_tools.plotting.common import plot_series
from analysis_tools.transforms import MinMaxTransform
from analysis_tools.transforms.geometry import (
    compute_towall,
    compute_wall,
)

# ============================================================
# Input files
# ============================================================

VERTEX_PREDICTIONS_PATH = Path(
    "../../tmp/2026-10_gat_e100k/regression_vertex_61467821_e100k/outputs/preds.npy"
)
VERTEX_TARGETS_PATH = Path(
    "../../tmp/2026-10_gat_e100k/regression_vertex_61467821_e100k/outputs/targets.npy"
)

DIRECTION_PREDICTIONS_PATH = Path(
    "../../tmp/2026-10_gat_e100k/regression_directions_64914021_e100k/outputs/preds.npy"
)
DIRECTION_TARGETS_PATH = Path(
    "../../tmp/2026-10_gat_e100k/regression_directions_64914021_e100k/outputs/targets.npy"
)

ENERGY_PREDICTIONS_PATH = Path(
    "../../tmp/2026-10_gat_e100k/regression_energy_62378219_e100k/outputs/preds.npy"
)
ENERGY_TARGETS_PATH = Path(
    "../../tmp/2026-10_gat_e100k/regression_energy_62378219_e100k/outputs/targets.npy"
)


# ============================================================
# Output
# ============================================================

OUTPUT_DIRECTORY = Path("outputs/resolution_dependencies")

FIGURE_FILENAME = "resolution_dependencies.png"

WALL_CSV_FILENAME = "resolution_vs_wall.csv"
TOWALL_CSV_FILENAME = "resolution_vs_towall.csv"
ENERGY_CSV_FILENAME = "resolution_vs_energy.csv"


# ============================================================
# Detector geometry
# ============================================================

DETECTOR_RADIUS_CM = 1690.0
DETECTOR_HALF_HEIGHT_CM = 1810.0


# ============================================================
# Normalization
# ============================================================

VERTEX_MINIMUM_CM = [-1690.0, -1690.0, -1810.0]
VERTEX_MAXIMUM_CM = [1690.0, 1690.0, 1810.0]

APPLY_INVERSE_DIRECTION_NORMALIZATION = True
DIRECTION_MINIMUM = [-1.0, -1.0, -1.0]
DIRECTION_MAXIMUM = [1.0, 1.0, 1.0]

ENERGY_MINIMUM_MEV = 0.5
ENERGY_MAXIMUM_MEV = 2000.0


# ============================================================
# Binning
# ============================================================

QUANTILE_LEVEL = 0.68
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
SHOW_FIGURE = True

FIGSIZE = (17.0, 14.0)

VERTEX_COLOR = "tab:blue"
DIRECTION_COLOR = "tab:orange"
ENERGY_COLOR = "tab:green"

MARKER = "o"
LINESTYLE = "-"
REFERENCE_COLOR = "black"


def load_data() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Load and standardize the three regression outputs."""
    vertex_predictions, vertex_targets = load_array_pair(
        VERTEX_PREDICTIONS_PATH,
        VERTEX_TARGETS_PATH,
        first_name="vertex_predictions",
        second_name="vertex_targets",
    )
    direction_predictions, direction_targets = load_array_pair(
        DIRECTION_PREDICTIONS_PATH,
        DIRECTION_TARGETS_PATH,
        first_name="direction_predictions",
        second_name="direction_targets",
    )
    energy_predictions, energy_targets = load_array_pair(
        ENERGY_PREDICTIONS_PATH,
        ENERGY_TARGETS_PATH,
        first_name="energy_predictions",
        second_name="energy_targets",
    )

    vertex_predictions = as_vector_array(
        vertex_predictions,
        size=3,
        name="vertex_predictions",
    ).astype(float)
    vertex_targets = as_vector_array(
        vertex_targets,
        size=3,
        name="vertex_targets",
    ).astype(float)

    direction_predictions = as_vector_array(
        direction_predictions,
        size=3,
        name="direction_predictions",
    ).astype(float)
    direction_targets = as_vector_array(
        direction_targets,
        size=3,
        name="direction_targets",
    ).astype(float)

    energy_predictions = as_scalar_array(
        energy_predictions,
        name="energy_predictions",
    ).astype(float)
    energy_targets = as_scalar_array(
        energy_targets,
        name="energy_targets",
    ).astype(float)

    validate_same_length(
        vertex_predictions,
        vertex_targets,
        direction_predictions,
        direction_targets,
        energy_predictions,
        energy_targets,
        names=(
            "vertex_predictions",
            "vertex_targets",
            "direction_predictions",
            "direction_targets",
            "energy_predictions",
            "energy_targets",
        ),
    )

    return (
        vertex_predictions,
        vertex_targets,
        direction_predictions,
        direction_targets,
        energy_predictions,
        energy_targets,
    )


def inverse_transform(
    vertex_predictions: np.ndarray,
    vertex_targets: np.ndarray,
    direction_predictions: np.ndarray,
    direction_targets: np.ndarray,
    energy_predictions: np.ndarray,
    energy_targets: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Convert normalized outputs to physical quantities."""
    vertex_transform = MinMaxTransform(
        minimum=VERTEX_MINIMUM_CM,
        maximum=VERTEX_MAXIMUM_CM,
    )
    energy_transform = MinMaxTransform(
        minimum=ENERGY_MINIMUM_MEV,
        maximum=ENERGY_MAXIMUM_MEV,
    )

    vertex_predictions = vertex_transform.inverse(vertex_predictions)
    vertex_targets = vertex_transform.inverse(vertex_targets)

    energy_predictions = energy_transform.inverse(energy_predictions)
    energy_targets = energy_transform.inverse(energy_targets)

    if APPLY_INVERSE_DIRECTION_NORMALIZATION:
        direction_transform = MinMaxTransform(
            minimum=DIRECTION_MINIMUM,
            maximum=DIRECTION_MAXIMUM,
        )
        direction_predictions = direction_transform.inverse(direction_predictions)
        direction_targets = direction_transform.inverse(direction_targets)

    return (
        vertex_predictions,
        vertex_targets,
        direction_predictions,
        direction_targets,
        energy_predictions,
        energy_targets,
    )


def remove_nonfinite_events(
    *arrays: np.ndarray,
) -> tuple[tuple[np.ndarray, ...], int]:
    """Remove events containing non-finite values in any array."""
    valid_mask = np.ones(len(arrays[0]), dtype=bool)

    for array in arrays:
        if array.ndim == 1:
            valid_mask &= np.isfinite(array)
        else:
            valid_mask &= np.all(
                np.isfinite(array),
                axis=tuple(range(1, array.ndim)),
            )

    removed_count = int(np.count_nonzero(~valid_mask))

    selected = tuple(array[valid_mask] for array in arrays)

    if not selected or len(selected[0]) == 0:
        raise ValueError("No finite events remain after filtering.")

    return selected, removed_count


def calculate_binned_resolution(
    dependency: np.ndarray,
    errors: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the absolute-error quantile in each bin."""
    return binned_absolute_quantile(
        dependency,
        errors,
        edges,
        quantile_level=QUANTILE_LEVEL,
        minimum_events=MINIMUM_EVENTS,
    )


def add_resolution_plot(
    axis,
    dependency: np.ndarray,
    errors: np.ndarray,
    edges: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    color: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate and draw one resolution dependency."""
    centers, resolution, counts = calculate_binned_resolution(
        dependency,
        errors,
        edges,
    )

    valid = np.isfinite(resolution)

    plot_series(
        centers[valid],
        resolution[valid],
        ax=axis,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        color=color,
        marker=MARKER,
        linestyle=LINESTYLE,
    )

    axis.grid(alpha=0.3)

    return centers, resolution, counts


def save_dependency_csv(
    path: Path,
    centers: np.ndarray,
    vertex_resolution: np.ndarray,
    vertex_counts: np.ndarray,
    direction_resolution: np.ndarray,
    direction_counts: np.ndarray,
    energy_resolution: np.ndarray,
    energy_counts: np.ndarray,
) -> None:
    """Save the three resolutions for one dependency."""
    table = np.column_stack(
        (
            centers,
            vertex_resolution,
            vertex_counts,
            direction_resolution,
            direction_counts,
            energy_resolution,
            energy_counts,
        )
    )

    save_csv(
        path,
        table,
        column_names=(
            "bin_center",
            "vertex_resolution_cm",
            "vertex_count",
            "direction_resolution_degree",
            "direction_count",
            "energy_resolution_fraction",
            "energy_count",
        ),
    )


def print_correlations(
    dependencies: tuple[
        tuple[str, np.ndarray],
        ...,
    ],
    errors: tuple[
        tuple[str, np.ndarray],
        ...,
    ],
) -> None:
    """Print Pearson correlations for all dependencies."""
    print("\n========== Pearson correlations ==========")

    for error_name, error_values in errors:
        print(f"\n{error_name}:")

        for dependency_name, dependency_values in dependencies:
            correlation = pearson_correlation(
                dependency_values,
                error_values,
            )
            print(f"  {dependency_name}: {correlation:+.6f}")

    print("==========================================\n")


def main() -> None:
    arrays = load_data()
    arrays = inverse_transform(*arrays)

    arrays, removed_count = remove_nonfinite_events(*arrays)

    (
        vertex_predictions,
        vertex_targets,
        direction_predictions,
        direction_targets,
        energy_predictions,
        energy_targets,
    ) = arrays

    vertex_error_cm = vertex_errors(
        vertex_predictions,
        vertex_targets,
    )
    direction_error_degree = direction_errors(
        direction_predictions,
        direction_targets,
        degrees=True,
    )
    energy_error_fraction = fractional_energy_residuals(
        energy_predictions,
        energy_targets,
    )

    true_wall_cm = compute_wall(
        vertex_targets,
        radius=DETECTOR_RADIUS_CM,
        half_height=DETECTOR_HALF_HEIGHT_CM,
    )
    true_towall_cm = compute_towall(
        vertex_targets,
        direction_targets,
        radius=DETECTOR_RADIUS_CM,
        half_height=DETECTOR_HALF_HEIGHT_CM,
    )

    finite_energy_fraction = np.isfinite(energy_error_fraction)

    if not np.all(finite_energy_fraction):
        removed_fractional = int(np.count_nonzero(~finite_energy_fraction))
        print(
            f"Events excluded from fractional-energy analysis: {removed_fractional:,}"
        )

    print(f"Events: {len(vertex_error_cm):,}")
    print(f"Removed non-finite events: {removed_count:,}")

    print_correlations(
        dependencies=(
            ("True Wall", true_wall_cm),
            ("True ToWall", true_towall_cm),
            ("True energy", energy_targets),
        ),
        errors=(
            ("Vertex error", vertex_error_cm),
            ("Direction error", direction_error_degree),
            (
                "Fractional energy error",
                energy_error_fraction,
            ),
        ),
    )

    dependencies = (
        (
            true_wall_cm,
            WALL_EDGES,
            "True Wall [cm]",
            "wall",
        ),
        (
            true_towall_cm,
            TOWALL_EDGES,
            "True ToWall [cm]",
            "towall",
        ),
        (
            energy_targets,
            ENERGY_EDGES,
            "True energy [MeV]",
            "energy",
        ),
    )

    tasks = (
        (
            vertex_error_cm,
            "Vertex resolution [cm]",
            "Vertex",
            VERTEX_COLOR,
        ),
        (
            direction_error_degree,
            "Direction resolution [degree]",
            "Direction",
            DIRECTION_COLOR,
        ),
        (
            energy_error_fraction,
            "Energy resolution",
            "Energy",
            ENERGY_COLOR,
        ),
    )

    figure, axes = plt.subplots(
        3,
        3,
        figsize=FIGSIZE,
    )

    csv_results = {}

    for column, (
        dependency_values,
        edges,
        xlabel,
        dependency_name,
    ) in enumerate(dependencies):
        task_results = []

        for row, (
            error_values,
            ylabel,
            task_name,
            color,
        ) in enumerate(tasks):
            centers, resolution, counts = add_resolution_plot(
                axes[row, column],
                dependency_values,
                error_values,
                edges,
                xlabel=xlabel,
                ylabel=ylabel,
                title=f"{task_name} vs {xlabel.removeprefix('True ')}",
                color=color,
            )

            task_results.append((resolution, counts))

        csv_results[dependency_name] = (
            centers,
            task_results,
        )

    figure.suptitle(
        (f"{100.0 * QUANTILE_LEVEL:.0f}% reconstruction resolution dependencies"),
        fontsize=16,
    )
    figure.tight_layout()

    save_figure(
        figure,
        OUTPUT_DIRECTORY / FIGURE_FILENAME,
        dpi=DPI,
        close=not SHOW_FIGURE,
    )

    csv_filenames = {
        "wall": WALL_CSV_FILENAME,
        "towall": TOWALL_CSV_FILENAME,
        "energy": ENERGY_CSV_FILENAME,
    }

    for dependency_name, (
        centers,
        task_results,
    ) in csv_results.items():
        (
            vertex_result,
            direction_result,
            energy_result,
        ) = task_results

        save_dependency_csv(
            OUTPUT_DIRECTORY / csv_filenames[dependency_name],
            centers,
            vertex_result[0],
            vertex_result[1],
            direction_result[0],
            direction_result[1],
            energy_result[0],
            energy_result[1],
        )

    print(f"Saved: {OUTPUT_DIRECTORY / FIGURE_FILENAME}")
    print(f"Saved CSV files to: {OUTPUT_DIRECTORY}")

    if SHOW_FIGURE:
        plt.show()


if __name__ == "__main__":
    main()
