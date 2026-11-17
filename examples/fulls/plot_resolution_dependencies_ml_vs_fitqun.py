"""Plot ML resolution dependencies, optionally compared with fiTQun.

Set FITQUN_PATH to None to plot only ML results.

Edit the settings below and run:

    uv run python examples/plot_resolution_dependencies.py
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis_tools.data import (
    as_scalar_array,
    as_vector_array,
    validate_same_length,
)
from analysis_tools.io import (
    FitQunKeys,
    load_array_pair,
    load_fitqun,
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
# ML input files
# ============================================================

VERTEX_PREDICTIONS_PATH = Path("path/to/vertex/outputs/preds.npy")
VERTEX_TARGETS_PATH = Path("path/to/vertex/outputs/targets.npy")

DIRECTION_PREDICTIONS_PATH = Path("path/to/direction/outputs/preds.npy")
DIRECTION_TARGETS_PATH = Path("path/to/direction/outputs/targets.npy")

ENERGY_PREDICTIONS_PATH = Path("path/to/energy/outputs/preds.npy")
ENERGY_TARGETS_PATH = Path("path/to/energy/outputs/targets.npy")


# ============================================================
# Optional fiTQun input
# ============================================================

# Set to None to plot ML results only.
FITQUN_PATH: Path | None = Path("path/to/common_multi_fitqun.hy")

# FITQUN_PATH: Path | None = None

FITQUN_KEYS = FitQunKeys(
    labels="multi/labels",
    truth_positions="multi/positions",
    truth_directions="multi/directions",
    truth_energies="multi/energies",
    electron_nll="fitqun/e_1rnll",
    muon_nll="fitqun/mu_1rnll",
    electron_momentum="fitqun/e_1rmom",
    reconstructed_positions="fitqun/e_1rpos",
    reconstructed_directions="fitqun/e_1rdir",
    reconstructed_momenta="fitqun/e_1rmom",
)

# Used to convert reconstructed momentum to total energy.
# Change this when using another particle hypothesis.
FITQUN_PARTICLE_MASS_MEV = 0.51099895


# ============================================================
# Output
# ============================================================

OUTPUT_DIRECTORY = Path("outputs/resolution_dependencies")

ML_ONLY_FIGURE_FILENAME = "resolution_dependencies.png"

COMPARISON_FIGURE_FILENAME = "resolution_dependencies_ml_vs_fitqun.png"

ML_ONLY_CSV_FILENAME = "resolution_dependencies.csv"

COMPARISON_CSV_FILENAME = "resolution_dependencies_ml_vs_fitqun.csv"


# ============================================================
# Detector geometry
# ============================================================

DETECTOR_RADIUS_CM = 1690.0
DETECTOR_HALF_HEIGHT_CM = 1810.0


# ============================================================
# ML normalization
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

ML_COLOR = "tab:blue"
ML_MARKER = "o"
ML_LINESTYLE = "-"

FITQUN_COLOR = "black"
FITQUN_MARKER = "s"
FITQUN_LINESTYLE = "--"


@dataclass(frozen=True)
class ResolutionData:
    """Quantities used for resolution-dependency plots."""

    name: str

    wall: np.ndarray
    towall: np.ndarray
    energy: np.ndarray

    vertex_error: np.ndarray
    direction_error: np.ndarray
    energy_error: np.ndarray

    color: str
    marker: str
    linestyle: str

    @property
    def event_count(self) -> int:
        return len(self.wall)


def remove_nonfinite_events(
    *arrays: np.ndarray,
) -> tuple[tuple[np.ndarray, ...], int]:
    """Remove events containing non-finite values in any array."""
    if not arrays:
        raise ValueError("At least one array is required.")

    validate_same_length(*arrays)

    valid_mask = np.ones(
        len(arrays[0]),
        dtype=bool,
    )

    for array in arrays:
        if array.ndim == 1:
            valid_mask &= np.isfinite(array)
        else:
            axes = tuple(range(1, array.ndim))
            valid_mask &= np.all(
                np.isfinite(array),
                axis=axes,
            )

    removed_count = int(np.count_nonzero(~valid_mask))

    selected = tuple(array[valid_mask] for array in arrays)

    if len(selected[0]) == 0:
        raise ValueError("No finite events remain after filtering.")

    return selected, removed_count


def load_ml_resolution_data() -> ResolutionData:
    """Load ML outputs and calculate reconstruction errors."""
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

    arrays, removed_count = remove_nonfinite_events(
        vertex_predictions,
        vertex_targets,
        direction_predictions,
        direction_targets,
        energy_predictions,
        energy_targets,
    )

    (
        vertex_predictions,
        vertex_targets,
        direction_predictions,
        direction_targets,
        energy_predictions,
        energy_targets,
    ) = arrays

    wall = compute_wall(
        vertex_targets,
        radius=DETECTOR_RADIUS_CM,
        half_height=DETECTOR_HALF_HEIGHT_CM,
    )

    towall = compute_towall(
        vertex_targets,
        direction_targets,
        radius=DETECTOR_RADIUS_CM,
        half_height=DETECTOR_HALF_HEIGHT_CM,
    )

    print(f"ML non-finite events removed: {removed_count:,}")

    return ResolutionData(
        name="ML",
        wall=wall,
        towall=towall,
        energy=energy_targets,
        vertex_error=vertex_errors(
            vertex_predictions,
            vertex_targets,
        ),
        direction_error=direction_errors(
            direction_predictions,
            direction_targets,
            degrees=True,
        ),
        energy_error=fractional_energy_residuals(
            energy_predictions,
            energy_targets,
        ),
        color=ML_COLOR,
        marker=ML_MARKER,
        linestyle=ML_LINESTYLE,
    )


def load_fitqun_resolution_data(
    path: Path,
) -> ResolutionData:
    """Load fiTQun outputs and calculate reconstruction errors."""
    data = load_fitqun(
        path,
        keys=FITQUN_KEYS,
    )

    if data.reconstructed_positions is None:
        raise ValueError("fiTQun reconstructed positions are unavailable.")

    if data.reconstructed_directions is None:
        raise ValueError("fiTQun reconstructed directions are unavailable.")

    if data.reconstructed_momenta is None:
        raise ValueError("fiTQun reconstructed momenta are unavailable.")

    arrays, removed_count = remove_nonfinite_events(
        data.truth_positions,
        data.truth_directions,
        data.truth_energies,
        data.reconstructed_positions,
        data.reconstructed_directions,
        data.reconstructed_momenta,
    )

    (
        truth_positions,
        truth_directions,
        truth_energies,
        reconstructed_positions,
        reconstructed_directions,
        reconstructed_momenta,
    ) = arrays

    reconstructed_energies = np.sqrt(
        reconstructed_momenta**2 + FITQUN_PARTICLE_MASS_MEV**2
    )

    wall = compute_wall(
        truth_positions,
        radius=DETECTOR_RADIUS_CM,
        half_height=DETECTOR_HALF_HEIGHT_CM,
    )

    towall = compute_towall(
        truth_positions,
        truth_directions,
        radius=DETECTOR_RADIUS_CM,
        half_height=DETECTOR_HALF_HEIGHT_CM,
    )

    print(f"fiTQun non-finite events removed: {removed_count:,}")

    return ResolutionData(
        name="fiTQun",
        wall=wall,
        towall=towall,
        energy=truth_energies,
        vertex_error=vertex_errors(
            reconstructed_positions,
            truth_positions,
        ),
        direction_error=direction_errors(
            reconstructed_directions,
            truth_directions,
            degrees=True,
        ),
        energy_error=fractional_energy_residuals(
            reconstructed_energies,
            truth_energies,
        ),
        color=FITQUN_COLOR,
        marker=FITQUN_MARKER,
        linestyle=FITQUN_LINESTYLE,
    )


def calculate_binned_resolution(
    dependency: np.ndarray,
    errors: np.ndarray,
    edges: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Calculate the absolute-error quantile in each bin."""
    return binned_absolute_quantile(
        dependency,
        errors,
        edges,
        quantile_level=QUANTILE_LEVEL,
        minimum_events=MINIMUM_EVENTS,
    )


def add_resolution_series(
    axis,
    method: ResolutionData,
    *,
    dependency_name: str,
    error_name: str,
    edges: np.ndarray,
    scale: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Add one reconstruction method to an Axes."""
    dependency = getattr(
        method,
        dependency_name,
    )
    errors = getattr(
        method,
        error_name,
    )

    centers, resolution, counts = calculate_binned_resolution(
        dependency,
        errors * scale,
        edges,
    )

    valid = np.isfinite(resolution)

    plot_series(
        centers[valid],
        resolution[valid],
        ax=axis,
        label=method.name,
        color=method.color,
        marker=method.marker,
        linestyle=method.linestyle,
        linewidth=2.0,
    )

    return centers, resolution, counts


def print_correlations(
    methods: list[ResolutionData],
) -> None:
    """Print Pearson correlations for each method."""
    dependency_names = (
        ("wall", "True Wall"),
        ("towall", "True ToWall"),
        ("energy", "True energy"),
    )

    error_names = (
        ("vertex_error", "Vertex error"),
        ("direction_error", "Direction error"),
        ("energy_error", "Fractional energy error"),
    )

    print("\n========== Pearson correlations ==========")

    for method in methods:
        print(f"\n{method.name}:")

        for error_attribute, error_label in error_names:
            print(f"  {error_label}:")

            errors = getattr(
                method,
                error_attribute,
            )

            for (
                dependency_attribute,
                dependency_label,
            ) in dependency_names:
                dependency = getattr(
                    method,
                    dependency_attribute,
                )

                correlation = pearson_correlation(
                    dependency,
                    errors,
                )

                print(f"    {dependency_label}: {correlation:+.6f}")

    print("==========================================\n")


def save_results_csv(
    path: Path,
    rows: list[list[object]],
) -> None:
    """Save long-format resolution results."""
    save_csv(
        path,
        np.asarray(
            rows,
            dtype=object,
        ),
        column_names=(
            "method",
            "dependency",
            "task",
            "bin_center",
            "resolution",
            "event_count",
        ),
        fmt="%s",
    )


def main() -> None:
    ml_data = load_ml_resolution_data()

    methods = [ml_data]

    if FITQUN_PATH is not None:
        fitqun_data = load_fitqun_resolution_data(FITQUN_PATH)
        methods.append(fitqun_data)

    for method in methods:
        print(f"{method.name} events: {method.event_count:,}")

    print_correlations(methods)

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

    tasks = (
        (
            "vertex_error",
            "Vertex resolution [cm]",
            "Vertex",
            1.0,
        ),
        (
            "direction_error",
            "Direction resolution [degree]",
            "Direction",
            1.0,
        ),
        (
            "energy_error",
            "Energy resolution [%]",
            "Energy",
            100.0,
        ),
    )

    figure, axes = plt.subplots(
        3,
        3,
        figsize=FIGSIZE,
    )

    csv_rows: list[list[object]] = []

    for column, (
        dependency_name,
        edges,
        xlabel,
        dependency_title,
    ) in enumerate(dependencies):
        for row, (
            error_name,
            ylabel,
            task_name,
            scale,
        ) in enumerate(tasks):
            axis = axes[row, column]

            for method in methods:
                centers, resolution, counts = add_resolution_series(
                    axis,
                    method,
                    dependency_name=dependency_name,
                    error_name=error_name,
                    edges=edges,
                    scale=scale,
                )

                for center, value, count in zip(
                    centers,
                    resolution,
                    counts,
                    strict=True,
                ):
                    csv_rows.append(
                        [
                            method.name,
                            dependency_name,
                            task_name,
                            center,
                            value,
                            count,
                        ]
                    )

            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel)
            axis.set_title(f"{task_name} vs {dependency_title}")
            axis.grid(alpha=0.3)
            axis.legend()

    comparison_enabled = len(methods) > 1

    title_suffix = "ML vs fiTQun" if comparison_enabled else "ML"

    figure.suptitle(
        (f"{100.0 * QUANTILE_LEVEL:.0f}% resolution dependencies: {title_suffix}"),
        fontsize=16,
    )
    figure.tight_layout()

    figure_filename = (
        COMPARISON_FIGURE_FILENAME if comparison_enabled else ML_ONLY_FIGURE_FILENAME
    )
    csv_filename = (
        COMPARISON_CSV_FILENAME if comparison_enabled else ML_ONLY_CSV_FILENAME
    )

    figure_path = OUTPUT_DIRECTORY / figure_filename
    csv_path = OUTPUT_DIRECTORY / csv_filename

    save_figure(
        figure,
        figure_path,
        dpi=DPI,
        close=not SHOW_FIGURE,
    )

    save_results_csv(
        csv_path,
        csv_rows,
    )

    print(f"Saved: {figure_path}")
    print(f"Saved: {csv_path}")

    if SHOW_FIGURE:
        plt.show()


if __name__ == "__main__":
    main()
