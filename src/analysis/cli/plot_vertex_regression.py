import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.geometry import cylindrical_radius, wall
from analysis.io import load_outputs
from analysis.paths import output_path
from analysis.plotting import save
from analysis.regression import inverse_minmax, xyz

UNIT = "cm"
EPSILON = 1.0e-10

COORDINATE_LABELS = [
    "x",
    "y",
    "z",
    "R",
    "Wall",
]


def calculate_resolution(residuals):
    """Return the 68th percentile of the absolute residual."""
    residuals = np.asarray(
        residuals,
        dtype=float,
    )
    residuals = residuals[np.isfinite(residuals)]

    if residuals.size == 0:
        return np.nan

    return float(
        np.quantile(
            np.abs(residuals),
            0.68,
        )
    )


def calculate_profile(
    x_values,
    residuals,
    *,
    n_bins,
):
    """Calculate bias, resolution, and count in bins of x."""
    x_values = np.asarray(
        x_values,
        dtype=float,
    )
    residuals = np.asarray(
        residuals,
        dtype=float,
    )

    finite = (
        np.isfinite(x_values)
        & np.isfinite(residuals)
    )

    x_values = x_values[finite]
    residuals = residuals[finite]

    if x_values.size == 0:
        raise ValueError(
            "No finite values are available "
            "for profile calculation."
        )

    x_min = float(np.min(x_values))
    x_max = float(np.max(x_values))

    if x_max <= x_min:
        x_max = x_min + EPSILON

    edges = np.linspace(
        x_min,
        x_max,
        n_bins + 1,
    )
    centres = 0.5 * (
        edges[:-1] + edges[1:]
    )

    biases = np.full(
        n_bins,
        np.nan,
    )
    resolutions = np.full(
        n_bins,
        np.nan,
    )
    counts = np.zeros(
        n_bins,
        dtype=int,
    )

    for index in range(n_bins):
        if index == n_bins - 1:
            in_bin = (
                (x_values >= edges[index])
                & (x_values <= edges[index + 1])
            )
        else:
            in_bin = (
                (x_values >= edges[index])
                & (x_values < edges[index + 1])
            )

        values = residuals[in_bin]
        counts[index] = len(values)

        if values.size > 0:
            biases[index] = np.mean(values)
            resolutions[index] = calculate_resolution(
                values
            )

    return (
        centres,
        biases,
        resolutions,
        counts,
    )


def calculate_vertex_quantities(
    predictions,
    targets,
    *,
    tank_radius,
    tank_half_height,
):
    """Calculate residuals and geometry-dependent quantities."""
    residuals = predictions - targets

    radius_predicted = cylindrical_radius(
        predictions
    )
    radius_true = cylindrical_radius(
        targets
    )

    wall_predicted = wall(
        predictions,
        radius=tank_radius,
        half_height=tank_half_height,
    )
    wall_true = wall(
        targets,
        radius=tank_radius,
        half_height=tank_half_height,
    )

    return {
        "residuals": residuals,
        "error_3d": np.linalg.norm(
            residuals,
            axis=1,
        ),
        "radius_predicted": radius_predicted,
        "radius_true": radius_true,
        "radius_residual": (
            radius_predicted - radius_true
        ),
        "wall_predicted": wall_predicted,
        "wall_true": wall_true,
        "wall_residual": (
            wall_predicted - wall_true
        ),
    }


def print_statistics(
    predictions,
    targets,
    *,
    tank_radius,
    tank_half_height,
):
    """Print Cartesian, cylindrical, Wall, and 3D statistics."""
    quantities = calculate_vertex_quantities(
        predictions,
        targets,
        tank_radius=tank_radius,
        tank_half_height=tank_half_height,
    )

    residuals = quantities["residuals"]
    error_3d = quantities["error_3d"]
    radius_residual = quantities[
        "radius_residual"
    ]
    wall_residual = quantities[
        "wall_residual"
    ]

    print()
    print(
        "========== Vertex regression statistics =========="
    )
    print(f"Number of events: {len(targets):,}")
    print(f"Unit: {UNIT}")

    print()
    print("Cartesian residuals:")

    for index, label in enumerate(
        ("x", "y", "z")
    ):
        values = residuals[:, index]

        print(
            f"  {label}: "
            f"bias={np.mean(values):.6g}, "
            f"resolution="
            f"{calculate_resolution(values):.6g} "
            f"{UNIT}"
        )

    print()
    print("Cylindrical and Wall residuals:")
    print(
        "  R:    "
        f"bias={np.mean(radius_residual):.6g}, "
        f"resolution="
        f"{calculate_resolution(radius_residual):.6g} "
        f"{UNIT}"
    )
    print(
        "  Wall: "
        f"bias={np.mean(wall_residual):.6g}, "
        f"resolution="
        f"{calculate_resolution(wall_residual):.6g} "
        f"{UNIT}"
    )

    print()
    print("3D position error:")
    print(
        f"  Mean: {np.mean(error_3d):.6g} "
        f"{UNIT}"
    )
    print(
        f"  Median: {np.median(error_3d):.6g} "
        f"{UNIT}"
    )
    print(
        "  68% quantile: "
        f"{np.quantile(error_3d, 0.68):.6g} "
        f"{UNIT}"
    )
    print(
        "  RMS: "
        f"{np.sqrt(np.mean(error_3d**2)):.6g} "
        f"{UNIT}"
    )
    print(
        "=================================================="
    )
    print()


def plot_prediction_vs_truth(
    predictions,
    targets,
    *,
    tank_radius,
    tank_half_height,
    output_dir,
    dpi,
):
    """Plot x, y, z, R, and Wall predictions against truth."""
    quantities = calculate_vertex_quantities(
        predictions,
        targets,
        tank_radius=tank_radius,
        tank_half_height=tank_half_height,
    )

    true_values = [
        targets[:, 0],
        targets[:, 1],
        targets[:, 2],
        quantities["radius_true"],
        quantities["wall_true"],
    ]

    predicted_values = [
        predictions[:, 0],
        predictions[:, 1],
        predictions[:, 2],
        quantities["radius_predicted"],
        quantities["wall_predicted"],
    ]

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(28, 5),
    )

    for ax, truth, prediction, label in zip(
        axes,
        true_values,
        predicted_values,
        COORDINATE_LABELS,
        strict=True,
    ):
        ax.scatter(
            truth,
            prediction,
            s=5,
            alpha=0.25,
            rasterized=True,
        )

        low = min(
            np.min(truth),
            np.min(prediction),
        )
        high = max(
            np.max(truth),
            np.max(prediction),
        )

        ax.plot(
            [low, high],
            [low, high],
            color="black",
            linestyle="--",
            linewidth=1.5,
        )

        ax.set(
            xlabel=f"True {label} [{UNIT}]",
            ylabel=f"Predicted {label} [{UNIT}]",
            title=f"{label} prediction",
        )
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Vertex regression: prediction vs truth",
        fontsize=16,
    )
    fig.tight_layout()

    save(
        fig,
        output_path(
            output_dir,
            Path(
                "vertex_prediction_vs_truth_"
                "cylindrical_cm.png"
            ),
        ),
        dpi,
    )


def plot_residuals(
    predictions,
    targets,
    *,
    tank_radius,
    tank_half_height,
    histogram_bins,
    output_dir,
    dpi,
):
    """Plot x, y, z, R, Wall residuals and 3D error."""
    quantities = calculate_vertex_quantities(
        predictions,
        targets,
        tank_radius=tank_radius,
        tank_half_height=tank_half_height,
    )

    residuals = quantities["residuals"]
    error_3d = quantities["error_3d"]

    residual_values = [
        residuals[:, 0],
        residuals[:, 1],
        residuals[:, 2],
        quantities["radius_residual"],
        quantities["wall_residual"],
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 10),
    )

    for ax, residual, label in zip(
        axes.flat[:5],
        residual_values,
        COORDINATE_LABELS,
        strict=True,
    ):
        bias = np.mean(residual)
        resolution = calculate_resolution(
            residual
        )

        ax.hist(
            residual,
            bins=histogram_bins,
            histtype="step",
            linewidth=1.5,
        )
        ax.axvline(
            0,
            color="black",
            linestyle="--",
            linewidth=1.2,
        )
        ax.axvline(
            bias,
            color="red",
            linewidth=1.5,
            label=f"Bias = {bias:.2f}",
        )
        ax.axvline(
            resolution,
            color="purple",
            linestyle=":",
            linewidth=1.5,
        )
        ax.axvline(
            -resolution,
            color="purple",
            linestyle=":",
            linewidth=1.5,
            label=(
                "68% resolution = "
                f"{resolution:.2f}"
            ),
        )

        ax.set(
            xlabel=(
                f"Predicted {label} - "
                f"true {label} [{UNIT}]"
            ),
            ylabel="Events",
            title=f"{label} residual",
        )
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    ax_error = axes.flat[5]

    median = np.median(error_3d)
    quantile_68 = np.quantile(
        error_3d,
        0.68,
    )

    ax_error.hist(
        error_3d,
        bins=histogram_bins,
        histtype="step",
        linewidth=1.5,
        color="tab:green",
    )
    ax_error.axvline(
        median,
        color="red",
        linestyle="--",
        label=f"Median = {median:.2f}",
    )
    ax_error.axvline(
        quantile_68,
        color="purple",
        linestyle=":",
        label=f"68% quantile = {quantile_68:.2f}",
    )
    ax_error.set(
        xlabel=f"3D position error [{UNIT}]",
        ylabel="Events",
        title="3D position error",
    )
    ax_error.grid(alpha=0.3)
    ax_error.legend(fontsize=8)

    fig.suptitle(
        "Vertex regression residuals",
        fontsize=16,
    )
    fig.tight_layout()

    save(
        fig,
        output_path(
            output_dir,
            Path(
                "vertex_residuals_cylindrical_cm.png"
            ),
        ),
        dpi,
    )


def plot_cylindrical_profiles(
    predictions,
    targets,
    *,
    tank_radius,
    tank_half_height,
    profile_bins,
    output_dir,
    dpi,
):
    """Plot vertex performance versus true R, z, and Wall."""
    quantities = calculate_vertex_quantities(
        predictions,
        targets,
        tank_radius=tank_radius,
        tank_half_height=tank_half_height,
    )

    radius_true = quantities["radius_true"]
    z_true = targets[:, 2]
    wall_true = quantities["wall_true"]

    radius_residual = quantities[
        "radius_residual"
    ]
    z_residual = (
        predictions[:, 2]
        - targets[:, 2]
    )
    error_3d = quantities["error_3d"]

    configurations = [
        (
            radius_true,
            radius_residual,
            "True R [cm]",
            "R residual",
        ),
        (
            z_true,
            z_residual,
            "True z [cm]",
            "z residual",
        ),
        (
            wall_true,
            error_3d,
            "True Wall [cm]",
            "3D position error",
        ),
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 9),
    )

    for column, configuration in enumerate(
        configurations
    ):
        (
            x_values,
            residual,
            xlabel,
            title,
        ) = configuration

        (
            centres,
            biases,
            resolutions,
            counts,
        ) = calculate_profile(
            x_values,
            residual,
            n_bins=profile_bins,
        )

        valid = counts > 0

        axes[0, column].plot(
            centres[valid],
            biases[valid],
            "o-",
        )
        axes[0, column].axhline(
            0,
            color="black",
            linestyle="--",
            linewidth=1.2,
        )
        axes[0, column].set(
            xlabel=xlabel,
            ylabel=f"Bias [{UNIT}]",
            title=f"{title} bias",
        )
        axes[0, column].grid(alpha=0.3)

        axes[1, column].plot(
            centres[valid],
            resolutions[valid],
            "o-",
            color="tab:orange",
        )
        axes[1, column].set(
            xlabel=xlabel,
            ylabel=f"68% resolution [{UNIT}]",
            title=f"{title} resolution",
        )
        axes[1, column].grid(alpha=0.3)

    fig.suptitle(
        "Vertex regression in cylindrical geometry",
        fontsize=16,
    )
    fig.tight_layout()

    save(
        fig,
        output_path(
            output_dir,
            Path("vertex_cylindrical_profiles_cm.png"),
        ),
        dpi,
    )


def build_parser():
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot vertex-regression performance in "
            "Cartesian and cylindrical coordinates."
        )
    )

    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs/preds.npy"),
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path("outputs/targets.npy"),
    )
    parser.add_argument(
        "--indices",
        type=Path,
        default=Path("outputs/indices.npy"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/vertex"),
    )

    parser.add_argument(
        "--minimum",
        nargs=3,
        type=float,
        default=[
            -1690.0,
            -1690.0,
            -1810.0,
        ],
        metavar=(
            "X_MIN",
            "Y_MIN",
            "Z_MIN",
        ),
    )
    parser.add_argument(
        "--maximum",
        nargs=3,
        type=float,
        default=[
            1690.0,
            1690.0,
            1810.0,
        ],
        metavar=(
            "X_MAX",
            "Y_MAX",
            "Z_MAX",
        ),
    )

    parser.add_argument(
        "--tank-radius",
        type=float,
        default=1690.0,
    )
    parser.add_argument(
        "--tank-half-height",
        type=float,
        default=1810.0,
    )
    parser.add_argument(
        "--histogram-bins",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--profile-bins",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--already-cm",
        action="store_true",
        help=(
            "Treat predictions and targets as coordinates "
            "already expressed in centimetres."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
    )

    return parser


def main():
    """Load vertex outputs and generate all vertex plots."""
    args = build_parser().parse_args()

    if args.histogram_bins <= 0:
        raise ValueError(
            "--histogram-bins must be positive."
        )

    if args.profile_bins <= 0:
        raise ValueError(
            "--profile-bins must be positive."
        )

    if args.tank_radius <= 0:
        raise ValueError(
            "--tank-radius must be positive."
        )

    if args.tank_half_height <= 0:
        raise ValueError(
            "--tank-half-height must be positive."
        )

    minimum = np.asarray(
        args.minimum,
        dtype=float,
    )
    maximum = np.asarray(
        args.maximum,
        dtype=float,
    )

    if np.any(maximum <= minimum):
        raise ValueError(
            "Each value in --maximum must be greater "
            "than the corresponding value in --minimum."
        )

    outputs = load_outputs(
        args.run_dir,
        args.predictions,
        args.targets,
        args.indices,
    )

    predictions_raw = xyz(
        outputs.predictions
    )
    targets_raw = xyz(
        outputs.targets
    )

    if predictions_raw.shape != targets_raw.shape:
        raise ValueError(
            "Prediction and target shapes do not match: "
            f"{predictions_raw.shape} and "
            f"{targets_raw.shape}."
        )

    valid = (
        np.all(
            np.isfinite(predictions_raw),
            axis=1,
        )
        & np.all(
            np.isfinite(targets_raw),
            axis=1,
        )
    )

    if np.any(~valid):
        n_removed = int(
            np.sum(~valid)
        )

        print(
            f"Warning: removing {n_removed} events "
            "with non-finite values."
        )

        predictions_raw = predictions_raw[valid]
        targets_raw = targets_raw[valid]

    if len(targets_raw) == 0:
        raise ValueError(
            "No valid events remain."
        )

    if args.already_cm:
        predictions = predictions_raw
        targets = targets_raw
    else:
        predictions = inverse_minmax(
            predictions_raw,
            minimum,
            maximum,
        )
        targets = inverse_minmax(
            targets_raw,
            minimum,
            maximum,
        )

    print_statistics(
        predictions,
        targets,
        tank_radius=args.tank_radius,
        tank_half_height=args.tank_half_height,
    )

    plot_prediction_vs_truth(
        predictions,
        targets,
        tank_radius=args.tank_radius,
        tank_half_height=args.tank_half_height,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )

    plot_residuals(
        predictions,
        targets,
        tank_radius=args.tank_radius,
        tank_half_height=args.tank_half_height,
        histogram_bins=args.histogram_bins,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )

    plot_cylindrical_profiles(
        predictions,
        targets,
        tank_radius=args.tank_radius,
        tank_half_height=args.tank_half_height,
        profile_bins=args.profile_bins,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()