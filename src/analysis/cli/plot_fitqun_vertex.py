#!/usr/bin/env python3
"""Evaluate fiTQun electron-hypothesis vertex reconstruction.

Selection:
  - common events stored in common_multi_fitqun.hy
  - multi/labels == 1 (electron)
  - number of hits > --min-hits (default: 200)
  - true Wall > --min-true-wall (default: 200 cm)
  - reconstructed fiTQun vertex is inside the tank
  - fiTQun evis > --min-evis (default: 30)

Truth vertex:
  multi/positions

Reconstructed vertex:
  fitqun/e_1rpos

Visible energy:
  fitqun/evis by default

Example
-------
python scripts/plot_fitqun_vertex.py \
    --input-file /path/to/common_multi_fitqun.hy \
    --output-dir plots/fitqun_vertex_electron

If the evis dataset has a different name:

python scripts/plot_fitqun_vertex.py \
    --input-file /path/to/common_multi_fitqun.hy \
    --evis-dataset fitqun/e_1revis \
    --output-dir plots/fitqun_vertex_electron
"""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from analysis.geometry import cylindrical_radius, wall
from analysis.metrics import residual_profile, resolution68
from analysis.paths import output_path
from analysis.plotting import save
from analysis.regression import xyz

UNIT = "cm"
ELECTRON_LABEL = 1

COORDINATE_LABELS = [
    "x",
    "y",
    "z",
    "R",
    "Wall",
]


def compute_nhits(event_hits_index, total_hits, n_events):
    """Compute per-event hit counts for N or N+1 index conventions."""
    event_hits_index = np.asarray(
        event_hits_index,
        dtype=np.int64,
    ).reshape(-1)

    if len(event_hits_index) == n_events:
        boundaries = np.append(event_hits_index, total_hits)
    elif len(event_hits_index) == n_events + 1:
        boundaries = event_hits_index
    else:
        raise ValueError(
            "Unexpected event_hits_index length: "
            f"{len(event_hits_index)}. "
            f"Expected {n_events} or {n_events + 1}."
        )

    if len(boundaries) == 0 or boundaries[0] != 0:
        raise ValueError("event_hits_index must start at 0.")

    if boundaries[-1] != total_hits:
        raise ValueError(
            "The final hit boundary does not match the hit-array length: "
            f"{boundaries[-1]} != {total_hits}."
        )

    nhits = np.diff(boundaries)

    if np.any(nhits < 0):
        raise ValueError("event_hits_index is not monotonically increasing.")

    return nhits


def load_scalar_per_event(dataset, dataset_name, n_events):
    """Load a dataset expected to contain one scalar value per event."""
    values = np.asarray(dataset[:])

    values = np.squeeze(values)

    if values.ndim != 1:
        raise ValueError(
            f"Unexpected {dataset_name} shape after squeeze: "
            f"{values.shape}. Expected one scalar per event."
        )

    if len(values) != n_events:
        raise ValueError(
            f"Unexpected {dataset_name} length: {len(values)}. Expected {n_events}."
        )

    return values


def load_selected_vertices(
    input_file,
    min_hits,
    min_true_wall,
    min_evis,
    evis_dataset,
    *,
    tank_radius,
    tank_half_height,
):
    """Load and select electron truth and fiTQun vertices."""
    required = (
        "multi/labels",
        "multi/positions",
        "multi/event_hits_index",
        "multi/hit_charge",
        "fitqun/e_1rpos",
        evis_dataset,
    )

    with h5py.File(input_file, "r") as h5file:
        missing = [name for name in required if name not in h5file]

        if missing:
            raise KeyError("Missing required datasets: " + ", ".join(missing))

        labels = np.asarray(h5file["multi/labels"][:]).reshape(-1)

        n_events = len(labels)

        truth = xyz(np.asarray(h5file["multi/positions"][:]))
        prediction = xyz(np.asarray(h5file["fitqun/e_1rpos"][:]))

        evis = load_scalar_per_event(
            h5file[evis_dataset],
            evis_dataset,
            n_events,
        )

        event_hits_index = np.asarray(h5file["multi/event_hits_index"][:])
        total_hits = len(h5file["multi/hit_charge"])

    if truth.shape != (n_events, 3):
        raise ValueError(
            "Unexpected multi/positions shape after xyz(): "
            f"{truth.shape}. Expected ({n_events}, 3)."
        )

    if prediction.shape != (n_events, 3):
        raise ValueError(
            "Unexpected fitqun/e_1rpos shape after xyz(): "
            f"{prediction.shape}. Expected ({n_events}, 3)."
        )

    nhits = compute_nhits(
        event_hits_index,
        total_hits,
        n_events,
    )

    truth_radius = cylindrical_radius(truth)
    prediction_radius = cylindrical_radius(prediction)

    truth_wall = wall(
        truth,
        radius=tank_radius,
        half_height=tank_half_height,
    )

    electron = labels == ELECTRON_LABEL
    hit_selection = nhits > min_hits

    finite_truth = np.all(
        np.isfinite(truth),
        axis=1,
    )
    finite_prediction = np.all(
        np.isfinite(prediction),
        axis=1,
    )
    finite_evis = np.isfinite(evis)

    finite = finite_truth & finite_prediction & finite_evis

    truth_inside = (truth_radius <= tank_radius) & (
        np.abs(truth[:, 2]) <= tank_half_height
    )

    truth_fiducial = truth_inside & (truth_wall > min_true_wall)

    prediction_inside = (prediction_radius <= tank_radius) & (
        np.abs(prediction[:, 2]) <= tank_half_height
    )

    evis_selection = evis > min_evis

    after_electron = electron
    after_hit_cut = after_electron & hit_selection
    after_finite = after_hit_cut & finite
    after_true_wall = after_finite & truth_fiducial
    after_prediction_inside = after_true_wall & prediction_inside
    selected = after_prediction_inside & evis_selection

    n_after_electron = np.count_nonzero(after_electron)
    n_after_hit_cut = np.count_nonzero(after_hit_cut)
    n_after_finite = np.count_nonzero(after_finite)
    n_after_true_wall = np.count_nonzero(after_true_wall)
    n_after_prediction_inside = np.count_nonzero(after_prediction_inside)
    n_selected = np.count_nonzero(selected)

    n_nonfinite_removed = n_after_hit_cut - n_after_finite
    n_true_wall_removed = n_after_finite - n_after_true_wall
    n_outside_removed = n_after_true_wall - n_after_prediction_inside
    n_evis_removed = n_after_prediction_inside - n_selected

    print()
    print("========== fiTQun vertex selection ==========")
    print(f"Input file: {input_file}")
    print(f"evis dataset: {evis_dataset}")
    print()
    print(f"Input events:                       {n_events:,}")
    print(f"Electron events (labels == 1):      {n_after_electron:,}")
    print(f"After nhits > {min_hits}:{n_after_hit_cut:>20,}")
    print(f"After finite-value selection:       {n_after_finite:,}")
    print(f"After true Wall > {min_true_wall:g} cm:{n_after_true_wall:>13,}")
    print(f"After fiTQun vertex inside tank:    {n_after_prediction_inside:,}")
    print(f"After fiTQun evis > {min_evis:g}:{n_selected:>17,}")
    print()
    print("Removed at each selection step:")
    print(f"  Non-finite values:                {n_nonfinite_removed:,}")
    print(f"  True vertex outside fiducial cut: {n_true_wall_removed:,}")
    print(f"  fiTQun vertex outside tank:       {n_outside_removed:,}")
    print(f"  fiTQun evis cut:                  {n_evis_removed:,}")
    print()
    print(f"Selected events:                    {n_selected:,}")
    print("=============================================")
    print()

    if not np.any(selected):
        raise ValueError(
            "No events remain after selection. Check the dataset names and cut values."
        )

    return (
        prediction[selected],
        truth[selected],
        nhits[selected],
        evis[selected],
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

    radius_predicted = cylindrical_radius(predictions)
    radius_true = cylindrical_radius(targets)

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
        "radius_residual": (radius_predicted - radius_true),
        "wall_predicted": wall_predicted,
        "wall_true": wall_true,
        "wall_residual": (wall_predicted - wall_true),
    }


def print_statistics(
    predictions,
    targets,
    evis,
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
    radius_residual = quantities["radius_residual"]
    wall_residual = quantities["wall_residual"]

    print("========== fiTQun vertex statistics ==========")
    print(f"Number of events: {len(targets):,}")
    print("Particle: electron (label 1)")
    print(f"Position unit: {UNIT}")
    print()

    print("Selected fiTQun evis:")
    print(f"  Minimum: {np.min(evis):.6g}")
    print(f"  Maximum: {np.max(evis):.6g}")
    print(f"  Mean:    {np.mean(evis):.6g}")
    print(f"  Median:  {np.median(evis):.6g}")
    print()

    print("Cartesian residuals:")

    for index, label in enumerate(("x", "y", "z")):
        values = residuals[:, index]

        print(
            f"  {label}: "
            f"bias={np.mean(values):.6g} {UNIT}, "
            f"resolution={resolution68(values):.6g} {UNIT}"
        )

    print()
    print("Cylindrical and Wall residuals:")
    print(
        "  R:    "
        f"bias={np.mean(radius_residual):.6g} {UNIT}, "
        f"resolution="
        f"{resolution68(radius_residual):.6g} {UNIT}"
    )
    print(
        "  Wall: "
        f"bias={np.mean(wall_residual):.6g} {UNIT}, "
        f"resolution="
        f"{resolution68(wall_residual):.6g} {UNIT}"
    )

    print()
    print("3D position error:")
    print(f"  Mean:         {np.mean(error_3d):.6g} {UNIT}")
    print(f"  Median:       {np.median(error_3d):.6g} {UNIT}")
    print(f"  68% quantile: {np.quantile(error_3d, 0.68):.6g} {UNIT}")
    print(f"  RMS:          {np.sqrt(np.mean(error_3d**2)):.6g} {UNIT}")
    print("==============================================")
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
    """Plot x, y, z, R, and Wall reconstruction against truth."""
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
            ylabel=f"fiTQun {label} [{UNIT}]",
            title=f"{label} reconstruction",
        )
        ax.grid(alpha=0.3)

    fig.suptitle(
        "fiTQun electron vertex: reconstruction vs truth",
        fontsize=16,
    )
    fig.tight_layout()

    save(
        fig,
        output_path(
            output_dir,
            Path("fitqun_vertex_prediction_vs_truth_cylindrical_cm.png"),
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
    """Plot Cartesian, R, Wall residuals and 3D error."""
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
        resolution = resolution68(residual)

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
            label=f"68% resolution = {resolution:.2f}",
        )

        ax.set(
            xlabel=(f"fiTQun {label} - true {label} [{UNIT}]"),
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
        "fiTQun electron vertex residuals",
        fontsize=16,
    )
    fig.tight_layout()

    save(
        fig,
        output_path(
            output_dir,
            Path("fitqun_vertex_residuals_cylindrical_cm.png"),
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

    configurations = [
        (
            quantities["radius_true"],
            quantities["radius_residual"],
            "True R [cm]",
            "R residual",
            "Bias [cm]",
            "68% resolution [cm]",
        ),
        (
            targets[:, 2],
            predictions[:, 2] - targets[:, 2],
            "True z [cm]",
            "z residual",
            "Bias [cm]",
            "68% resolution [cm]",
        ),
        (
            quantities["wall_true"],
            quantities["error_3d"],
            "True Wall [cm]",
            "3D position error",
            "Mean error [cm]",
            "68% spread [cm]",
        ),
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 9),
    )

    for column, configuration in enumerate(configurations):
        (
            x_values,
            profile_values,
            xlabel,
            title,
            bias_ylabel,
            resolution_ylabel,
        ) = configuration

        centres, biases, resolutions, counts = residual_profile(
            x_values,
            profile_values,
            n_bins=profile_bins,
        )

        valid = (
            (counts > 0)
            & np.isfinite(centres)
            & np.isfinite(biases)
            & np.isfinite(resolutions)
        )

        axes[0, column].plot(
            centres[valid],
            biases[valid],
            "o-",
        )

        if column < 2:
            axes[0, column].axhline(
                0,
                color="black",
                linestyle="--",
                linewidth=1.2,
            )

        axes[0, column].set(
            xlabel=xlabel,
            ylabel=bias_ylabel,
            title=f"{title} profile",
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
            ylabel=resolution_ylabel,
            title=f"{title} resolution",
        )
        axes[1, column].grid(alpha=0.3)

    fig.suptitle(
        "fiTQun electron vertex in cylindrical geometry",
        fontsize=16,
    )
    fig.tight_layout()

    save(
        fig,
        output_path(
            output_dir,
            Path("fitqun_vertex_cylindrical_profiles_cm.png"),
        ),
        dpi,
    )


def build_parser():
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate fiTQun electron-hypothesis vertex "
            "reconstruction using common_multi_fitqun.hy."
        )
    )

    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Input common_multi_fitqun.hy file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/fitqun_vertex_electron"),
    )
    parser.add_argument(
        "--min-hits",
        type=int,
        default=200,
        help=("Require nhits > this value (default: 200)"),
    )
    parser.add_argument(
        "--min-true-wall",
        type=float,
        default=-1e-10,
        help=("Require true Wall > this value in cm (default: -1e-10)"),
    )
    parser.add_argument(
        "--min-evis",
        type=float,
        default=30.0,
        help=("Require fiTQun evis > this value (default: 30)"),
    )
    parser.add_argument(
        "--evis-dataset",
        type=str,
        default="fitqun/evis",
        help=("HDF5 dataset containing fiTQun visible energy (default: fitqun/evis)"),
    )
    parser.add_argument(
        "--tank-radius",
        type=float,
        default=1690.0,
        help="Tank radius in cm (default: 1690)",
    )
    parser.add_argument(
        "--tank-half-height",
        type=float,
        default=1810.0,
        help=("Tank half-height in cm (default: 1810)"),
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
        "--dpi",
        type=int,
        default=200,
    )

    return parser


def validate_arguments(args):
    """Validate command-line arguments."""
    if not args.input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    if args.min_hits < 0:
        raise ValueError("--min-hits must be non-negative.")

    if not np.isfinite(args.min_true_wall):
        raise ValueError("--min-true-wall must be finite.")

    if args.min_true_wall < 0:
        raise ValueError("--min-true-wall must be non-negative.")

    maximum_wall = min(
        args.tank_radius,
        args.tank_half_height,
    )

    if args.min_true_wall >= maximum_wall:
        raise ValueError(
            "--min-true-wall must be smaller than "
            "both the tank radius and tank half-height."
        )

    if not np.isfinite(args.min_evis):
        raise ValueError("--min-evis must be finite.")

    if not args.evis_dataset.strip():
        raise ValueError("--evis-dataset must not be empty.")

    if args.histogram_bins <= 0:
        raise ValueError("--histogram-bins must be positive.")

    if args.profile_bins <= 0:
        raise ValueError("--profile-bins must be positive.")

    if args.tank_radius <= 0:
        raise ValueError("--tank-radius must be positive.")

    if args.tank_half_height <= 0:
        raise ValueError("--tank-half-height must be positive.")

    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")


def main():
    """Load selected events and generate all fiTQun vertex plots."""
    args = build_parser().parse_args()

    validate_arguments(args)

    predictions, targets, nhits, evis = load_selected_vertices(
        args.input_file,
        args.min_hits,
        args.min_true_wall,
        args.min_evis,
        args.evis_dataset,
        tank_radius=args.tank_radius,
        tank_half_height=args.tank_half_height,
    )

    print_statistics(
        predictions,
        targets,
        evis,
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

    print(f"Plots saved under: {args.output_dir}")
    print(f"Selected nhits range: {np.min(nhits):,} to {np.max(nhits):,}")


if __name__ == "__main__":
    main()
