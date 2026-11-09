import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.geometry import opening_angle, unit
from analysis.io import load_outputs
from analysis.paths import output_path
from analysis.plotting import save
from analysis.regression import inverse_minmax, xyz

COMPONENT_LABELS = ["dx", "dy", "dz"]
ANGLE_UNIT = "deg"
EPSILON = 1.0e-12


def calculate_resolution(residuals):
    """Return the 68th percentile of the absolute residual."""
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]

    if residuals.size == 0:
        return np.nan

    return float(np.quantile(np.abs(residuals), 0.68))


def valid_direction_mask(vectors):
    """Return a mask selecting finite, non-zero direction vectors."""
    vectors = xyz(vectors)
    norms = np.linalg.norm(vectors, axis=1)

    return np.all(np.isfinite(vectors), axis=1) & np.isfinite(norms) & (norms > EPSILON)


def calculate_opening_angles(predictions, targets):
    """Calculate opening angles for valid direction-vector pairs."""
    prediction_valid = valid_direction_mask(predictions)
    target_valid = valid_direction_mask(targets)
    valid = prediction_valid & target_valid

    if np.any(~valid):
        n_invalid = int(np.sum(~valid))
        print(
            "Warning: excluding "
            f"{n_invalid} events from angular calculations "
            "because of non-finite or zero-length vectors."
        )

    if not np.any(valid):
        return np.array([], dtype=float), valid

    angles = opening_angle(
        targets[valid],
        predictions[valid],
    )

    return angles, valid


def calculate_direction_angles(vectors):
    """Return zenith and azimuth angles in degrees."""
    vectors = xyz(vectors)
    valid = valid_direction_mask(vectors)

    zenith = np.full(len(vectors), np.nan)
    azimuth = np.full(len(vectors), np.nan)

    if np.any(valid):
        unit_vectors = unit(vectors[valid])

        zenith[valid] = np.degrees(
            np.arccos(
                np.clip(
                    unit_vectors[:, 2],
                    -1.0,
                    1.0,
                )
            )
        )
        azimuth[valid] = np.degrees(
            np.arctan2(
                unit_vectors[:, 1],
                unit_vectors[:, 0],
            )
        )

    return zenith, azimuth, valid


def wrapped_angle_difference(predicted, target):
    """Return predicted minus target angle wrapped to [-180, 180)."""
    return (predicted - target + 180.0) % 360.0 - 180.0


def print_statistics(predictions, targets):
    """Print component, vector-norm, and angular statistics."""
    residuals = predictions - targets
    prediction_norms = np.linalg.norm(predictions, axis=1)
    target_norms = np.linalg.norm(targets, axis=1)

    angles, angle_valid = calculate_opening_angles(
        predictions,
        targets,
    )

    print()
    print("========== Direction regression statistics ==========")
    print(f"Number of events: {len(targets):,}")

    print()
    print("Component residuals:")

    for index, label in enumerate(COMPONENT_LABELS):
        values = residuals[:, index]

        print(f"  {label}:")
        print(f"    Bias: {np.mean(values):.6g}")
        print(f"    Median residual: {np.median(values):.6g}")
        print(f"    68% resolution: {calculate_resolution(values):.6g}")
        print(f"    MAE: {np.mean(np.abs(values)):.6g}")
        print(f"    RMSE: {np.sqrt(np.mean(values**2)):.6g}")

    print()
    print("Vector norms:")
    print(
        "  Prediction norm: "
        f"mean={np.mean(prediction_norms):.6g}, "
        f"median={np.median(prediction_norms):.6g}, "
        f"min={np.min(prediction_norms):.6g}, "
        f"max={np.max(prediction_norms):.6g}"
    )
    print(
        "  Target norm:     "
        f"mean={np.mean(target_norms):.6g}, "
        f"median={np.median(target_norms):.6g}, "
        f"min={np.min(target_norms):.6g}, "
        f"max={np.max(target_norms):.6g}"
    )

    print()
    print("Opening angle:")
    print(f"  Valid events: {np.sum(angle_valid):,} / {len(targets):,}")

    if angles.size > 0:
        print(f"  Mean: {np.mean(angles):.6g} {ANGLE_UNIT}")
        print(f"  Median: {np.median(angles):.6g} {ANGLE_UNIT}")
        print(f"  68% angular resolution: {np.quantile(angles, 0.68):.6g} {ANGLE_UNIT}")
        print(f"  90% quantile: {np.quantile(angles, 0.90):.6g} {ANGLE_UNIT}")
    else:
        print("  No valid direction pairs are available.")

    print("=====================================================")
    print()


def plot_prediction_vs_truth(
    predictions,
    targets,
    *,
    output_dir,
    dpi,
):
    """Plot direction-component predictions against truth."""
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5),
    )

    for index, label in enumerate(COMPONENT_LABELS):
        truth = targets[:, index]
        prediction = predictions[:, index]

        axes[index].scatter(
            truth,
            prediction,
            s=5,
            alpha=0.25,
            rasterized=True,
        )

        value_min = min(
            np.min(truth),
            np.min(prediction),
        )
        value_max = max(
            np.max(truth),
            np.max(prediction),
        )

        axes[index].plot(
            [value_min, value_max],
            [value_min, value_max],
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="Perfect prediction",
        )
        axes[index].set(
            xlabel=f"True {label}",
            ylabel=f"Predicted {label}",
            title=f"{label} prediction",
        )
        axes[index].grid(alpha=0.3)
        axes[index].legend()

    fig.suptitle(
        "Direction regression: prediction vs truth",
        fontsize=16,
    )
    fig.tight_layout()

    save(
        fig,
        output_path(
            output_dir,
            Path("direction_prediction_vs_truth.png"),
        ),
        dpi,
    )


def plot_residuals(
    predictions,
    targets,
    *,
    histogram_bins,
    output_dir,
    dpi,
):
    """Plot component residuals and opening-angle errors."""
    residuals = predictions - targets
    angles, _ = calculate_opening_angles(
        predictions,
        targets,
    )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(22, 5),
    )

    for index, label in enumerate(COMPONENT_LABELS):
        component_residuals = residuals[:, index]
        bias = np.mean(component_residuals)
        resolution = calculate_resolution(component_residuals)

        axes[index].hist(
            component_residuals,
            bins=histogram_bins,
            histtype="step",
            linewidth=1.5,
            color="tab:blue",
        )
        axes[index].axvline(
            0.0,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="Zero residual",
        )
        axes[index].axvline(
            bias,
            color="red",
            linewidth=1.8,
            label=f"Bias = {bias:.4f}",
        )
        axes[index].axvline(
            resolution,
            color="purple",
            linestyle=":",
            linewidth=1.8,
            label=f"68% resolution = {resolution:.4f}",
        )
        axes[index].axvline(
            -resolution,
            color="purple",
            linestyle=":",
            linewidth=1.8,
        )
        axes[index].set(
            xlabel=f"Predicted {label} - true {label}",
            ylabel="Events",
            title=(
                f"{label} residual\nBias = {bias:.4f}\nResolution = {resolution:.4f}"
            ),
        )
        axes[index].grid(alpha=0.3)
        axes[index].legend(fontsize=8)

    angle_ax = axes[3]

    if angles.size > 0:
        median = np.median(angles)
        quantile_68 = np.quantile(angles, 0.68)

        angle_ax.hist(
            angles,
            bins=histogram_bins,
            histtype="step",
            linewidth=1.5,
            color="tab:green",
        )
        angle_ax.axvline(
            median,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Median = {median:.2f} deg",
        )
        angle_ax.axvline(
            quantile_68,
            color="purple",
            linestyle=":",
            linewidth=1.8,
            label=f"68% quantile = {quantile_68:.2f} deg",
        )
        angle_ax.set_title(
            "Opening angle\n"
            f"Median = {median:.2f} deg\n"
            f"68% resolution = {quantile_68:.2f} deg"
        )
    else:
        angle_ax.text(
            0.5,
            0.5,
            "No valid direction vectors",
            ha="center",
            va="center",
            transform=angle_ax.transAxes,
        )
        angle_ax.set_title("Opening angle")

    angle_ax.set(
        xlabel="Opening angle [deg]",
        ylabel="Events",
    )
    angle_ax.grid(alpha=0.3)

    if angles.size > 0:
        angle_ax.legend(fontsize=8)

    fig.suptitle(
        "Direction regression residuals",
        fontsize=16,
    )
    fig.tight_layout()

    save(
        fig,
        output_path(
            output_dir,
            Path("direction_residuals.png"),
        ),
        dpi,
    )


def plot_angle_results(
    predictions,
    targets,
    *,
    histogram_bins,
    output_dir,
    dpi,
):
    """Plot zenith and azimuth predictions and residuals."""
    pred_zenith, pred_azimuth, pred_valid = calculate_direction_angles(predictions)
    true_zenith, true_azimuth, true_valid = calculate_direction_angles(targets)

    valid = pred_valid & true_valid

    if not np.any(valid):
        raise ValueError("No valid direction vectors are available for angle plots.")

    pred_zenith = pred_zenith[valid]
    pred_azimuth = pred_azimuth[valid]
    true_zenith = true_zenith[valid]
    true_azimuth = true_azimuth[valid]

    zenith_residual = pred_zenith - true_zenith
    azimuth_residual = wrapped_angle_difference(
        pred_azimuth,
        true_azimuth,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 11),
    )

    axes[0, 0].scatter(
        true_zenith,
        pred_zenith,
        s=5,
        alpha=0.25,
        rasterized=True,
    )
    axes[0, 0].plot(
        [0.0, 180.0],
        [0.0, 180.0],
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Perfect prediction",
    )
    axes[0, 0].set(
        xlim=(0.0, 180.0),
        ylim=(0.0, 180.0),
        xlabel="True zenith angle [deg]",
        ylabel="Predicted zenith angle [deg]",
        title="Zenith angle: prediction vs truth",
    )
    axes[0, 0].set_aspect(
        "equal",
        adjustable="box",
    )
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].scatter(
        true_azimuth,
        pred_azimuth,
        s=5,
        alpha=0.25,
        rasterized=True,
    )
    axes[0, 1].plot(
        [-180.0, 180.0],
        [-180.0, 180.0],
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Perfect prediction",
    )
    axes[0, 1].set(
        xlim=(-180.0, 180.0),
        ylim=(-180.0, 180.0),
        xlabel="True azimuth angle [deg]",
        ylabel="Predicted azimuth angle [deg]",
        title="Azimuth angle: prediction vs truth",
    )
    axes[0, 1].set_aspect(
        "equal",
        adjustable="box",
    )
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    angle_residuals = [
        (
            zenith_residual,
            "Predicted zenith - true zenith [deg]",
            "Zenith-angle residual",
            None,
        ),
        (
            azimuth_residual,
            "Wrapped predicted azimuth - true azimuth [deg]",
            "Azimuth-angle residual",
            (-180.0, 180.0),
        ),
    ]

    for ax, configuration in zip(
        axes[1],
        angle_residuals,
        strict=True,
    ):
        (
            residual,
            xlabel,
            title,
            histogram_range,
        ) = configuration

        bias = np.mean(residual)
        resolution = calculate_resolution(residual)

        ax.hist(
            residual,
            bins=histogram_bins,
            range=histogram_range,
            histtype="step",
            linewidth=1.8,
            color="tab:green",
        )
        ax.axvline(
            0.0,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="Zero residual",
        )
        ax.axvline(
            bias,
            color="red",
            linewidth=1.5,
            label=f"Bias = {bias:.2f} deg",
        )
        ax.axvline(
            resolution,
            color="purple",
            linestyle=":",
            linewidth=1.5,
            label=(f"68% resolution = {resolution:.2f} deg"),
        )
        ax.axvline(
            -resolution,
            color="purple",
            linestyle=":",
            linewidth=1.5,
        )
        ax.set(
            xlabel=xlabel,
            ylabel="Events",
            title=(
                f"{title}\nBias = {bias:.2f} deg, resolution = {resolution:.2f} deg"
            ),
        )
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    axes[1, 1].set_xlim(
        -180.0,
        180.0,
    )

    fig.suptitle(
        "Direction regression: zenith and azimuth angles",
        fontsize=16,
    )
    fig.tight_layout()

    save(
        fig,
        output_path(
            output_dir,
            Path("direction_zenith_azimuth_results.png"),
        ),
        dpi,
    )


def calculate_binned_profiles(
    predictions,
    targets,
    *,
    n_bins,
):
    """Calculate opening-angle performance in true-dz bins."""
    prediction_valid = valid_direction_mask(predictions)
    target_valid = valid_direction_mask(targets)
    valid = prediction_valid & target_valid

    if not np.any(valid):
        raise ValueError("No valid vectors are available for profiles.")

    prediction_units = unit(predictions[valid])
    target_units = unit(targets[valid])

    true_dz = target_units[:, 2]
    angles = opening_angle(
        target_units,
        prediction_units,
    )

    edges = np.linspace(
        -1.0,
        1.0,
        n_bins + 1,
    )
    centres = 0.5 * (edges[:-1] + edges[1:])

    means = np.full(
        n_bins,
        np.nan,
    )
    medians = np.full(
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
            in_bin = (true_dz >= edges[index]) & (true_dz <= edges[index + 1])
        else:
            in_bin = (true_dz >= edges[index]) & (true_dz < edges[index + 1])

        values = angles[in_bin]
        counts[index] = len(values)

        if values.size > 0:
            means[index] = np.mean(values)
            medians[index] = np.median(values)
            resolutions[index] = np.quantile(
                values,
                0.68,
            )

    return {
        "centres": centres,
        "means": means,
        "medians": medians,
        "resolutions": resolutions,
        "counts": counts,
    }


def plot_binned_profiles(
    predictions,
    targets,
    *,
    profile_bins,
    output_dir,
    dpi,
):
    """Plot opening-angle performance versus true dz."""
    profile = calculate_binned_profiles(
        predictions,
        targets,
        n_bins=profile_bins,
    )

    centres = profile["centres"]
    counts = profile["counts"]
    valid = counts > 0

    quantities = [
        (
            profile["means"],
            "Mean opening angle",
            "tab:blue",
        ),
        (
            profile["medians"],
            "Median opening angle",
            "tab:green",
        ),
        (
            profile["resolutions"],
            "68% angular resolution",
            "tab:red",
        ),
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5),
        sharex=True,
    )

    for ax, configuration in zip(
        axes,
        quantities,
        strict=True,
    ):
        values, title, color = configuration

        ax.plot(
            centres[valid],
            values[valid],
            "o-",
            color=color,
            linewidth=1.8,
            markersize=5,
        )
        ax.set(
            xlabel="True dz",
            ylabel="Opening angle [deg]",
            title=title,
        )
        ax.grid(alpha=0.3)

    for x_value, resolution, count in zip(
        centres[valid],
        profile["resolutions"][valid],
        counts[valid],
        strict=True,
    ):
        axes[2].annotate(
            f"N={count}",
            (x_value, resolution),
            fontsize=7,
            alpha=0.6,
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
        )

    fig.suptitle(
        "Direction regression performance vs true dz",
        fontsize=16,
    )
    fig.tight_layout()

    save(
        fig,
        output_path(
            output_dir,
            Path("direction_angular_profiles_vs_true_dz.png"),
        ),
        dpi,
    )


def build_parser():
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Plot direction-regression performance."
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
        default=Path("plots/direction"),
    )
    parser.add_argument(
        "--inverse-component-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=("Convert direction components from [0, 1] to [-1, 1]. Default: enabled."),
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


def main():
    """Load direction outputs and generate all direction plots."""
    args = build_parser().parse_args()

    if args.histogram_bins <= 0:
        raise ValueError("--histogram-bins must be positive.")

    if args.profile_bins <= 0:
        raise ValueError("--profile-bins must be positive.")

    outputs = load_outputs(
        args.run_dir,
        args.predictions,
        args.targets,
        args.indices,
    )

    predictions = xyz(outputs.predictions)
    targets = xyz(outputs.targets)

    if predictions.shape != targets.shape:
        raise ValueError(
            "Prediction and target shapes do not match: "
            f"{predictions.shape} and {targets.shape}."
        )

    finite = np.all(
        np.isfinite(predictions),
        axis=1,
    ) & np.all(
        np.isfinite(targets),
        axis=1,
    )

    if np.any(~finite):
        n_removed = int(np.sum(~finite))
        print(f"Warning: removing {n_removed} events containing non-finite values.")

        predictions = predictions[finite]
        targets = targets[finite]

    if len(targets) == 0:
        raise ValueError("No valid events remain.")

    if args.inverse_component_norm:
        predictions = inverse_minmax(
            predictions,
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
        )
        targets = inverse_minmax(
            targets,
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
        )

        print("Applied component-wise [0, 1] to [-1, 1] conversion.")
    else:
        print("No inverse component normalization was applied.")

    print()
    print("Component ranges:")

    for index, label in enumerate(COMPONENT_LABELS):
        print(
            f"  Prediction {label}: "
            f"{np.min(predictions[:, index]):.6g} to "
            f"{np.max(predictions[:, index]):.6g}"
        )
        print(
            f"  Target {label}:     "
            f"{np.min(targets[:, index]):.6g} to "
            f"{np.max(targets[:, index]):.6g}"
        )

    print_statistics(
        predictions,
        targets,
    )

    plot_prediction_vs_truth(
        predictions,
        targets,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )

    plot_residuals(
        predictions,
        targets,
        histogram_bins=args.histogram_bins,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )

    plot_angle_results(
        predictions,
        targets,
        histogram_bins=args.histogram_bins,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )

    plot_binned_profiles(
        predictions,
        targets,
        profile_bins=args.profile_bins,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
