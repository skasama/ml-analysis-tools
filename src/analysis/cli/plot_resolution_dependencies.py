import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.geometry import opening_angle, towall, wall
from analysis.io import assert_aligned, load_outputs
from analysis.paths import output_path
from analysis.plotting import save
from analysis.regression import inverse_minmax, scalar, xyz

EPSILON = 1.0e-12


def make_edges(
    values,
    width,
    *,
    lower=None,
    upper=None,
):
    """Create fixed-width bin edges."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]

    if finite.size == 0:
        raise ValueError("No finite values are available for binning.")

    if width <= 0:
        raise ValueError("Bin width must be positive.")

    start = width * np.floor(np.min(finite) / width) if lower is None else float(lower)

    stop = width * np.ceil(np.max(finite) / width) if upper is None else float(upper)

    if stop <= start:
        stop = start + width

    return np.arange(
        start,
        stop + 0.5 * width,
        width,
    )


def binned_quantile(
    x_values,
    errors,
    edges,
    *,
    quantile,
    minimum_count,
):
    """Calculate an absolute-error quantile in fixed bins."""
    x_values = np.asarray(
        x_values,
        dtype=float,
    )
    errors = np.asarray(
        errors,
        dtype=float,
    )

    if x_values.shape != errors.shape:
        raise ValueError(
            "x_values and errors must have the same shape: "
            f"{x_values.shape} and {errors.shape}."
        )

    centres = 0.5 * (edges[:-1] + edges[1:])
    quantiles = np.full(
        len(centres),
        np.nan,
    )
    counts = np.zeros(
        len(centres),
        dtype=int,
    )

    for index, (low, high) in enumerate(
        zip(
            edges[:-1],
            edges[1:],
            strict=True,
        )
    ):
        in_bin = np.isfinite(x_values) & np.isfinite(errors) & (x_values >= low)

        if index == len(centres) - 1:
            in_bin &= x_values <= high
        else:
            in_bin &= x_values < high

        counts[index] = np.count_nonzero(in_bin)

        if counts[index] >= minimum_count:
            quantiles[index] = np.quantile(
                np.abs(errors[in_bin]),
                quantile,
            )

    return centres, quantiles, counts


def collect_binned_values(
    x_values,
    values,
    edges,
    *,
    minimum_count,
):
    """Collect finite values in fixed bins."""
    x_values = np.asarray(
        x_values,
        dtype=float,
    )
    values = np.asarray(
        values,
        dtype=float,
    )

    if x_values.shape != values.shape:
        raise ValueError(
            "x_values and values must have the same shape: "
            f"{x_values.shape} and {values.shape}."
        )

    centres = 0.5 * (edges[:-1] + edges[1:])

    groups = []
    positions = []
    counts = []

    for index, (low, high) in enumerate(
        zip(
            edges[:-1],
            edges[1:],
            strict=True,
        )
    ):
        in_bin = np.isfinite(x_values) & np.isfinite(values) & (x_values >= low)

        if index == len(centres) - 1:
            in_bin &= x_values <= high
        else:
            in_bin &= x_values < high

        selected = values[in_bin]

        if selected.size >= minimum_count:
            groups.append(selected)
            positions.append(centres[index])
            counts.append(selected.size)

    return (
        groups,
        np.asarray(positions),
        np.asarray(counts, dtype=int),
    )


def pearson_correlation(x_values, y_values):
    """Return the Pearson correlation coefficient."""
    x_values = np.asarray(
        x_values,
        dtype=float,
    )
    y_values = np.asarray(
        y_values,
        dtype=float,
    )

    valid = np.isfinite(x_values) & np.isfinite(y_values)

    if np.count_nonzero(valid) < 2:
        return np.nan

    if np.std(x_values[valid]) <= EPSILON or np.std(y_values[valid]) <= EPSILON:
        return np.nan

    return float(
        np.corrcoef(
            x_values[valid],
            y_values[valid],
        )[0, 1]
    )


def calculate_towall(
    vertices,
    directions,
    *,
    radius,
    half_height,
):
    """Calculate ToWall while retaining invalid events as NaN."""
    vertices = xyz(vertices)
    directions = xyz(directions)

    direction_norms = np.linalg.norm(
        directions,
        axis=1,
    )

    valid = (
        np.all(np.isfinite(vertices), axis=1)
        & np.all(np.isfinite(directions), axis=1)
        & (direction_norms > EPSILON)
    )

    values = np.full(
        len(vertices),
        np.nan,
    )

    if np.any(valid):
        values[valid] = towall(
            vertices[valid],
            directions[valid],
            radius=radius,
            half_height=half_height,
        )

    return values


def calculate_opening_angles(
    predictions,
    targets,
):
    """Calculate opening angles while retaining invalid events as NaN."""
    predictions = xyz(predictions)
    targets = xyz(targets)

    prediction_norms = np.linalg.norm(
        predictions,
        axis=1,
    )
    target_norms = np.linalg.norm(
        targets,
        axis=1,
    )

    valid = (
        np.all(np.isfinite(predictions), axis=1)
        & np.all(np.isfinite(targets), axis=1)
        & (prediction_norms > EPSILON)
        & (target_norms > EPSILON)
    )

    values = np.full(
        len(targets),
        np.nan,
    )

    if np.any(valid):
        values[valid] = opening_angle(
            targets[valid],
            predictions[valid],
        )

    return values


def calculate_relative_energy_error(
    predictions,
    targets,
):
    """Calculate relative energy residuals in percent."""
    predictions = np.asarray(
        predictions,
        dtype=float,
    )
    targets = np.asarray(
        targets,
        dtype=float,
    )

    values = np.full(
        len(targets),
        np.nan,
    )

    valid = (
        np.isfinite(predictions) & np.isfinite(targets) & (np.abs(targets) > EPSILON)
    )

    values[valid] = 100.0 * (predictions[valid] - targets[valid]) / targets[valid]

    return values


def save_resolution_csv(
    rows,
    output_dir,
):
    """Save binned resolution values to CSV."""
    path = output_path(
        output_dir,
        Path("resolution_vs_wall_towall_energy.csv"),
    )

    if not rows:
        raise ValueError("No resolution rows are available for CSV output.")

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {path}")


def save_correlation_csv(
    variables,
    errors,
    output_dir,
):
    """Calculate and save Pearson correlations."""
    path = output_path(
        output_dir,
        Path("resolution_correlations.csv"),
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.writer(stream)

        writer.writerow(
            [
                "variable",
                "error_metric",
                "pearson_r",
                "valid_events",
            ]
        )

        for variable_name, x_values in variables:
            for error_name, error_values in errors:
                valid = np.isfinite(x_values) & np.isfinite(error_values)
                correlation = pearson_correlation(
                    x_values,
                    error_values,
                )

                writer.writerow(
                    [
                        variable_name,
                        error_name,
                        correlation,
                        np.count_nonzero(valid),
                    ]
                )

                print(
                    f"Pearson r("
                    f"{variable_name}, "
                    f"{error_name}"
                    f") = {correlation:.5f} "
                    f"(N={np.count_nonzero(valid):,})"
                )

    print(f"Saved: {path}")


def plot_resolution_profiles(
    dependencies,
    metrics,
    *,
    minimum_count,
    quantile,
    output_dir,
    output_file,
    dpi,
):
    """Plot resolution versus Wall, ToWall, and true energy."""
    fig, axes = plt.subplots(
        len(dependencies),
        len(metrics),
        figsize=(22, 15),
    )

    csv_rows = []

    for row, dependency in enumerate(dependencies):
        (
            xlabel,
            x_values,
            edges,
            variable_name,
        ) = dependency

        for column, metric in enumerate(metrics):
            (
                ylabel,
                error_values,
                color,
                metric_name,
            ) = metric

            (
                centres,
                resolutions,
                counts,
            ) = binned_quantile(
                x_values,
                error_values,
                edges,
                quantile=quantile,
                minimum_count=minimum_count,
            )

            valid = np.isfinite(resolutions)

            axis = axes[row, column]

            axis.plot(
                centres[valid],
                resolutions[valid],
                "o-",
                color=color,
                linewidth=1.8,
                markersize=4,
            )

            for (
                x_point,
                y_point,
                count,
            ) in zip(
                centres[valid],
                resolutions[valid],
                counts[valid],
                strict=True,
            ):
                axis.annotate(
                    f"{count:,}",
                    xy=(x_point, y_point),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="dimgray",
                )

            axis.margins(y=0.18)
            axis.set(
                xlabel=xlabel,
                ylabel=ylabel,
                title=(f"{ylabel.split(' [')[0]} vs {variable_name}"),
            )
            axis.grid(alpha=0.3)

            for index in range(len(centres)):
                csv_rows.append(
                    {
                        "variable": variable_name,
                        "bin_low": edges[index],
                        "bin_high": edges[index + 1],
                        "bin_center": centres[index],
                        "metric": metric_name,
                        "resolution_quantile": quantile,
                        "resolution": resolutions[index],
                        "count": counts[index],
                        "minimum_count": minimum_count,
                    }
                )

    fig.suptitle(
        "Reconstruction resolution versus event geometry and energy",
        fontsize=17,
    )
    fig.tight_layout()

    save(
        fig,
        output_path(
            output_dir,
            output_file,
        ),
        dpi,
    )

    save_resolution_csv(
        csv_rows,
        output_dir,
    )


def plot_error_boxplots(
    variable_name,
    xlabel,
    x_values,
    edges,
    boxplot_metrics,
    *,
    minimum_count,
    output_dir,
    dpi,
):
    """Plot binned error distributions using boxplots."""
    fig, axes = plt.subplots(
        1,
        len(boxplot_metrics),
        figsize=(22, 5.5),
    )

    bin_widths = np.diff(edges)
    box_width = 0.65 * np.median(bin_widths)

    for axis, metric in zip(
        axes,
        boxplot_metrics,
        strict=True,
    ):
        (
            ylabel,
            values,
            color,
            metric_name,
        ) = metric

        (
            groups,
            positions,
            counts,
        ) = collect_binned_values(
            x_values,
            values,
            edges,
            minimum_count=minimum_count,
        )

        if not groups:
            axis.text(
                0.5,
                0.5,
                "No bins satisfy the minimum count",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        else:
            boxplot = axis.boxplot(
                groups,
                positions=positions,
                widths=box_width,
                showfliers=False,
                patch_artist=True,
                manage_ticks=False,
                whis=1.5,
            )

            for box in boxplot["boxes"]:
                box.set(
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.25,
                )

            for median in boxplot["medians"]:
                median.set(
                    color=color,
                    linewidth=1.8,
                )

            for whisker in boxplot["whiskers"]:
                whisker.set(
                    color=color,
                    linewidth=1.0,
                )

            for cap in boxplot["caps"]:
                cap.set(
                    color=color,
                    linewidth=1.0,
                )

            for position, group, count in zip(
                positions,
                groups,
                counts,
                strict=True,
            ):
                q75 = np.quantile(
                    group,
                    0.75,
                )

                axis.annotate(
                    f"N={count:,}",
                    xy=(position, q75),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color="dimgray",
                )

        axis.set(
            xlabel=xlabel,
            ylabel=ylabel,
            title=metric_name,
        )
        axis.grid(
            axis="y",
            alpha=0.3,
        )

        if np.any(np.isfinite(values) & (values < 0)):
            axis.axhline(
                0.0,
                color="black",
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
            )

    fig.suptitle(
        f"Error distributions versus {variable_name}",
        fontsize=16,
    )
    fig.text(
        0.5,
        0.01,
        (
            "Boxes show the interquartile range. "
            "Whiskers extend to 1.5 times the IQR. "
            "Outliers are hidden."
        ),
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.95))

    save(
        fig,
        output_path(
            output_dir,
            Path(f"error_boxplots_vs_{variable_name}.png"),
        ),
        dpi,
    )


def build_parser():
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot reconstruction resolution versus true Wall, ToWall, and energy."
        )
    )

    for task in (
        "vertex",
        "direction",
        "energy",
    ):
        parser.add_argument(
            f"--{task}-run",
            type=Path,
            required=True,
        )
        parser.add_argument(
            f"--{task}-predictions",
            type=Path,
            default=Path("outputs/preds.npy"),
        )
        parser.add_argument(f"--{task}-predictions-variable")
        parser.add_argument(
            f"--{task}-targets",
            type=Path,
            default=Path("outputs/targets.npy"),
        )
        parser.add_argument(f"--{task}-targets-variable")
        parser.add_argument(
            f"--{task}-indices",
            type=Path,
            default=Path("outputs/indices.npy"),
        )
        parser.add_argument(f"--{task}-indices-variable")

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/resolution_dependencies"),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("resolution_vs_wall_towall_energy.png"),
    )

    parser.add_argument(
        "--vertex-minimum",
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
        "--vertex-maximum",
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
        "--energy-minimum",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--energy-maximum",
        type=float,
        default=2000.0,
    )
    parser.add_argument(
        "--vertex-normalized-minimum",
        nargs=3,
        type=float,
        default=[0, 0, 0],
    )
    parser.add_argument(
        "--vertex-normalized-maximum",
        nargs=3,
        type=float,
        default=[1, 1, 1],
    )
    parser.add_argument("--energy-normalized-minimum", type=float, default=0.0)
    parser.add_argument("--energy-normalized-maximum", type=float, default=1.0)
    parser.add_argument("--direction-normalized-minimum", type=float, default=0.0)
    parser.add_argument("--direction-normalized-maximum", type=float, default=1.0)
    parser.add_argument(
        "--inverse-direction-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
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
        "--wall-bin-width",
        type=float,
        default=50.0,
    )
    parser.add_argument(
        "--towall-bin-width",
        type=float,
        default=50.0,
    )
    parser.add_argument(
        "--energy-bin-width",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--minimum-count",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.68,
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
    )

    return parser


def main():
    """Generate resolution profiles, CSV files, and boxplots."""
    args = build_parser().parse_args()

    if args.minimum_count <= 0:
        raise ValueError("--minimum-count must be positive.")

    if not 0.0 < args.quantile < 1.0:
        raise ValueError("--quantile must be between 0 and 1.")

    if args.tank_radius <= 0:
        raise ValueError("--tank-radius must be positive.")

    if args.tank_half_height <= 0:
        raise ValueError("--tank-half-height must be positive.")

    if args.energy_maximum <= args.energy_minimum:
        raise ValueError("--energy-maximum must be greater than --energy-minimum.")
    if args.energy_normalized_maximum <= args.energy_normalized_minimum:
        raise ValueError(
            "--energy-normalized-maximum must be greater than "
            "--energy-normalized-minimum."
        )
    if args.direction_normalized_maximum <= args.direction_normalized_minimum:
        raise ValueError(
            "--direction-normalized-maximum must be greater than "
            "--direction-normalized-minimum."
        )

    vertex_outputs = load_outputs(
        args.vertex_run,
        args.vertex_predictions,
        args.vertex_targets,
        args.vertex_indices,
        prediction_variable=args.vertex_predictions_variable,
        target_variable=args.vertex_targets_variable,
        indices_variable=args.vertex_indices_variable,
    )
    direction_outputs = load_outputs(
        args.direction_run,
        args.direction_predictions,
        args.direction_targets,
        args.direction_indices,
        prediction_variable=args.direction_predictions_variable,
        target_variable=args.direction_targets_variable,
        indices_variable=args.direction_indices_variable,
    )
    energy_outputs = load_outputs(
        args.energy_run,
        args.energy_predictions,
        args.energy_targets,
        args.energy_indices,
        prediction_variable=args.energy_predictions_variable,
        target_variable=args.energy_targets_variable,
        indices_variable=args.energy_indices_variable,
    )

    assert_aligned(
        vertex_outputs,
        direction_outputs,
        energy_outputs,
        require_indices=True,
    )

    vertex_predictions = inverse_minmax(
        xyz(vertex_outputs.predictions),
        args.vertex_minimum,
        args.vertex_maximum,
        source_min=args.vertex_normalized_minimum,
        source_max=args.vertex_normalized_maximum,
    )
    vertex_targets = inverse_minmax(
        xyz(vertex_outputs.targets),
        args.vertex_minimum,
        args.vertex_maximum,
        source_min=args.vertex_normalized_minimum,
        source_max=args.vertex_normalized_maximum,
    )

    direction_predictions = xyz(direction_outputs.predictions)
    direction_targets = xyz(direction_outputs.targets)

    if args.inverse_direction_norm:
        direction_predictions = inverse_minmax(
            direction_predictions,
            -1.0,
            1.0,
            source_min=args.direction_normalized_minimum,
            source_max=args.direction_normalized_maximum,
        )
        direction_targets = inverse_minmax(
            direction_targets,
            -1.0,
            1.0,
            source_min=args.direction_normalized_minimum,
            source_max=args.direction_normalized_maximum,
        )

    energy_predictions = inverse_minmax(
        scalar(
            energy_outputs.predictions,
            "energy predictions",
        ),
        args.energy_minimum,
        args.energy_maximum,
        source_min=args.energy_normalized_minimum,
        source_max=args.energy_normalized_maximum,
    )
    energy_targets = inverse_minmax(
        scalar(
            energy_outputs.targets,
            "energy targets",
        ),
        args.energy_minimum,
        args.energy_maximum,
        source_min=args.energy_normalized_minimum,
        source_max=args.energy_normalized_maximum,
    )

    wall_values = wall(
        vertex_targets,
        radius=args.tank_radius,
        half_height=args.tank_half_height,
    )
    towall_values = calculate_towall(
        vertex_targets,
        direction_targets,
        radius=args.tank_radius,
        half_height=args.tank_half_height,
    )
    true_energy = energy_targets

    vertex_error = np.linalg.norm(
        vertex_predictions - vertex_targets,
        axis=1,
    )
    direction_error = calculate_opening_angles(
        direction_predictions,
        direction_targets,
    )
    energy_error = energy_predictions - energy_targets
    relative_energy_error = calculate_relative_energy_error(
        energy_predictions,
        energy_targets,
    )

    bin_widths = {
        "--wall-bin-width": args.wall_bin_width,
        "--towall-bin-width": args.towall_bin_width,
        "--energy-bin-width": args.energy_bin_width,
    }

    for option, width in bin_widths.items():
        if width <= 0:
            raise ValueError(f"{option} must be positive.")

    vertex_minimum = np.asarray(
        args.vertex_minimum,
        dtype=float,
    )
    vertex_maximum = np.asarray(
        args.vertex_maximum,
        dtype=float,
    )

    if np.any(vertex_maximum <= vertex_minimum):
        raise ValueError(
            "Each value in --vertex-maximum must be greater "
            "than the corresponding value in --vertex-minimum."
        )

    print()
    print("========== Resolution dependency summary ==========")
    print(f"Number of aligned events: {len(vertex_targets):,}")
    print(f"Finite Wall events: {np.count_nonzero(np.isfinite(wall_values)):,}")
    print(f"Finite ToWall events: {np.count_nonzero(np.isfinite(towall_values)):,}")
    print(f"Finite angular errors: {np.count_nonzero(np.isfinite(direction_error)):,}")
    print(
        f"Finite relative energy errors: "
        f"{np.count_nonzero(np.isfinite(relative_energy_error)):,}"
    )
    print("===================================================")
    print()

    dependencies = [
        (
            "True Wall [cm]",
            wall_values,
            make_edges(
                wall_values,
                args.wall_bin_width,
            ),
            "wall",
        ),
        (
            "True ToWall [cm]",
            towall_values,
            make_edges(
                towall_values,
                args.towall_bin_width,
                lower=0.0,
            ),
            "towall",
        ),
        (
            "True energy [MeV]",
            true_energy,
            make_edges(
                true_energy,
                args.energy_bin_width,
                lower=0.0,
            ),
            "energy",
        ),
    ]

    resolution_metrics = [
        (
            "Vertex resolution [cm]",
            vertex_error,
            "tab:blue",
            "vertex_error",
        ),
        (
            "Angular resolution [deg]",
            direction_error,
            "tab:green",
            "direction_error",
        ),
        (
            "Energy resolution [MeV]",
            energy_error,
            "tab:orange",
            "energy_error",
        ),
        (
            "Relative energy resolution [%]",
            relative_energy_error,
            "tab:red",
            "relative_energy_error",
        ),
    ]

    boxplot_metrics = [
        (
            "Vertex error [cm]",
            vertex_error,
            "tab:blue",
            "Vertex error",
        ),
        (
            "Opening angle [deg]",
            direction_error,
            "tab:green",
            "Opening angle",
        ),
        (
            "Energy residual [MeV]",
            energy_error,
            "tab:orange",
            "Energy residual",
        ),
        (
            "Relative energy residual [%]",
            relative_energy_error,
            "tab:red",
            "Relative energy residual",
        ),
    ]

    plot_resolution_profiles(
        dependencies,
        resolution_metrics,
        minimum_count=args.minimum_count,
        quantile=args.quantile,
        output_dir=args.output_dir,
        output_file=args.output_file,
        dpi=args.dpi,
    )

    for (
        xlabel,
        x_values,
        edges,
        variable_name,
    ) in dependencies:
        plot_error_boxplots(
            variable_name,
            xlabel,
            x_values,
            edges,
            boxplot_metrics,
            minimum_count=args.minimum_count,
            output_dir=args.output_dir,
            dpi=args.dpi,
        )

    correlation_variables = [
        (
            "wall",
            wall_values,
        ),
        (
            "towall",
            towall_values,
        ),
        (
            "true_energy",
            true_energy,
        ),
    ]

    correlation_errors = [
        (
            "vertex_error",
            vertex_error,
        ),
        (
            "direction_error",
            direction_error,
        ),
        (
            "abs_energy_error",
            np.abs(energy_error),
        ),
        (
            "abs_relative_energy_error",
            np.abs(relative_energy_error),
        ),
    ]

    save_correlation_csv(
        correlation_variables,
        correlation_errors,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
