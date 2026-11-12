import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.cli.plot_resolution_dependencies import (
    binned_quantile,
    calculate_opening_angles,
    calculate_relative_energy_error,
    calculate_towall,
    make_edges,
)
from analysis.geometry import wall
from analysis.io import assert_aligned, load_outputs
from analysis.paths import output_path
from analysis.plotting import save
from analysis.regression import inverse_minmax, scalar, xyz


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compare resolution dependencies for multiple vertex, direction, "
            "and energy result sets."
        )
    )
    parser.add_argument("--label", action="append", required=True)
    for task in ("vertex", "direction", "energy"):
        parser.add_argument(f"--{task}-run", action="append", required=True)
        parser.add_argument(f"--{task}-predictions", action="append")
        parser.add_argument(f"--{task}-targets", action="append")
        parser.add_argument(f"--{task}-indices", action="append")
        parser.add_argument(f"--{task}-predictions-variable", action="append")
        parser.add_argument(f"--{task}-targets-variable", action="append")
        parser.add_argument(f"--{task}-indices-variable", action="append")

    parser.add_argument(
        "--output-dir", type=Path, default=Path("plots/resolution-comparison")
    )
    parser.add_argument(
        "--output-file", type=Path, default=Path("resolution_comparison.png")
    )
    parser.add_argument(
        "--vertex-minimum", nargs=3, type=float, default=[-1690, -1690, -1810]
    )
    parser.add_argument(
        "--vertex-maximum", nargs=3, type=float, default=[1690, 1690, 1810]
    )
    parser.add_argument("--energy-minimum", type=float, default=0.5)
    parser.add_argument("--energy-maximum", type=float, default=2000.0)
    parser.add_argument(
        "--vertex-normalized-minimum", nargs=3, type=float, default=[0, 0, 0]
    )
    parser.add_argument(
        "--vertex-normalized-maximum", nargs=3, type=float, default=[1, 1, 1]
    )
    parser.add_argument("--energy-normalized-minimum", type=float, default=0.0)
    parser.add_argument("--energy-normalized-maximum", type=float, default=1.0)
    parser.add_argument("--direction-normalized-minimum", type=float, default=0.0)
    parser.add_argument("--direction-normalized-maximum", type=float, default=1.0)
    parser.add_argument(
        "--inverse-direction-norm", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--tank-radius", type=float, default=1690.0)
    parser.add_argument("--tank-half-height", type=float, default=1810.0)
    parser.add_argument("--wall-bin-width", type=float, default=50.0)
    parser.add_argument("--towall-bin-width", type=float, default=50.0)
    parser.add_argument("--energy-bin-width", type=float, default=100.0)
    parser.add_argument("--minimum-count", type=int, default=20)
    parser.add_argument("--quantile", type=float, default=0.68)
    parser.add_argument("--dpi", type=int, default=200)
    return parser


def _per_group(values, count, name, default):
    if not values:
        return [default] * count
    if len(values) != count:
        raise ValueError(f"{name} must be specified once per comparison group")
    return values


def _load_group(args, index):
    outputs = {}
    for task in ("vertex", "direction", "energy"):
        predictions = _per_group(
            getattr(args, f"{task}_predictions"),
            len(args.label),
            f"--{task}-predictions",
            Path("outputs/preds.npy"),
        )[index]
        targets = _per_group(
            getattr(args, f"{task}_targets"),
            len(args.label),
            f"--{task}-targets",
            Path("outputs/targets.npy"),
        )[index]
        indices = _per_group(
            getattr(args, f"{task}_indices"),
            len(args.label),
            f"--{task}-indices",
            Path("outputs/indices.npy"),
        )[index]
        prediction_variable = _per_group(
            getattr(args, f"{task}_predictions_variable"),
            len(args.label),
            f"--{task}-predictions-variable",
            None,
        )[index]
        target_variable = _per_group(
            getattr(args, f"{task}_targets_variable"),
            len(args.label),
            f"--{task}-targets-variable",
            None,
        )[index]
        indices_variable = _per_group(
            getattr(args, f"{task}_indices_variable"),
            len(args.label),
            f"--{task}-indices-variable",
            None,
        )[index]
        outputs[task] = load_outputs(
            getattr(args, f"{task}_run")[index],
            predictions,
            targets,
            indices,
            require_indices=True,
            prediction_variable=prediction_variable,
            target_variable=target_variable,
            indices_variable=indices_variable,
        )
    assert_aligned(
        outputs["vertex"], outputs["direction"], outputs["energy"], require_indices=True
    )
    return outputs


def _calculate_group(args, outputs):
    vertex_predictions = inverse_minmax(
        xyz(outputs["vertex"].predictions),
        args.vertex_minimum,
        args.vertex_maximum,
        source_min=args.vertex_normalized_minimum,
        source_max=args.vertex_normalized_maximum,
    )
    vertex_targets = inverse_minmax(
        xyz(outputs["vertex"].targets),
        args.vertex_minimum,
        args.vertex_maximum,
        source_min=args.vertex_normalized_minimum,
        source_max=args.vertex_normalized_maximum,
    )
    direction_predictions = xyz(outputs["direction"].predictions)
    direction_targets = xyz(outputs["direction"].targets)
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
        scalar(outputs["energy"].predictions),
        args.energy_minimum,
        args.energy_maximum,
        source_min=args.energy_normalized_minimum,
        source_max=args.energy_normalized_maximum,
    )
    energy_targets = inverse_minmax(
        scalar(outputs["energy"].targets),
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
    return {
        "wall": wall_values,
        "towall": towall_values,
        "energy": energy_targets,
        "vertex_error": np.linalg.norm(vertex_predictions - vertex_targets, axis=1),
        "direction_error": calculate_opening_angles(
            direction_predictions,
            direction_targets,
        ),
        "energy_error": energy_predictions - energy_targets,
        "relative_energy_error": calculate_relative_energy_error(
            energy_predictions,
            energy_targets,
        ),
    }


def main():
    args = build_parser().parse_args()
    count = len(args.label)
    if count < 2:
        raise ValueError("at least two --label options are required")
    for task in ("vertex", "direction", "energy"):
        if len(getattr(args, f"{task}_run")) != count:
            raise ValueError(f"--{task}-run must be specified once per label")
    if args.minimum_count <= 0 or not 0 < args.quantile < 1:
        raise ValueError(
            "--minimum-count must be positive and --quantile must be in (0, 1)"
        )

    groups = [
        _calculate_group(args, _load_group(args, index)) for index in range(count)
    ]
    dependency_specs = (
        ("wall", "True Wall [cm]", args.wall_bin_width),
        ("towall", "True ToWall [cm]", args.towall_bin_width),
        ("energy", "True energy [MeV]", args.energy_bin_width),
    )
    metric_specs = (
        ("vertex_error", "Vertex resolution [cm]"),
        ("direction_error", "Angular resolution [deg]"),
        ("energy_error", "Energy resolution [MeV]"),
        ("relative_energy_error", "Relative energy resolution [%]"),
    )
    figure, axes = plt.subplots(3, 4, figsize=(24, 15), squeeze=False)
    for row, (dependency, xlabel, width) in enumerate(dependency_specs):
        edges = make_edges(
            np.concatenate([group[dependency] for group in groups]),
            width,
            lower=0.0 if dependency != "wall" else None,
        )
        for column, (metric, ylabel) in enumerate(metric_specs):
            axis = axes[row, column]
            for label, group in zip(args.label, groups, strict=True):
                centres, values, counts = binned_quantile(
                    group[dependency],
                    group[metric],
                    edges,
                    quantile=args.quantile,
                    minimum_count=args.minimum_count,
                )
                valid = np.isfinite(values)
                axis.plot(centres[valid], values[valid], "o-", label=label)
            axis.set(xlabel=xlabel, ylabel=ylabel, title=f"{ylabel} vs {dependency}")
            axis.grid(alpha=0.3)
            axis.legend()
    figure.suptitle("Resolution dependencies comparison")
    figure.tight_layout()
    save(figure, output_path(args.output_dir, args.output_file), args.dpi)


if __name__ == "__main__":
    main()
