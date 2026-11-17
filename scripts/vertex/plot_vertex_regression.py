"""Generate vertex-regression performance plots."""

import argparse
from pathlib import Path

from analysis_tools import plotting
from analysis_tools.analysis import analyze_vertex_regression
from analysis_tools.data import as_vector_array
from analysis_tools.io import load_analysis_config, load_array_pair
from analysis_tools.metrics import make_bin_edges
from analysis_tools.transforms import MinMaxTransform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate vertex-regression performance plots.",
    )
    parser.add_argument(
        "--sample-config",
        type=Path,
        required=True,
        help="Path to the sample configuration YAML file.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the vertex analysis configuration YAML file.",
    )
    return parser.parse_args()


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    path = Path(value).expanduser()

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def make_edges(config: dict[str, float]):
    """Create bin edges from a configuration mapping."""
    return make_bin_edges(
        minimum=config["minimum"],
        maximum=config["maximum"],
        width=config["width"],
    )


def main() -> None:
    args = parse_args()

    sample_config_path = args.sample_config.expanduser().resolve()
    analysis_config_path = args.config.expanduser().resolve()

    config = load_analysis_config(
        sample_config_path,
        analysis_config_path,
    )

    predictions_path = resolve_path(config["input"]["predictions"])
    targets_path = resolve_path(config["input"]["targets"])
    output_directory = resolve_path(config["output"]["directory"])

    predictions, targets = load_array_pair(
        predictions_path,
        targets_path,
        first_name="predictions",
        second_name="targets",
    )

    predictions = as_vector_array(
        predictions,
        size=3,
        name="predictions",
    )
    targets = as_vector_array(
        targets,
        size=3,
        name="targets",
    )

    normalization = config["normalization"]["vertex"]
    transform = MinMaxTransform(
        minimum=normalization["minimum"],
        maximum=normalization["maximum"],
    )

    predictions_cm = transform.inverse(predictions)
    targets_cm = transform.inverse(targets)

    detector = config["detector"]
    result = analyze_vertex_regression(
        predictions_cm,
        targets_cm,
        detector_radius=detector["radius_cm"],
        detector_half_height=detector["half_height_cm"],
    )

    plot_config = config["plot"]
    output_config = config["output"]
    filenames = output_config["filenames"]
    dpi = output_config["dpi"]

    errors_config = plot_config["errors"]
    errors_figure, _ = plotting.plot_vertex_error_summary(
        result,
        bins=errors_config["bins"],
        figsize=tuple(errors_config["figsize"]),
    )
    plotting.save_figure(
        errors_figure,
        output_directory / filenames["errors"],
        dpi=dpi,
        close=True,
    )

    prediction_config = plot_config["prediction"]
    prediction_figure, _ = plotting.plot_vertex_prediction(
        result,
        figsize=tuple(prediction_config["figsize"]),
        color=prediction_config["color"],
        alpha=prediction_config["alpha"],
        marker_size=prediction_config["marker_size"],
    )
    plotting.save_figure(
        prediction_figure,
        output_directory / filenames["prediction"],
        dpi=dpi,
        close=True,
    )

    profiles_config = plot_config["profiles"]
    profiles_figure, _ = plotting.plot_vertex_profiles(
        result,
        radius_edges=make_edges(profiles_config["radius_edges"]),
        z_edges=make_edges(profiles_config["z_edges"]),
        wall_edges=make_edges(profiles_config["wall_edges"]),
        quantile_level=profiles_config["quantile"],
        minimum_events=profiles_config["minimum_events"],
        figsize=tuple(profiles_config["figsize"]),
        color=profiles_config["color"],
        marker=profiles_config["marker"],
    )
    plotting.save_figure(
        profiles_figure,
        output_directory / filenames["profiles"],
        dpi=dpi,
        close=True,
    )

    print(f"Analysis: {config['name']}")
    print(f"Events: {result.event_count:,}")
    print(f"Removed non-finite events: {result.removed_event_count:,}")
    print(f"Saved: {output_directory / filenames['errors']}")
    print(f"Saved: {output_directory / filenames['prediction']}")
    print(f"Saved: {output_directory / filenames['profiles']}")


if __name__ == "__main__":
    main()
