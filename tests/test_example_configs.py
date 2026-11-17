from pathlib import Path

from analysis_tools.io import load_analysis_config, load_yaml

PROJECT_ROOT = Path(__file__).parents[1]

SAMPLE_CONFIG = PROJECT_ROOT / "configs/samples/example.yaml"
LOSS_CONFIG = PROJECT_ROOT / "configs/loss/example.yaml"


def test_sample_example_config() -> None:
    config = load_yaml(SAMPLE_CONFIG)

    assert config["sample"]["name"] == "example_sample"

    assert config["detector"]["radius_cm"] > 0.0
    assert config["detector"]["half_height_cm"] > 0.0

    assert len(config["normalization"]["vertex"]["minimum"]) == 3
    assert len(config["normalization"]["vertex"]["maximum"]) == 3

    assert len(config["normalization"]["direction"]["minimum"]) == 3
    assert len(config["normalization"]["direction"]["maximum"]) == 3

    assert (
        config["normalization"]["energy"]["maximum"]
        > config["normalization"]["energy"]["minimum"]
    )


def test_loss_example_config() -> None:
    config = load_yaml(LOSS_CONFIG)

    assert config["name"] == "example_loss"
    assert isinstance(config["watchmal_root"], str)
    assert isinstance(config["run_directory"], str)

    assert config["output"]["path"]
    assert config["output"]["dpi"] > 0

    assert len(config["plot"]["figsize"]) == 2
    assert len(config["plot"]["zoom_percentiles"]) == 2

    lower, upper = config["plot"]["zoom_percentiles"]
    assert 0.0 <= lower < upper <= 100.0


def test_sample_and_loss_configs_can_be_merged() -> None:
    config = load_analysis_config(
        SAMPLE_CONFIG,
        LOSS_CONFIG,
    )

    assert config["sample"]["name"] == "example_sample"
    assert config["name"] == "example_loss"
    assert config["normalization"]["vertex"]["minimum"] == [
        -1690.0,
        -1690.0,
        -1810.0,
    ]


VERTEX_CONFIG = PROJECT_ROOT / "configs/vertex/example.yaml"


def test_vertex_example_config() -> None:
    config = load_yaml(VERTEX_CONFIG)

    assert config["name"] == "example_vertex"

    assert isinstance(config["input"]["predictions"], str)
    assert isinstance(config["input"]["targets"], str)

    output = config["output"]
    assert output["directory"]
    assert output["dpi"] > 0
    assert output["filenames"]["errors"]
    assert output["filenames"]["prediction"]
    assert output["filenames"]["profiles"]

    plot = config["plot"]

    errors = plot["errors"]
    assert errors["bins"] > 0
    assert len(errors["figsize"]) == 2
    assert all(value > 0.0 for value in errors["figsize"])

    prediction = plot["prediction"]
    assert len(prediction["figsize"]) == 2
    assert prediction["color"]
    assert 0.0 <= prediction["alpha"] <= 1.0
    assert prediction["marker_size"] > 0.0

    profiles = plot["profiles"]
    assert len(profiles["figsize"]) == 2
    assert 0.0 <= profiles["quantile"] <= 1.0
    assert profiles["minimum_events"] >= 1
    assert profiles["color"]
    assert profiles["marker"]

    for edge_name in (
        "radius_edges",
        "z_edges",
        "wall_edges",
    ):
        edge_config = profiles[edge_name]

        assert edge_config["maximum"] > edge_config["minimum"]
        assert edge_config["width"] > 0.0


def test_sample_and_vertex_configs_can_be_merged() -> None:
    config = load_analysis_config(
        SAMPLE_CONFIG,
        VERTEX_CONFIG,
    )

    assert config["sample"]["name"] == "example_sample"
    assert config["name"] == "example_vertex"

    vertex_normalization = config["normalization"]["vertex"]

    assert vertex_normalization["minimum"] == [
        -1690.0,
        -1690.0,
        -1810.0,
    ]
    assert vertex_normalization["maximum"] == [
        1690.0,
        1690.0,
        1810.0,
    ]

    detector = config["detector"]

    assert detector["radius_cm"] == 1690.0
    assert detector["half_height_cm"] == 1810.0
