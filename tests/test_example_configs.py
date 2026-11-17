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
