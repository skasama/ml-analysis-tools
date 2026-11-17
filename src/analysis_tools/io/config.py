from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

Config = dict[str, Any]


def load_yaml(path: str | Path) -> Config:
    """

    Load a YAML configuration file.

    Parameters
    ----------
    path : str or Path
        The path to the YAML file to load.

    Returns
    -------
    Config
        The loaded configuration as a dictionary.
    """
    path = Path(path).expanduser()

    if not path.is_file():
        raise FileNotFoundError(path)

    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"Expected a YAML file: {path}")

    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if config is None:
        return {}

    if not isinstance(config, dict):
        raise TypeError("The top-level YAML value must be a mapping.")

    return config


def merge_configs(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> Config:
    """

    Recursively merge two configuration mappings.


    Parameters
    ----------
    base : Mapping[str, Any]
        The base configuration.
    override : Mapping[str, Any]
        The override configuration.

    Returns
    -------
    Config
        The merged configuration.
    """
    merged = deepcopy(dict(base))

    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = deepcopy(value)

    return merged


def load_analysis_config(
    common_path: str | Path,
    run_path: str | Path,
) -> Config:
    """

    Load and merge common and run-specific configurations.

    Parameters
    ----------
    common_path : str or Path
        The path to the common configuration file.
    run_path : str or Path
        The path to the run-specific configuration file.

    Returns
    -------
    Config
        The merged configuration.
    """
    common = load_yaml(common_path)
    run = load_yaml(run_path)

    return merge_configs(common, run)


def resolve_run_path(
    config: Mapping[str, Any],
    relative_path: str | Path,
) -> Path:
    """

    Resolve an input path relative to the run directory.

    Parameters
    ----------
    config : Mapping[str, Any]
        The run configuration.
    relative_path : str or Path
        The relative path to resolve.

    Returns
    -------
    Path
        The resolved absolute path.
    """
    if "directory" not in config:
        raise KeyError("Run configuration does not define 'directory'.")

    path = Path(relative_path).expanduser()

    if path.is_absolute():
        return path

    return Path(config["directory"]).expanduser() / path
