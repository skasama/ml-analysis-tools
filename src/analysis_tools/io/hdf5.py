from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray


def list_hdf5_datasets(path: str | Path) -> list[str]:
    """Return all dataset paths in an HDF5 file."""
    path = _validate_hdf5_path(path)
    datasets: list[str] = []
    with h5py.File(path, "r") as file:

        def collect(name: str, item: h5py.Group | h5py.Dataset) -> None:
            if isinstance(item, h5py.Dataset):
                datasets.append(name)

        file.visititems(collect)

    return datasets


def load_hdf5(
    path: str | Path,
    keys: Sequence[str],
) -> dict[str, NDArray]:
    """Load selected datasets from an HDF5 file."""
    path = _validate_hdf5_path(path)
    if not keys:
        raise ValueError("At least one dataset key is required.")

    arrays: dict[str, NDArray] = {}

    with h5py.File(path, "r") as file:
        missing = {key for key in keys if key not in file}
        if missing:
            available = list_hdf5_datasets(path)
            raise KeyError(
                f"Missing datasets in {path}: {missing}. "
                f"Available datasets: {available}."
            )

        for key in keys:
            item = file[key]

            if not isinstance(item, h5py.Dataset):
                raise TypeError(f"{key!r} is not a dataset.")

            arrays[key] = np.asarray(item)

        return arrays


def _validate_hdf5_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)

    if path.suffix.lower() not in (".h5", ".hdf5", ".hy"):
        raise ValueError(f"Unsupported HDF5 file extension: {path}")

    return path
