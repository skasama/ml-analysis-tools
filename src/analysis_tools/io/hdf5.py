from collections.abc import Mapping, Sequence
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import ArrayLike, NDArray

from analysis_tools.data.arrays import as_integer_array


def list_hdf5_datasets(path: str | Path) -> list[str]:
    """
    List all dataset paths in an HDF5 file.
    Return all dataset paths in an HDF5 file.

    Parameters
    ----------
    path : str | Path
        Path to the HDF5 file.

    Returns
    -------
    list[str]
        A list of dataset paths in the HDF5 file.
    """
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
    *,
    indices: ArrayLike | None = None,
) -> dict[str, NDArray]:
    """

    Load selected datasets from an HDF5 file.

    Parameters
    ----------
    path : str | Path
        Path to the HDF5 file.
    keys : Sequence[str]
        A sequence of dataset keys to be loaded from the HDF5 file.
    indices
        Optional source-event indices. When provided, each selected dataset
        is loaded in the requested row order.

    Returns
    -------
    dict[str, NDArray]
        A dictionary mapping dataset keys to their corresponding NumPy arrays.

    Raises
    ------
    FileNotFoundError
        If the specified HDF5 file does not exist.
    KeyError
        If any of the specified keys are not found in the HDF5 file.
    TypeError
        If any of the specified keys do not correspond to datasets in the HDF5 file.
    """
    path = _validate_hdf5_path(path)
    if not keys:
        raise ValueError("At least one dataset key is required.")

    selected_indices = None

    if indices is not None:
        selected_indices = as_integer_array(
            indices,
            name="indices",
        )

        if np.any(selected_indices < 0):
            raise ValueError("indices must be non-negative.")

    arrays: dict[str, NDArray] = {}
    source_event_count: int | None = None

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

            values = np.asarray(item)

            if selected_indices is not None:
                if values.ndim == 0:
                    raise ValueError(f"{key!r} must have an event dimension.")

                if source_event_count is None:
                    source_event_count = len(values)
                elif len(values) != source_event_count:
                    raise ValueError(
                        "Selected HDF5 datasets must have the same event count: "
                        f"{key!r} has {len(values)}, expected "
                        f"{source_event_count}."
                    )

                if np.any(selected_indices >= len(values)):
                    raise IndexError(
                        f"indices exceed the event count for {key!r}: {len(values)}."
                    )

                values = values[selected_indices]

            arrays[key] = values

        return arrays


def _validate_hdf5_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)

    if path.suffix.lower() not in (".h5", ".hdf5", ".hy"):
        raise ValueError(f"Unsupported HDF5 file extension: {path}")

    return path


def save_hdf5(
    path: str | Path,
    arrays: Mapping[str, ArrayLike],
    *,
    compression: str | None = "gzip",
    overwrite: bool = False,
    create_parents: bool = True,
) -> Path:
    """

    Save named arrays to an HDF5 file.

    Parameters
    ----------
    path : str | Path
        Path to the HDF5 file.
    arrays : Mapping[str, ArrayLike]
        A mapping of names to arrays to save.
    compression : str | None, optional
        Compression strategy to use for the datasets. Default is "gzip".
    overwrite : bool, optional
        Whether to overwrite the file if it already exists. Default is False.
    create_parents : bool, optional
        Whether to create parent directories if they don't exist. Default is True.

    Returns
    -------
    Path
        The path to the saved HDF5 file.
    """
    path = Path(path).expanduser()

    if path.suffix.lower() not in {".h5", ".hdf5", ".hy"}:
        raise ValueError(f"Unsupported HDF5 file extension: {path}")

    if not arrays:
        raise ValueError("At least one array is required.")

    if path.exists() and not overwrite:
        raise FileExistsError(path)

    if create_parents:
        path.parent.mkdir(parents=True, exist_ok=True)
    elif not path.parent.exists():
        raise FileNotFoundError(path.parent)

    with h5py.File(path, "w") as file:
        for key, values in arrays.items():
            file.create_dataset(
                key,
                data=np.asarray(values),
                compression=compression,
            )

    return path
